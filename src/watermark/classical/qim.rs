//! Non-overlapping log-magnitude QIM.
//!
//! Each payload bit owns a fixed set of bins for the whole clip. The
//! detector votes per block: averaging dB across blocks first walks off
//! the lattice. Polarity matches the bakeoff: closer to the bit-1 lattice
//! (half a step) decodes as bit 1.
//!
//! # Detector
//!
//! The embedded format is fixed; what the detector does with the received
//! blocks is not. It
//!
//! - **tracks level.** Bit 0 sits on multiples of 6 dB and bit 1 on odd
//!   multiples of 3 dB, so every marked carrier lies on a 3 dB lattice
//!   whatever the bits are. The circular mean of `exp(2πi·dB/3)` over a
//!   block's carriers is that block's gain modulo 3 dB. Unwrapped across
//!   blocks it follows a constant or slowly moving gain, and a 3 dB shift
//!   that swaps every bit is resolved by trying both polarities against the
//!   CRC. The same statistic is independent of the bits and of the level,
//!   so it also ranks alignments.
//! - **finds the alignment to the sample.** A coarse search in steps of
//!   [`COARSE_STEP`] samples, then 8, then 1, each ranked on at most
//!   [`SCORE_BLOCKS`] evenly spaced blocks; the best few are decoded on every
//!   block.
//! - **skips what the embedder skipped.** A carrier below [`SKIP_MAG`] was
//!   never put on a lattice, so it takes no part in ranking or voting.
//!   Digital silence therefore carries no evidence, and a bit nothing voted
//!   on is an erasure rather than a 1.

use alloc::vec;
use alloc::vec::Vec;

use num_complex::Complex;
use rustfft::FftPlannerScalar;

use crate::dsp::stft::RealFft;

use super::coding::{hard_bits, unframe_either_polarity};
use super::keying::keyed_bins;
use super::stft::modulus;
use super::{Detection, N_FFT};

const Q_DB: f32 = 6.0;
const BINS_PER_BIT: usize = 4;
const BIN_LO: usize = 20;
const BIN_HI: usize = 512;
/// Step of the first alignment search, over the whole block of `N_FFT`
/// offsets. Each later stage looks one coarser step to either side of the
/// previous winner.
///
/// The statistic that ranks alignments is sharply peaked. On six real tracks
/// it was 0.6 to 1.0 at the right alignment, 0.15 to 0.4 one sample away,
/// about 0.1 four samples away and the noise floor, about 0.05, from a few
/// dozen. The coarse grid therefore decides which alignments can be found.
/// Steps of 32 rather than 64 halve the misalignment the coarse stage has to
/// see through and cost no more, because the medium stage then needs half
/// the reach: at most 124 alignments scored per clip instead of 116.
const COARSE_STEP: usize = 32;
/// Step of the second stage; the last steps by one sample.
const MEDIUM_STEP: usize = 8;
/// Best coarse alignments that are refined and decoded. Each decode is two
/// CRC trials (the two polarities), so a detection on unmarked audio is a
/// false accept with probability about `2 * CANDIDATES * 2^-32`.
const CANDIDATES: usize = 3;
/// Blocks an alignment hypothesis is ranked on. The final decode uses all.
const SCORE_BLOCKS: usize = 24;
/// Blocks on each side averaged into one block's level estimate.
const GAIN_SMOOTH: usize = 4;
/// A carrier below this magnitude was left alone by the embedder (see the
/// `mag < 1e-6` test in [`embed`]), so it carries no mark.
const SKIP_MAG: f32 = 1e-6;
/// A block whose samples all stay below this cannot hold a carrier at
/// [`SKIP_MAG`]: no bin of an `N_FFT`-point transform exceeds `N_FFT` times
/// the peak sample.
const LIVE_PEAK: f32 = SKIP_MAG / N_FFT as f32;
/// Every marked carrier lies on a lattice of this period.
const LATTICE_DB: f64 = 3.0;

/// Blocks (about 4.1 s) from which clean PCM16 audio decoded 30 times in 30
/// on ten real tracks; shorter clips decoded 27 to 29 times in 30. See
/// [`usable_min_samples`].
const USABLE_BLOCKS: usize = 32;

/// Clip length this algorithm can be expected to decode from: [`USABLE_BLOCKS`]
/// whole blocks plus one block of slack for the alignment search.
pub(crate) const fn usable_min_samples() -> usize {
    (USABLE_BLOCKS + 1) * N_FFT
}

/// True when `n_bits` can each take [`BINS_PER_BIT`] bins inside the carrier band.
pub(crate) fn fits(n_bits: usize) -> bool {
    n_bits.saturating_mul(BINS_PER_BIT) <= BIN_HI - BIN_LO
}

pub(crate) fn embed(samples: &[f32], bits: &[u8], key: &[u8]) -> Vec<f32> {
    let n_bits = bits.len().max(1);
    let order = qim_bins(key, n_bits);
    let mut planner = FftPlannerScalar::<f32>::new();
    let fwd = planner.plan_fft_forward(N_FFT);
    let inv = planner.plan_fft_inverse(N_FFT);
    let scratch_len = fwd
        .get_inplace_scratch_len()
        .max(inv.get_inplace_scratch_len());
    let mut scratch = vec![Complex::new(0.0, 0.0); scratch_len];
    let mut buf = vec![Complex::new(0.0, 0.0); N_FFT];
    let mut y = samples.to_vec();
    let mut pos = 0usize;
    while pos + N_FFT <= samples.len() {
        for i in 0..N_FFT {
            buf[i] = Complex::new(samples[pos + i], 0.0);
        }
        fwd.process_with_scratch(&mut buf, &mut scratch);
        for (b, &bit) in bits.iter().enumerate() {
            let target = if bit == 1 { Q_DB * 0.5 } else { 0.0 };
            for k in 0..BINS_PER_BIT {
                let idx = b + k * n_bits;
                if idx >= order.len() {
                    break;
                }
                let bin = order[idx];
                let mag = modulus(buf[bin]);
                if mag < 1e-6 {
                    continue;
                }
                let db = 20.0 * libm::log10f(mag);
                let new_mag = libm::powf(10.0, quantize_db(db, target, Q_DB) / 20.0).max(1e-7);
                set_herm_mag(&mut buf, bin, new_mag);
            }
        }
        inv.process_with_scratch(&mut buf, &mut scratch);
        let scale = 1.0 / N_FFT as f32;
        for i in 0..N_FFT {
            y[pos + i] = buf[i].re * scale;
        }
        pos += N_FFT;
    }
    y
}

/// Detect a payload: find the alignment, track the level, vote, check the CRC.
///
/// `score` is the mean absolute lattice vote of the best alignment after
/// level compensation, and `sync_offset` that alignment in samples. Both are
/// zero when the clip is all digital silence or no ranked block had a carrier
/// above the skip floor: there is nothing to vote on.
pub(crate) fn detect(samples: &[f32], payload_len: usize, n_bits: usize, key: &[u8]) -> Detection {
    let order = qim_bins(key, n_bits);
    if n_bits == 0 || order.len() < n_bits * BINS_PER_BIT || samples.len() < N_FFT {
        return Detection::none();
    }
    let mut search = Search::new(samples, &order);
    if !search.live.iter().any(|&live| live) {
        // Nothing but digital silence: no carrier can reach the skip floor.
        return Detection::none();
    }

    let mut ranked: Vec<(f64, usize)> = Vec::new();
    for offset in (0..N_FFT).step_by(COARSE_STEP) {
        let score = search.score(offset);
        if score.is_finite() {
            ranked.push((score, offset));
        }
    }
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    ranked.truncate(CANDIDATES);
    // A zero score means every ranked block was without a usable carrier.
    if ranked.first().is_none_or(|&(score, _)| score <= 0.0) {
        return Detection::none();
    }

    let mut candidates: Vec<(f64, usize)> = Vec::with_capacity(ranked.len());
    for &(_, coarse) in &ranked {
        let (score, offset) = search.refine(coarse);
        if candidates.iter().all(|&(_, seen)| seen != offset) {
            candidates.push((score, offset));
        }
    }
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut detection = Detection::none();
    for (rank, &(_, offset)) in candidates.iter().enumerate() {
        let decoded = search.decode(offset, payload_len, n_bits);
        if rank == 0 {
            detection.score = decoded.mean_vote;
            detection.sync_offset = offset;
        }
        if decoded.payload.is_some() {
            return Detection {
                payload: decoded.payload,
                score: decoded.mean_vote,
                sync_offset: offset,
            };
        }
    }
    detection
}

/// Marker for a carrier below [`SKIP_MAG`]: no evidence, not a vote.
const SKIPPED: f32 = f32::NAN;

/// The carrier bins' log-magnitudes, one block at a time.
struct Carriers<'a> {
    order: &'a [usize],
    // Detect consumes magnitudes of a real block only, so it uses the
    // crate's real-input FFT (half-length complex FFT + twiddle pass)
    // rather than a full complex transform: the classical fingerprinters
    // already run on this backend, and it is the target-independent one.
    fft: RealFft,
    time: Vec<f32>,
    spec: Vec<Complex<f32>>,
}

impl<'a> Carriers<'a> {
    fn new(order: &'a [usize]) -> Self {
        Self {
            order,
            fft: RealFft::new(N_FFT),
            time: vec![0.0; N_FFT],
            spec: vec![Complex::new(0.0, 0.0); N_FFT / 2 + 1],
        }
    }

    /// dB of every carrier in the block at `pos`, [`SKIPPED`] where the
    /// embedder would have left the bin alone. `out` has one slot per carrier.
    fn block_db(&mut self, samples: &[f32], pos: usize, out: &mut [f32]) {
        self.time.copy_from_slice(&samples[pos..pos + N_FFT]);
        self.fft.process(&mut self.time, &mut self.spec);
        for (slot, &bin) in out.iter_mut().zip(self.order) {
            let mag = modulus(self.spec[bin]);
            *slot = if mag < SKIP_MAG {
                SKIPPED
            } else {
                20.0 * libm::log10f(mag)
            };
        }
    }
}

/// Sum of `exp(2πi·dB/3)` over one block's usable carriers.
#[derive(Clone, Copy, Default)]
struct Phasor {
    re: f64,
    im: f64,
    count: usize,
}

impl Phasor {
    const RAD_PER_DB: f32 = core::f32::consts::TAU / LATTICE_DB as f32;

    fn of(row: &[f32]) -> Self {
        let mut sum = Self::default();
        for &db in row.iter().filter(|db| !db.is_nan()) {
            let (sin, cos) = libm::sincosf(db * Self::RAD_PER_DB);
            sum.re += f64::from(cos);
            sum.im += f64::from(sin);
            sum.count += 1;
        }
        sum
    }

    fn length(self) -> f64 {
        libm::sqrt(self.re * self.re + self.im * self.im)
    }
}

/// What one alignment decoded to.
struct Decoded {
    payload: Option<Vec<u8>>,
    /// Mean absolute vote over the bits; 0 when some bit had no evidence.
    mean_vote: f64,
}

/// Alignment search and decoding over one clip.
struct Search<'a> {
    samples: &'a [f32],
    /// Per aligned `N_FFT`-sample segment of the clip: does it hold a sample
    /// big enough for any carrier to reach [`SKIP_MAG`]?
    live: Vec<bool>,
    carriers: Carriers<'a>,
    /// Scratch: one block's carrier dB.
    row: Vec<f32>,
    /// Scratch: the blocks an alignment is ranked on.
    picked: Vec<usize>,
    /// Alignment score by offset; NaN until evaluated.
    memo: Vec<f64>,
}

impl<'a> Search<'a> {
    fn new(samples: &'a [f32], order: &'a [usize]) -> Self {
        Self {
            samples,
            live: samples
                .chunks(N_FFT)
                .map(|segment| segment.iter().any(|s| s.abs() >= LIVE_PEAK))
                .collect(),
            carriers: Carriers::new(order),
            row: vec![0.0; order.len()],
            picked: Vec::new(),
            memo: vec![f64::NAN; N_FFT],
        }
    }

    /// Blocks that start at `offset` and fit in the clip.
    fn blocks_at(&self, offset: usize) -> usize {
        (self.samples.len() - offset) / N_FFT
    }

    /// Can block `j` of the grid starting at `offset` hold any carrier? It
    /// straddles segments `j` and `j + 1` of the zero-offset grid.
    fn is_live(&self, offset: usize, j: usize) -> bool {
        self.live[j] || (offset > 0 && self.live[j + 1])
    }

    /// Alignment statistic: the mean length of the per-block phasors. On
    /// real audio it is 0.6 to 1.0 where the blocks line up with a marked
    /// clip and about 0.05 elsewhere, and it does not depend on the bits or
    /// on the level. `-inf` for an offset with no whole block.
    fn score(&mut self, offset: usize) -> f64 {
        if offset + N_FFT > self.samples.len() {
            return f64::NEG_INFINITY;
        }
        if !self.memo[offset].is_nan() {
            return self.memo[offset];
        }
        // Rank on evenly spaced blocks that can carry evidence, so a clip
        // that is mostly silence is not scored on its silence.
        self.picked.clear();
        for j in 0..self.blocks_at(offset) {
            if self.is_live(offset, j) {
                self.picked.push(j);
            }
        }
        let live = self.picked.len();
        if live > SCORE_BLOCKS {
            for i in 0..SCORE_BLOCKS {
                self.picked[i] = self.picked[i * live / SCORE_BLOCKS];
            }
            self.picked.truncate(SCORE_BLOCKS);
        }
        let (mut length, mut count) = (0.0f64, 0usize);
        for &j in &self.picked {
            self.carriers
                .block_db(self.samples, offset + j * N_FFT, &mut self.row);
            let phasor = Phasor::of(&self.row);
            length += phasor.length();
            count += phasor.count;
        }
        let score = if count == 0 {
            0.0
        } else {
            length / count as f64
        };
        self.memo[offset] = score;
        score
    }

    /// Best alignment within a coarse step of `start` in steps of
    /// [`MEDIUM_STEP`], then within a medium step of that in steps of 1.
    /// Offsets wrap: block grids repeat every `N_FFT` samples.
    fn refine(&mut self, start: usize) -> (f64, usize) {
        let mut best = (self.score(start), start);
        for (step, reach) in [
            (MEDIUM_STEP, COARSE_STEP / MEDIUM_STEP - 1),
            (1, MEDIUM_STEP - 1),
        ] {
            let centre = best.1;
            for k in 1..=reach {
                for signed in [k as isize, -(k as isize)] {
                    let offset = (centre as isize + signed * step as isize)
                        .rem_euclid(N_FFT as isize) as usize;
                    let score = self.score(offset);
                    if score > best.0 {
                        best = (score, offset);
                    }
                }
            }
        }
        best
    }

    /// Decode every block at `offset`.
    fn decode(&mut self, offset: usize, payload_len: usize, n_bits: usize) -> Decoded {
        let width = self.row.len();
        let blocks = self.blocks_at(offset);
        let mut db = vec![SKIPPED; blocks * width];
        let mut phasors = Vec::with_capacity(blocks);
        for (j, row) in db.chunks_exact_mut(width).enumerate() {
            if self.is_live(offset, j) {
                self.carriers
                    .block_db(self.samples, offset + j * N_FFT, row);
            }
            phasors.push(Phasor::of(row));
        }
        let level = level_track(&phasors);
        let Some(votes) = block_votes(&db, width, n_bits, &level) else {
            return Decoded {
                payload: None,
                mean_vote: 0.0,
            };
        };
        let mean_vote = votes.iter().map(|v| v.abs()).sum::<f64>() / n_bits as f64;
        let payload =
            hard_bits(&votes).and_then(|bits| unframe_either_polarity(&bits, payload_len));
        Decoded { payload, mean_vote }
    }
}

/// `x` wrapped into `[-1.5, 1.5]`: the distance to the nearest multiple of 3.
fn wrap_lattice(x: f64) -> f64 {
    x - LATTICE_DB * (x / LATTICE_DB).round()
}

/// Gain of each block in dB, from the phasors of the blocks around it.
///
/// A phasor gives the gain modulo 3 dB. Starting from the first block with
/// any carriers, each estimate is unwrapped onto the previous one, so a
/// level that drifts by less than 1.5 dB between neighbouring blocks is
/// followed through any number of lattice periods. Blocks without carriers
/// keep the last level.
fn level_track(phasors: &[Phasor]) -> Vec<f64> {
    let mut level = vec![0.0; phasors.len()];
    let mut previous: Option<f64> = None;
    for (j, slot) in level.iter_mut().enumerate() {
        let lo = j.saturating_sub(GAIN_SMOOTH);
        let hi = (j + GAIN_SMOOTH + 1).min(phasors.len());
        let (re, im) = phasors[lo..hi]
            .iter()
            .fold((0.0, 0.0), |(re, im), p| (re + p.re, im + p.im));
        if re * re + im * im < 1e-18 {
            *slot = previous.unwrap_or(0.0);
            continue;
        }
        let modulo = libm::atan2(im, re) * (LATTICE_DB / core::f64::consts::TAU);
        let unwrapped = previous.map_or(modulo, |p| p + wrap_lattice(modulo - p));
        *slot = unwrapped;
        previous = Some(unwrapped);
    }
    level
}

/// Positive when `db` is nearer the bit-1 lattice than the bit-0 lattice.
fn lattice_vote(db: f64) -> f64 {
    let q = f64::from(Q_DB);
    lattice_dist(db, 0.0, q) - lattice_dist(db, q * 0.5, q)
}

/// Mean lattice vote per bit over all blocks, after removing each block's
/// level. A block votes with the mean of its bit's usable carriers; `None`
/// when some bit never had one.
fn block_votes(db: &[f32], width: usize, n_bits: usize, level: &[f64]) -> Option<Vec<f64>> {
    let mut sum = vec![0.0f64; n_bits];
    let mut voters = vec![0u32; n_bits];
    for (row, &gain) in db.chunks_exact(width).zip(level) {
        for b in 0..n_bits {
            let (mut vote, mut n) = (0.0f64, 0u32);
            for k in 0..BINS_PER_BIT {
                let db = row[b + k * n_bits];
                if !db.is_nan() {
                    vote += lattice_vote(f64::from(db) - gain);
                    n += 1;
                }
            }
            if n > 0 {
                sum[b] += vote / f64::from(n);
                voters[b] += 1;
            }
        }
    }
    sum.iter()
        .zip(&voters)
        .map(|(&s, &n)| (n > 0).then(|| s / f64::from(n)))
        .collect()
}

fn qim_bins(key: &[u8], n_bits: usize) -> Vec<usize> {
    keyed_bins(
        key,
        b"qim-block-bins",
        BIN_LO,
        BIN_HI,
        n_bits.saturating_mul(BINS_PER_BIT),
    )
}

fn lattice_dist(db: f64, offset: f64, q: f64) -> f64 {
    let x = (db - offset) / q;
    (x - x.round()).abs()
}

fn quantize_db(db: f32, target: f32, q: f32) -> f32 {
    let step = ((db - target) / q).round();
    step * q + target
}

fn set_herm_mag(buf: &mut [Complex<f32>], bin: usize, new_mag: f32) {
    let n = buf.len();
    if bin == 0 || bin >= n / 2 {
        return;
    }
    let m = modulus(buf[bin]);
    if m < 1e-8 {
        return;
    }
    buf[bin] *= new_mag / m;
    buf[n - bin] = buf[bin].conj();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::watermark::classical::coding::frame;
    use crate::watermark::classical::testing::{gain_db, noise, through_pcm16};
    use alloc::format;

    const KEY: &[u8] = b"bakeoff-secret-key-0123456789";
    const PAYLOAD: [u8; 8] = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
    const N_BITS: usize = 96;

    /// `blocks` blocks of marked seeded noise, through PCM16.
    fn marked(blocks: usize) -> Vec<f32> {
        through_pcm16(&embed(&noise(blocks * N_FFT), &frame(&PAYLOAD), KEY))
    }

    fn detected(samples: &[f32]) -> Detection {
        detect(samples, PAYLOAD.len(), N_BITS, KEY)
    }

    fn assert_payload(detection: &Detection, what: &str) {
        assert_eq!(detection.payload.as_deref(), Some(&PAYLOAD[..]), "{what}");
    }

    /// The detector runs on [`RealFft`]; the embedder (and the bakeoff
    /// reference) uses a full complex FFT. Pin that both backends produce
    /// the same magnitudes in the carrier band, so the substitution cannot
    /// move a lattice vote.
    #[test]
    fn real_fft_matches_complex_fft_in_carrier_band() {
        let fwd = FftPlannerScalar::<f32>::new().plan_fft_forward(N_FFT);
        let mut scratch = vec![Complex::new(0.0, 0.0); fwd.get_inplace_scratch_len()];
        let mut rfft = RealFft::new(N_FFT);
        let mut worst = 0.0f64;
        let mut x = 0x1234_5678u32;
        for _ in 0..8 {
            let mut time = vec![0.0f32; N_FFT];
            for t in &mut time {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *t = (x >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
            }

            let mut complex: Vec<Complex<f32>> =
                time.iter().map(|&s| Complex::new(s, 0.0)).collect();
            fwd.process_with_scratch(&mut complex, &mut scratch);

            let mut real_time = time.clone();
            let mut real = vec![Complex::new(0.0, 0.0); N_FFT / 2 + 1];
            rfft.process(&mut real_time, &mut real);

            for bin in BIN_LO..BIN_HI.min(N_FFT / 2) {
                let a = modulus(complex[bin]);
                let b = modulus(real[bin]);
                worst = worst.max(f64::from((a - b).abs() / a.max(1e-30)));
            }
        }
        // Measured worst gap across eight frames: 4.8e-6 relative
        // (≈ 40 f32 ulps, −106 dB) — five orders below the 6 dB lattice
        // step, so a vote cannot move.
        assert!(worst < 1e-5, "real/complex magnitude gap {worst:.3e}");
    }

    fn phasor_at(gain: f64, count: usize) -> Phasor {
        let angle = core::f64::consts::TAU * gain / LATTICE_DB;
        Phasor {
            re: count as f64 * libm::cos(angle),
            im: count as f64 * libm::sin(angle),
            count,
        }
    }

    #[test]
    fn wrap_lattice_folds_into_half_a_period() {
        for (x, want) in [
            (0.0, 0.0),
            (1.4, 1.4),
            (1.6, -1.4),
            (-1.6, 1.4),
            (7.5, -1.5),
        ] {
            assert!((wrap_lattice(x) - want).abs() < 1e-12, "{x}");
        }
    }

    #[test]
    fn level_track_holds_a_constant_gain_modulo_the_lattice() {
        for gain in [-7.5, -3.0, -1.4, 0.0, 0.7, 1.4, 3.0, 7.5, 9.9] {
            let level = level_track(&vec![phasor_at(gain, 384); 30]);
            for (j, &l) in level.iter().enumerate() {
                assert!(
                    wrap_lattice(l - gain).abs() < 1e-9,
                    "gain {gain} block {j}: {l}"
                );
                assert!(
                    (l - level[0]).abs() < 1e-9,
                    "gain {gain} drifted at block {j}"
                );
            }
        }
    }

    #[test]
    fn level_track_unwraps_a_drifting_gain_across_lattice_periods() {
        // 0 to 12 dB over 60 blocks: four lattice periods, 0.2 dB per block.
        let phasors: Vec<Phasor> = (0..60).map(|j| phasor_at(0.2 * j as f64, 384)).collect();
        let level = level_track(&phasors);
        // Away from the ends, where the smoothing window is symmetric, the
        // track is the ramp itself, not the ramp folded into 3 dB.
        for j in 5..55 {
            assert!((level[j + 1] - level[j] - 0.2).abs() < 1e-9, "block {j}");
        }
        assert!(level[54] - level[5] > 9.7);
    }

    #[test]
    fn level_track_holds_its_value_through_blocks_without_carriers() {
        let mut phasors = vec![Phasor::default(); 12];
        phasors.extend(vec![phasor_at(2.0, 384); 12]);
        phasors.extend(vec![Phasor::default(); 40]);
        let level = level_track(&phasors);
        // Blocks whose whole smoothing window is empty have nothing to say:
        // zero before the first block with carriers, then the last level.
        assert!(level[..8].iter().all(|&l| l == 0.0));
        assert!(wrap_lattice(level[16] - 2.0).abs() < 1e-9);
        assert!(level[28..].iter().all(|&l| l == level[27]));
    }

    #[test]
    fn carriers_below_the_embedder_floor_are_skipped() {
        let order = qim_bins(KEY, N_BITS);
        let mut carriers = Carriers::new(&order);
        let mut row = vec![0.0f32; order.len()];
        carriers.block_db(&vec![0.0; N_FFT], 0, &mut row);
        assert!(row.iter().all(|d| d.is_nan()));
        carriers.block_db(&vec![1e-9; N_FFT], 0, &mut row);
        assert!(row.iter().all(|d| d.is_nan()));
        carriers.block_db(&noise(N_FFT), 0, &mut row);
        assert!(row.iter().all(|d| d.is_finite()));
        assert_eq!(Phasor::of(&row).count, order.len());
    }

    #[test]
    fn votes_follow_the_lattice_and_the_level() {
        // Bit 0 on multiples of 6 dB, bit 1 on odd multiples of 3.
        assert!(lattice_vote(-60.0) < -0.49 && lattice_vote(-63.0) > 0.49);
        assert!(lattice_vote(-61.0) < 0.0 && lattice_vote(-62.0) > 0.0);
        assert_eq!(lattice_vote(-61.5), 0.0);
        // A block at +3 dB reads every bit inverted unless its level is
        // removed. Two bits, four carriers each, laid out as `b + k * n_bits`.
        let (bit0, bit1) = (-60.0f32 + 3.0, -63.0f32 + 3.0);
        let row = [bit0, bit1, bit0, bit1, bit0, bit1, bit0, bit1];
        let raw = block_votes(&row, 8, 2, &[0.0]).unwrap();
        assert!(raw[0] > 0.0 && raw[1] < 0.0, "{raw:?}");
        let tracked = block_votes(&row, 8, 2, &[3.0]).unwrap();
        assert!(tracked[0] < 0.0 && tracked[1] > 0.0, "{tracked:?}");
    }

    #[test]
    fn a_bit_nothing_voted_on_is_an_erasure() {
        // Bit 1's four carriers were all below the floor in every block.
        let n_bits = 2;
        let mut db = vec![SKIPPED; 2 * 4 * n_bits];
        for row in db.chunks_exact_mut(4 * n_bits) {
            for k in 0..4 {
                row[k * n_bits] = -60.0;
            }
        }
        assert_eq!(block_votes(&db, 4 * n_bits, n_bits, &[0.0, 0.0]), None);
        db.iter_mut().for_each(|d| *d = SKIPPED);
        assert_eq!(block_votes(&db, 4 * n_bits, n_bits, &[0.0, 0.0]), None);
    }

    #[test]
    fn alignment_statistic_is_sharp_and_blind_to_bits_and_level() {
        let clip = marked(10);
        let order = qim_bins(KEY, N_BITS);
        let mut search = Search::new(&clip, &order);
        let aligned = search.score(0);
        assert!(aligned > 0.9, "{aligned}");
        // White noise is the forgiving host: every bin is strong, so even a
        // misalignment of a few samples keeps some coherence.
        for off in [16, 32, 500, 1023] {
            let off_score = search.score(off);
            assert!(off_score < 0.3, "offset {off}: {off_score}");
        }
        for db in [-7.5, 3.0, 9.0] {
            let louder = gain_db(&clip, db);
            let mut search = Search::new(&louder, &order);
            assert!((search.score(0) - aligned).abs() < 0.02, "{db} dB");
        }
        // An offset that leaves no whole block has no score.
        let short = &clip[..N_FFT + 100];
        let mut search = Search::new(short, &order);
        assert!(search.score(100).is_finite());
        assert_eq!(search.score(101), f64::NEG_INFINITY);
    }

    #[test]
    fn detects_through_level_changes_including_the_ones_that_swap_every_bit() {
        let clip = marked(12);
        for db in [-7.5, -3.0, 3.0, 6.0206, 9.0] {
            let detection = detected(&gain_db(&clip, db));
            assert_payload(&detection, &format!("{db} dB"));
            assert_eq!(detection.sync_offset, 0, "{db} dB");
        }
    }

    #[test]
    fn follows_a_level_that_moves_through_several_lattice_periods() {
        let clip = marked(24);
        let last = (clip.len() - 1) as f32;
        let ramp: Vec<f32> = clip
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                // -3 dB to +6 dB across the clip: three lattice periods.
                let db = -3.0 + 9.0 * i as f32 / last;
                s * libm::powf(10.0, db / 20.0)
            })
            .collect();
        assert_payload(&detected(&through_pcm16(&ramp)), "gain ramp");
    }

    #[test]
    fn finds_the_alignment_to_the_sample() {
        let clip = marked(12);
        for trim in [1usize, 17, 100, 2049] {
            let detection = detected(&clip[trim..]);
            assert_payload(&detection, &format!("lead trim {trim}"));
            // Blocks start at `-trim` modulo the block length.
            assert_eq!(
                detection.sync_offset,
                (N_FFT - trim % N_FFT) % N_FFT,
                "lead trim {trim}"
            );
        }
    }

    #[test]
    fn decodes_a_burst_surrounded_by_digital_silence() {
        // 90% silence: the mark survives only in the middle tenth.
        let clip = marked(30);
        let mut quiet = vec![0.0f32; clip.len()];
        quiet[13 * N_FFT..16 * N_FFT].copy_from_slice(&clip[13 * N_FFT..16 * N_FFT]);
        assert_payload(&detected(&quiet), "90% silence");

        // A burst in minutes of silence: ranking on evenly spaced blocks of
        // the whole clip would never meet it.
        let mut sparse = vec![0.0f32; 400 * N_FFT];
        sparse[250 * N_FFT + 17..256 * N_FFT + 17].copy_from_slice(&clip[..6 * N_FFT]);
        let detection = detected(&sparse);
        assert_payload(&detection, "6 blocks in 400");
        // Delayed by 17 samples, so the block grid starts there.
        assert_eq!(detection.sync_offset, 17);
    }

    #[test]
    fn mostly_silent_input_costs_only_its_live_blocks() {
        let clip = marked(8);
        let order = qim_bins(KEY, N_BITS);
        let mut sparse = vec![0.0f32; 100 * N_FFT];
        sparse[40 * N_FFT..48 * N_FFT].copy_from_slice(&clip);
        let search = Search::new(&sparse, &order);
        assert_eq!(search.live.iter().filter(|&&live| live).count(), 8);
        assert!(!search.is_live(0, 3) && search.is_live(0, 40));
        // Off the grid a block straddles two segments.
        assert!(search.is_live(5, 39) && search.is_live(5, 47) && !search.is_live(5, 48));
    }

    #[test]
    fn evidence_free_clips_return_no_payload_and_no_score() {
        for clip in [
            vec![0.0f32; 4 * N_FFT],
            vec![1e-9; 4 * N_FFT],
            vec![0.25; 4 * N_FFT],
        ] {
            let detection = detected(&clip);
            assert!(detection.payload.is_none());
            assert_eq!(detection.score, 0.0);
        }
    }

    #[test]
    fn score_is_the_mean_absolute_vote_and_larger_for_a_marked_clip() {
        let clip = marked(12);
        let marked_score = detected(&clip).score;
        assert!(marked_score > 0.2, "{marked_score}");
        let other_key = detect(&clip, PAYLOAD.len(), N_BITS, b"another-key-A");
        assert!(other_key.payload.is_none());
        assert!(other_key.score < marked_score / 2.0, "{}", other_key.score);
    }
}
