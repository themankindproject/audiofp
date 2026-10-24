//! Overlapped log-magnitude DSSS with per-bin host subtraction.
//!
//! Geometry matches the bakeoff `log_host`: FFT 2048, hop 512, ±3 dB chips,
//! 300 bins from bin 10 to 75% of the spectrum. The detector searches every
//! bit phase and keeps the one with the largest absolute correlation. A bit
//! nothing voted on, or one whose correlation is exactly zero, is an erasure:
//! it is never read as a 1.

use alloc::vec;
use alloc::vec::Vec;

use super::coding::{hard_bits, unframe};
use super::keying::{ChaCha, derive_key, keyed_bins};
use num_complex::Complex;

use super::stft::{ComplexStft, modulus};
use super::{Detection, HOP, N_FFT};

/// Linear gain for a +3 dB chip: `10^(3/20)`.
const GAIN_UP: f32 = 1.412_537_6;
/// Linear gain for a −3 dB chip: `10^(-3/20)`.
const GAIN_DOWN: f32 = 0.707_945_8;
const BIN_LO: usize = 10;
const BIN_HI_FRAC: f32 = 0.75;
const BINS_PER_FRAME: usize = 300;
/// Frames on each side of a frame that its per-bin host estimate averages
/// over (17 frames, about 0.5 s). A clip-wide mean leaves the host's spectral
/// drift in the correlation; this window follows it. Picked from 2..=33 on
/// ten tracks, which are also the tracks the recovery figures in the
/// changelog come from.
const LOCAL_MEAN_RADIUS: usize = 8;

/// STFT frames per framed bit a clip needs before recovery stops depending
/// on how loud the host is next to the mark. See [`usable_min_samples`].
const USABLE_FRAMES_PER_BIT: usize = 5;

/// Clip length this algorithm can be expected to decode from: every bit read
/// from [`USABLE_FRAMES_PER_BIT`] frames. The structural minimum is one.
pub(crate) const fn usable_min_samples(n_bits: usize) -> usize {
    n_bits
        .saturating_mul(USABLE_FRAMES_PER_BIT)
        .saturating_mul(HOP)
}

/// Embed `bits`, one STFT frame at a time: analyse, scale the keyed bins,
/// inverse-transform and overlap-add, so the whole complex spectrogram is
/// never held (see [`ComplexStft::synthesize_frame`]).
pub(crate) fn embed(samples: &[f32], bits: &[u8], key: &[u8]) -> Vec<f32> {
    let mut stft = ComplexStft::new(N_FFT, HOP);
    let n_frames = stft.n_frames(samples.len());
    let nb = stft.n_bins();
    let order = log_order(key, nb);
    let n_bits = bits.len().max(1);
    let n_ord = order.len();
    let chips = chips(key, n_bits, n_ord);
    let mut ola = stft.overlap_add(samples.len());
    let mut frame = vec![Complex::new(0.0f32, 0.0); nb];
    for f in 0..n_frames {
        frame.copy_from_slice(stft.analyze_frame(samples, f));
        let b = f % n_bits;
        let row = &chips[b * n_ord..(b + 1) * n_ord];
        // Bit 0 flips every chip; the chip itself picks the ±3 dB direction.
        let negate = bits[b] == 0;
        for (i, &bin) in order.iter().enumerate() {
            let cell = &mut frame[bin];
            let mag = modulus(*cell);
            if mag < 1e-7 {
                continue;
            }
            let gain = if (row[i] > 0.0) != negate {
                GAIN_UP
            } else {
                GAIN_DOWN
            };
            let new_mag = (mag * gain).max(1e-7);
            *cell *= new_mag / mag;
        }
        stft.synthesize_frame(&frame, f, &mut ola);
    }
    ola.finish(N_FFT)
}

/// Detect a payload: correlate against every bit phase, keep the strongest,
/// check the CRC.
///
/// `score` is the mean absolute correlation of the winning phase and
/// `sync_offset` that bit phase, counted in frames.
pub(crate) fn detect(samples: &[f32], payload_len: usize, n_bits: usize, key: &[u8]) -> Detection {
    if n_bits == 0 {
        return Detection::none();
    }
    let mut stft = ComplexStft::new(N_FFT, HOP);
    let n_frames = stft.n_frames(samples.len());
    let nb = stft.n_bins();
    let order = log_order(key, nb);
    if n_frames == 0 || order.is_empty() {
        return Detection::none();
    }
    // Only the keyed bins' dB levels are kept, frame by frame (300 per
    // frame instead of the 1025-bin complex spectrogram).
    let n_ord = order.len();
    let mut db = vec![0.0f64; n_frames * n_ord];
    for f in 0..n_frames {
        let bins = stft.analyze_frame(samples, f);
        for (i, &bin) in order.iter().enumerate() {
            let mag = modulus(bins[bin]).max(1e-7);
            db[f * n_ord + i] = f64::from(20.0 * libm::log10f(mag));
        }
    }
    let chips = chips(key, n_bits, n_ord);
    pick_phase(&db, n_frames, n_bits, &chips, payload_len)
}

/// One ±1 chip row per bit, drawn from that bit's domain-separated stream.
///
/// The rows repeat every frame, so embed and detect consume them from this
/// table instead of rebuilding a keystream per frame — the previous shape
/// allocated one `derive_key` buffer per (phase, frame) pair.
fn chips(key: &[u8], n_bits: usize, n_ord: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_bits * n_ord);
    for b in 0..n_bits {
        let mut rng = ChaCha::new(&derive_key(key, b"log-host-spread"), b as u64);
        for _ in 0..n_ord {
            out.push(hard_chip(&mut rng));
        }
    }
    out
}

fn log_order(key: &[u8], nb: usize) -> Vec<usize> {
    let bin_hi = (nb as f32 * BIN_HI_FRAC) as usize;
    keyed_bins(key, b"log-host-bins", BIN_LO, bin_hi, BINS_PER_FRAME)
}

/// Per-bin mean of the log-magnitudes over the frames within
/// [`LOCAL_MEAN_RADIUS`] of the frame being read, frames clipped to the clip.
///
/// The window slides: `residual` must be asked for frames in non-decreasing
/// order, and each call only adds and drops the frames that moved in or out.
struct LocalMean<'a> {
    /// `(n_frames, n_ord)`, row-major.
    db: &'a [f64],
    n_frames: usize,
    n_ord: usize,
    /// Per-bin sum over frames `lo..hi`.
    window: Vec<f64>,
    lo: usize,
    hi: usize,
}

impl<'a> LocalMean<'a> {
    fn new(db: &'a [f64], n_frames: usize, n_ord: usize) -> Self {
        Self {
            db,
            n_frames,
            n_ord,
            window: vec![0.0; n_ord],
            lo: 0,
            hi: 0,
        }
    }

    /// Write frame `f` minus its local mean into `out`.
    fn residual(&mut self, f: usize, out: &mut [f64]) {
        let (db, n_ord) = (self.db, self.n_ord);
        let frame = |f: usize| &db[f * n_ord..(f + 1) * n_ord];
        let want_hi = (f + LOCAL_MEAN_RADIUS + 1).min(self.n_frames);
        while self.hi < want_hi {
            for (w, &d) in self.window.iter_mut().zip(frame(self.hi)) {
                *w += d;
            }
            self.hi += 1;
        }
        let want_lo = f.saturating_sub(LOCAL_MEAN_RADIUS);
        while self.lo < want_lo {
            for (w, &d) in self.window.iter_mut().zip(frame(self.lo)) {
                *w -= d;
            }
            self.lo += 1;
        }
        // Dividing, not multiplying by a reciprocal: a window of identical
        // frames must leave a residual of exactly zero. The terms are `f32`
        // values widened to `f64`, so the sum of up to 17 copies of one is
        // exact, and so is the quotient.
        let count = (self.hi - self.lo) as f64;
        for ((r, &d), &w) in out.iter_mut().zip(frame(f)).zip(&self.window) {
            *r = d - w / count;
        }
    }
}

/// Correlate every frame with every bit's chip row after removing the host,
/// pool the correlations under each bit phase, and keep the strongest phase.
///
/// A frame carries a different bit from its neighbours, so the mark largely
/// averages out of the local mean while the host's own spectral drift stays
/// in it.
///
/// Phase `p` reads frame `f` as bit `(f + p) % n_bits`, so frames congruent
/// modulo `n_bits` always carry the same bit. They are summed first
/// (`folded[g] = Σ_{f ≡ g} residual_f`), and the correlation of bit `b`
/// under phase `p` is then one dot product, `folded[(b − p) mod n_bits] ·
/// chip_b`: `n_bits²` dot products instead of `n_frames · n_bits` (a clip
/// holds at least one frame per bit), ≈ 60× faster at typical lengths. Only
/// the folded rows and one phase's `n_bits` correlations are alive, never
/// the per-frame residual matrix or an `n_bits²` phase table.
fn pick_phase(
    db: &[f64],
    n_frames: usize,
    n_bits: usize,
    chips: &[f32],
    payload_len: usize,
) -> Detection {
    let n_ord = chips.len() / n_bits;
    let mut host = LocalMean::new(db, n_frames, n_ord);
    let mut row = vec![0.0f64; n_ord];
    let mut folded = vec![0.0f64; n_bits * n_ord];
    let mut any_live = false;
    for f in 0..n_frames {
        host.residual(f, &mut row);
        // A frame that equals its neighbours adds exactly nothing.
        if row.iter().any(|&r| r != 0.0) {
            any_live = true;
            let g = f % n_bits;
            for (acc, &r) in folded[g * n_ord..(g + 1) * n_ord].iter_mut().zip(&row) {
                *acc += r;
            }
        }
    }
    if !any_live {
        // Every frame equals its neighbours (digital silence, or a constant
        // host): there is nothing to correlate, so no bit has a vote.
        return Detection::none();
    }

    let mut corr = vec![0.0f64; n_bits];
    let mut strongest = Strongest::new();
    for p in 0..n_bits {
        for (b, c) in corr.iter_mut().enumerate() {
            let g = (b + n_bits - p) % n_bits;
            let frames = &folded[g * n_ord..(g + 1) * n_ord];
            let chip = &chips[b * n_ord..(b + 1) * n_ord];
            *c = frames
                .iter()
                .zip(chip)
                .map(|(&r, &chip)| r * f64::from(chip))
                .sum();
        }
        strongest.offer(p, &corr);
    }
    strongest.decode(payload_len)
}

/// The bit phase with the largest total absolute correlation seen so far.
struct Strongest {
    energy: f64,
    phase: usize,
    corr: Vec<f64>,
}

impl Strongest {
    fn new() -> Self {
        Self {
            energy: f64::NEG_INFINITY,
            phase: 0,
            corr: Vec::new(),
        }
    }

    /// Consider `corr`, one correlation per bit, as the reading under `phase`.
    fn offer(&mut self, phase: usize, corr: &[f64]) {
        let energy: f64 = corr.iter().map(|c| c.abs()).sum();
        if energy > self.energy {
            self.energy = energy;
            self.phase = phase;
            self.corr.clear();
            self.corr.extend_from_slice(corr);
        }
    }

    /// Decode the strongest phase. A correlation of exactly zero is a tie,
    /// and a tie is an erasure.
    fn decode(&self, payload_len: usize) -> Detection {
        Detection {
            payload: hard_bits(&self.corr).and_then(|bits| unframe(&bits, payload_len)),
            score: self.energy / self.corr.len().max(1) as f64,
            sync_offset: self.phase,
        }
    }
}

fn hard_chip(rng: &mut ChaCha) -> f32 {
    if rng.u32() & 1 == 0 { -1.0 } else { 1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::watermark::classical::coding::frame;
    use crate::watermark::classical::keying::ChaCha;
    use crate::watermark::classical::testing::{noise, through_pcm16};

    const KEY: &[u8] = b"bakeoff-secret-key-0123456789";
    const PAYLOAD: [u8; 8] = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];

    /// Log-magnitudes as the detector sees them: `f32` values in dB.
    fn random_db(n_frames: usize, n_ord: usize) -> Vec<f64> {
        let mut rng = ChaCha::new(b"local-mean", 7);
        (0..n_frames * n_ord)
            .map(|_| {
                f64::from(
                    (f64::from(rng.u32() >> 8) / f64::from(1u32 << 24) * 200.0 - 140.0) as f32,
                )
            })
            .collect()
    }

    #[test]
    fn local_mean_residual_matches_a_direct_window_mean() {
        let (n_frames, n_ord) = (60, 5);
        let db = random_db(n_frames, n_ord);
        let mut host = LocalMean::new(&db, n_frames, n_ord);
        let mut residual = vec![0.0; n_ord];
        for f in 0..n_frames {
            host.residual(f, &mut residual);
            let lo = f.saturating_sub(LOCAL_MEAN_RADIUS);
            let hi = (f + LOCAL_MEAN_RADIUS + 1).min(n_frames);
            for (i, &r) in residual.iter().enumerate() {
                let sum: f64 = (lo..hi).map(|g| db[g * n_ord + i]).sum();
                // The running sum is exact, so this is the same value to the bit.
                assert_eq!(
                    r,
                    db[f * n_ord + i] - sum / (hi - lo) as f64,
                    "frame {f} bin {i}"
                );
            }
        }
    }

    #[test]
    fn local_mean_window_is_clipped_at_both_ends_of_the_clip() {
        // 3 frames: every window is the whole clip, so the residuals sum to zero.
        let db = random_db(3, 4);
        let mut host = LocalMean::new(&db, 3, 4);
        let mut total = vec![0.0; 4];
        let mut residual = vec![0.0; 4];
        for f in 0..3 {
            host.residual(f, &mut residual);
            total.iter_mut().zip(&residual).for_each(|(t, r)| *t += r);
        }
        assert!(total.iter().all(|t| t.abs() < 1e-9), "{total:?}");
    }

    #[test]
    fn identical_frames_leave_a_residual_of_exactly_zero() {
        let n_ord = 9;
        let frame: Vec<f64> = (0..n_ord)
            .map(|i| f64::from((-100.0 + i as f64 * 0.37) as f32))
            .collect();
        let db: Vec<f64> = frame.iter().copied().cycle().take(40 * n_ord).collect();
        let mut host = LocalMean::new(&db, 40, n_ord);
        let mut residual = vec![1.0; n_ord];
        for f in 0..40 {
            host.residual(f, &mut residual);
            assert!(
                residual.iter().all(|&r| r == 0.0),
                "frame {f}: {residual:?}"
            );
        }
    }

    /// What digital silence, DC and any other constant host analyse to:
    /// identical frames. For every payload length, the whole 1..=64 range
    /// that a LogHost payload is likely to use.
    #[test]
    fn identical_frames_decode_to_no_payload_for_every_payload_length() {
        let n_ord = 300;
        for payload_len in 1..=64usize {
            let n_bits = payload_len * 8 + 32;
            // Never read: there is no evidence to correlate.
            let chips = vec![1.0f32; n_bits * n_ord];
            for host in [
                |_: usize| -140.0,
                |i: usize| f64::from((-100.0 + i as f64 * 0.37) as f32),
            ] {
                let n_frames = n_bits + 30;
                let db: Vec<f64> = (0..n_frames * n_ord).map(|k| host(k % n_ord)).collect();
                let found = pick_phase(&db, n_frames, n_bits, &chips, payload_len);
                assert!(found.payload.is_none(), "payload_len {payload_len}");
                assert_eq!((found.score, found.sync_offset), (0.0, 0));
            }
        }
    }

    /// A frame sequence as the detector reads it: a host whose spectrum
    /// drifts over time, plus the ±3 dB chips for the framed `PAYLOAD`.
    fn marked_analysis(n_frames: usize) -> (Vec<f64>, Vec<f32>) {
        let (n_bits, n_ord) = (96, 300);
        let chips = chips(KEY, n_bits, n_ord);
        let bits = frame(&PAYLOAD);
        let step = 20.0 * libm::log10(f64::from(GAIN_UP));
        let mut db = vec![0.0f64; n_frames * n_ord];
        for f in 0..n_frames {
            let b = f % n_bits;
            let sign = if bits[b] == 1 { 1.0 } else { -1.0 };
            for i in 0..n_ord {
                let drift = 18.0 * libm::sin(f as f64 / 11.0 + i as f64 / 23.0);
                let host = 45.0 - 0.05 * i as f64 + drift;
                db[f * n_ord + i] =
                    f64::from((host + sign * step * f64::from(chips[b * n_ord + i])) as f32);
            }
        }
        (db, chips)
    }

    /// Root-mean-square of what is left of the host after subtracting its
    /// estimate, over every frame and bin.
    fn residual_rms(db: &[f64], n_frames: usize, n_ord: usize, clip_wide: bool) -> f64 {
        let mut host = LocalMean::new(db, n_frames, n_ord);
        let mut residual = vec![0.0; n_ord];
        let mut energy = 0.0;
        for f in 0..n_frames {
            if clip_wide {
                for i in 0..n_ord {
                    let mean: f64 =
                        (0..n_frames).map(|g| db[g * n_ord + i]).sum::<f64>() / n_frames as f64;
                    residual[i] = db[f * n_ord + i] - mean;
                }
            } else {
                host.residual(f, &mut residual);
            }
            energy += residual.iter().map(|r| r * r).sum::<f64>();
        }
        libm::sqrt(energy / (n_frames * n_ord) as f64)
    }

    #[test]
    fn a_drifting_host_decodes_and_leaves_far_less_host_than_the_clip_wide_mean() {
        let n_frames = 96 * 6;
        let (db, chips) = marked_analysis(n_frames);
        let found = pick_phase(&db, n_frames, 96, &chips, PAYLOAD.len());
        assert_eq!(found.payload.as_deref(), Some(&PAYLOAD[..]));
        assert_eq!(found.sync_offset, 0);
        assert!(found.score > 0.0);
        // The drift is an 18 dB sinusoid (12.7 dB RMS) next to ±3 dB chips.
        let local = residual_rms(&db, n_frames, 300, false);
        let clip_wide = residual_rms(&db, n_frames, 300, true);
        assert!(clip_wide > 12.0, "{clip_wide}");
        assert!(local < 0.3 * clip_wide, "{local} vs {clip_wide}");
        // What remains is mostly the chips themselves.
        assert!(local > 2.5, "{local}");
    }

    #[test]
    fn the_bit_phase_follows_a_clip_that_starts_part_way_through_the_frame() {
        let n_frames = 96 * 6;
        let (db, chips) = marked_analysis(n_frames);
        let skipped = 7;
        let rest = &db[skipped * 300..];
        let found = pick_phase(rest, n_frames - skipped, 96, &chips, PAYLOAD.len());
        assert_eq!(found.payload.as_deref(), Some(&PAYLOAD[..]));
        assert_eq!(found.sync_offset, skipped);
    }

    /// Correlations that read as the framed `PAYLOAD`.
    fn correlations_of_payload() -> Vec<f64> {
        frame(&PAYLOAD)
            .iter()
            .map(|&bit| if bit == 1 { 40.0 } else { -40.0 })
            .collect()
    }

    #[test]
    fn a_tied_bit_is_an_erasure_not_a_one() {
        let bits = frame(&PAYLOAD);
        let corr = correlations_of_payload();
        let mut strongest = Strongest::new();
        strongest.offer(3, &corr);
        let found = strongest.decode(PAYLOAD.len());
        assert_eq!(found.payload.as_deref(), Some(&PAYLOAD[..]));
        assert_eq!(found.sync_offset, 3);
        // Zero one bit that is a 1 in the frame. Reading ties as 1 would
        // still decode this; an erasure must not. Nor a tie on a 0 bit.
        for (bit, name) in [(1u8, "a 1"), (0, "a 0")] {
            let at = bits.iter().position(|&b| b == bit).unwrap();
            for zero in [0.0, -0.0] {
                let mut tied = Strongest::new();
                let mut row = corr.clone();
                row[at] = zero;
                tied.offer(3, &row);
                assert!(
                    tied.decode(PAYLOAD.len()).payload.is_none(),
                    "{name} {zero:?}"
                );
            }
        }
    }

    #[test]
    fn a_table_of_ties_decodes_to_nothing_for_every_payload_length() {
        for payload_len in 1..=64usize {
            let n_bits = payload_len * 8 + 32;
            let mut strongest = Strongest::new();
            for phase in 0..n_bits {
                strongest.offer(phase, &vec![0.0; n_bits]);
            }
            let found = strongest.decode(payload_len);
            assert!(found.payload.is_none(), "payload_len {payload_len}");
            assert_eq!(found.score, 0.0);
        }
    }

    #[test]
    fn the_strongest_phase_wins_whatever_order_the_blocks_arrive_in() {
        let n_bits = 96;
        let weak = vec![1.0; n_bits];
        let strong = correlations_of_payload();
        let mut a = Strongest::new();
        a.offer(0, &weak);
        a.offer(5, &strong);
        a.offer(9, &weak);
        let mut b = Strongest::new();
        b.offer(9, &weak);
        b.offer(5, &strong);
        b.offer(0, &weak);
        for found in [a.decode(PAYLOAD.len()), b.decode(PAYLOAD.len())] {
            assert_eq!(found.sync_offset, 5);
            assert_eq!(found.payload.as_deref(), Some(&PAYLOAD[..]));
        }
        // On a tie the earlier phase wins.
        let mut c = Strongest::new();
        c.offer(2, &strong);
        c.offer(7, &strong);
        assert_eq!(c.decode(PAYLOAD.len()).sync_offset, 2);
    }

    /// The pre-folding detector: correlate each frame with its bit's chip
    /// under every phase and accumulate. `pick_phase` must agree with it.
    fn pick_phase_per_frame(
        db: &[f64],
        n_frames: usize,
        n_bits: usize,
        chips: &[f32],
        payload_len: usize,
    ) -> Detection {
        let n_ord = chips.len() / n_bits;
        let mut host = LocalMean::new(db, n_frames, n_ord);
        let mut frame = vec![0.0f64; n_ord];
        let mut table = vec![0.0f64; n_bits * n_bits];
        for f in 0..n_frames {
            host.residual(f, &mut frame);
            for p in 0..n_bits {
                let b = (f + p) % n_bits;
                let row = &chips[b * n_ord..(b + 1) * n_ord];
                let c: f64 = frame
                    .iter()
                    .zip(row)
                    .map(|(&r, &ch)| r * f64::from(ch))
                    .sum();
                table[p * n_bits + b] += c;
            }
        }
        let mut strongest = Strongest::new();
        for p in 0..n_bits {
            strongest.offer(p, &table[p * n_bits..(p + 1) * n_bits]);
        }
        strongest.decode(payload_len)
    }

    #[test]
    fn folded_correlation_matches_the_per_frame_sum() {
        for n_frames in [96, 96 * 3, 96 * 3 + 41] {
            let (db, chips) = marked_analysis(n_frames);
            let folded = pick_phase(&db, n_frames, 96, &chips, PAYLOAD.len());
            let reference = pick_phase_per_frame(&db, n_frames, 96, &chips, PAYLOAD.len());
            assert_eq!(folded.payload, reference.payload, "{n_frames} frames");
            assert_eq!(
                folded.sync_offset, reference.sync_offset,
                "{n_frames} frames"
            );
            // Only the summation order differs.
            let rel = (folded.score - reference.score).abs() / reference.score.abs().max(1e-300);
            assert!(
                rel < 1e-9,
                "{n_frames} frames: {} vs {}",
                folded.score,
                reference.score
            );
        }
        let (db, chips) = marked_analysis(96 * 3);
        assert_eq!(
            pick_phase(&db, 96 * 3, 96, &chips, PAYLOAD.len())
                .payload
                .as_deref(),
            Some(&PAYLOAD[..])
        );
    }

    #[test]
    fn decodes_marked_noise_through_pcm16() {
        let samples = noise(16_000 * 18);
        let bits = frame(&PAYLOAD);
        let marked = through_pcm16(&embed(&samples, &bits, KEY));
        let found = detect(&marked, PAYLOAD.len(), 96, KEY);
        assert_eq!(found.payload.as_deref(), Some(&PAYLOAD[..]));
        assert_eq!(found.sync_offset, 0);
        assert!(found.score > 1.0, "{}", found.score);
        // The wrong key finds nothing, and cannot tie its way to a payload.
        let wrong = detect(&marked, PAYLOAD.len(), 96, b"another-key-A");
        assert!(wrong.payload.is_none());
        assert!(wrong.score < found.score);
    }

    #[test]
    fn usable_length_is_five_frames_per_bit() {
        assert_eq!(usable_min_samples(96), 96 * 5 * HOP);
        // About 16 s for the 8-byte payload the review measured.
        assert!((15.0..16.0).contains(&(usable_min_samples(96) as f64 / 16_000.0)));
        assert_eq!(usable_min_samples(usize::MAX), usize::MAX);
    }
}
