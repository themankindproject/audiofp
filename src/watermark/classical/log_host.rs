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
    ola.finish()
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
    let chips = chips(key, n_bits, order.len());
    // Only the keyed bins' dB levels are read (300 per frame, not the
    // 1025-bin complex spectrum), and a frame is analysed when the host
    // estimate first needs it and forgotten once its window has passed, so
    // what the detector holds does not grow with the clip.
    let levels = |f: usize, row: &mut [f32]| {
        let bins = stft.analyze_frame(samples, f);
        for (level, &bin) in row.iter_mut().zip(&order) {
            *level = 20.0 * libm::log10f(modulus(bins[bin]).max(1e-7));
        }
    };
    pick_phase(levels, n_frames, n_bits, &chips, payload_len)
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

/// Frames of levels [`LocalMean`] keeps: the window around the frame being
/// read, and the frame that leaves the window as the next one enters it.
const LOCAL_MEAN_ROWS: usize = 2 * LOCAL_MEAN_RADIUS + 2;

/// Each frame's log-magnitudes minus their per-bin mean over the frames
/// within [`LOCAL_MEAN_RADIUS`] of it, frames clipped to the clip.
///
/// The window slides. Every frame is read once, in order, through `levels`,
/// when the window first reaches it, and each step only adds the frame that
/// moved in and drops the one that moved out. Only the frames the window
/// still needs are kept ([`LOCAL_MEAN_ROWS`] of them), never the clip's.
struct LocalMean<L> {
    /// Writes the level of every keyed bin of frame `f`, in dB, into a row.
    levels: L,
    n_frames: usize,
    n_ord: usize,
    /// The last [`LOCAL_MEAN_ROWS`] frames read, frame `f` in row
    /// `f % LOCAL_MEAN_ROWS`.
    rows: Vec<f32>,
    /// Per-bin sum over frames `lo..hi`.
    window: Vec<f64>,
    lo: usize,
    hi: usize,
    /// The frame [`Self::next_residual`] writes next.
    next: usize,
}

impl<L: FnMut(usize, &mut [f32])> LocalMean<L> {
    fn new(levels: L, n_frames: usize, n_ord: usize) -> Self {
        Self {
            levels,
            n_frames,
            n_ord,
            rows: vec![0.0; LOCAL_MEAN_ROWS * n_ord],
            window: vec![0.0; n_ord],
            lo: 0,
            hi: 0,
            next: 0,
        }
    }

    /// Write the next frame minus its local mean into `out`: frame 0 on the
    /// first call, then frame 1, up to the clip's `n_frames`.
    fn next_residual(&mut self, out: &mut [f64]) {
        let f = self.next;
        debug_assert!(f < self.n_frames, "frame {f} is past the clip");
        self.next += 1;
        let n_ord = self.n_ord;
        let row = |f: usize| {
            let start = (f % LOCAL_MEAN_ROWS) * n_ord;
            start..start + n_ord
        };
        let want_hi = (f + LOCAL_MEAN_RADIUS + 1).min(self.n_frames);
        while self.hi < want_hi {
            // The row this overwrites left the window at least a frame ago.
            let entering = &mut self.rows[row(self.hi)];
            (self.levels)(self.hi, entering);
            for (w, &d) in self.window.iter_mut().zip(&*entering) {
                *w += f64::from(d);
            }
            self.hi += 1;
        }
        let want_lo = f.saturating_sub(LOCAL_MEAN_RADIUS);
        while self.lo < want_lo {
            for (w, &d) in self.window.iter_mut().zip(&self.rows[row(self.lo)]) {
                *w -= f64::from(d);
            }
            self.lo += 1;
        }
        // Dividing, not multiplying by a reciprocal: a window of identical
        // frames must leave a residual of exactly zero. The terms are `f32`
        // values widened to `f64`, so the sum of up to 17 copies of one is
        // exact, and so is the quotient.
        let count = (self.hi - self.lo) as f64;
        for ((r, &d), &w) in out.iter_mut().zip(&self.rows[row(f)]).zip(&self.window) {
            *r = f64::from(d) - w / count;
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
///
/// `levels(f, row)` writes frame `f`'s level per keyed bin, in dB; it is
/// called once per frame, in order.
fn pick_phase(
    levels: impl FnMut(usize, &mut [f32]),
    n_frames: usize,
    n_bits: usize,
    chips: &[f32],
    payload_len: usize,
) -> Detection {
    let n_ord = chips.len() / n_bits;
    let mut host = LocalMean::new(levels, n_frames, n_ord);
    let mut row = vec![0.0f64; n_ord];
    let mut folded = vec![0.0f64; n_bits * n_ord];
    let mut any_live = false;
    for f in 0..n_frames {
        host.next_residual(&mut row);
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
        correlate_phase(&folded, chips, n_ord, p, &mut corr);
        strongest.offer(p, &corr);
    }
    strongest.decode(payload_len)
}

/// Dot products the phase search evaluates side by side.
const DOTS: usize = 4;

/// The correlation of every bit under phase `p`:
/// `corr[b] = folded[(b − p) mod n_bits] · chip_b`.
fn correlate_phase(folded: &[f64], chips: &[f32], n_ord: usize, p: usize, corr: &mut [f64]) {
    let n_bits = corr.len();
    for (group, out) in corr.chunks_mut(DOTS).enumerate() {
        // A short last group repeats its last bit; those sums are dropped.
        let bit = |k: usize| (group * DOTS + k).min(n_bits - 1);
        let sums = dots(
            core::array::from_fn(|k| {
                let g = (bit(k) + n_bits - p) % n_bits;
                &folded[g * n_ord..(g + 1) * n_ord]
            }),
            core::array::from_fn(|k| &chips[bit(k) * n_ord..(bit(k) + 1) * n_ord]),
        );
        out.copy_from_slice(&sums[..out.len()]);
    }
}

/// `Σ frames[k][i] · chips[k][i]` for each of the [`DOTS`] pairs of rows.
///
/// Each sum adds its own terms in index order starting from `-0.0`, as
/// `Iterator::sum` does, so it is the value a loop over one pair gives. A
/// sum of floats is a chain in which every addition waits for the one
/// before it; four chains side by side keep the processor busy while each
/// waits, which makes the phase search about three times as fast.
fn dots(frames: [&[f64]; DOTS], chips: [&[f32]; DOTS]) -> [f64; DOTS] {
    let len = frames[0].len();
    let frames = frames.map(|row| &row[..len]);
    let chips = chips.map(|row| &row[..len]);
    let mut sums = [-0.0f64; DOTS];
    for i in 0..len {
        for k in 0..DOTS {
            sums[k] += frames[k][i] * f64::from(chips[k][i]);
        }
    }
    sums
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
    ///
    /// When no phase was kept there is no evidence at all. That is the
    /// case when every correlation is NaN, which finite samples large
    /// enough to overflow the transform (about `1e35`) produce.
    fn decode(&self, payload_len: usize) -> Detection {
        if self.corr.is_empty() {
            return Detection::none();
        }
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

    /// Log-magnitudes as the detector sees them: `f32` values in dB, an
    /// `(n_frames, n_ord)` matrix in row-major order.
    fn random_db(n_frames: usize, n_ord: usize) -> Vec<f32> {
        let mut rng = ChaCha::new(b"local-mean", 7);
        (0..n_frames * n_ord)
            .map(|_| (f64::from(rng.u32() >> 8) / f64::from(1u32 << 24) * 200.0 - 140.0) as f32)
            .collect()
    }

    /// Hands the detector the frames of a matrix of levels.
    fn rows_of(db: &[f32], n_ord: usize) -> impl FnMut(usize, &mut [f32]) + '_ {
        move |f, row| row.copy_from_slice(&db[f * n_ord..(f + 1) * n_ord])
    }

    /// The host estimate before it streamed its frames: a sliding sum over
    /// the whole clip's levels held as one `f64` matrix.
    fn residuals_over_the_whole_matrix(db: &[f32], n_frames: usize, n_ord: usize) -> Vec<f64> {
        let db: Vec<f64> = db.iter().map(|&d| f64::from(d)).collect();
        let frame = |f: usize| &db[f * n_ord..(f + 1) * n_ord];
        let mut window = vec![0.0f64; n_ord];
        let (mut lo, mut hi) = (0, 0);
        let mut residuals = Vec::with_capacity(db.len());
        for f in 0..n_frames {
            let want_hi = (f + LOCAL_MEAN_RADIUS + 1).min(n_frames);
            while hi < want_hi {
                window.iter_mut().zip(frame(hi)).for_each(|(w, &d)| *w += d);
                hi += 1;
            }
            let want_lo = f.saturating_sub(LOCAL_MEAN_RADIUS);
            while lo < want_lo {
                window.iter_mut().zip(frame(lo)).for_each(|(w, &d)| *w -= d);
                lo += 1;
            }
            let count = (hi - lo) as f64;
            residuals.extend(frame(f).iter().zip(&window).map(|(&d, &w)| d - w / count));
        }
        residuals
    }

    #[test]
    fn streamed_frames_leave_the_residuals_of_the_whole_matrix_bit_for_bit() {
        let n_ord = 7;
        // Clips shorter than the window, as long as it, and many times longer.
        for n_frames in [1, 2, 8, 9, 10, 16, 17, 18, 19, 20, 37, 150] {
            let db = random_db(n_frames, n_ord);
            let expected = residuals_over_the_whole_matrix(&db, n_frames, n_ord);
            let mut read = Vec::new();
            let mut residual = vec![0.0; n_ord];
            {
                let mut frames = rows_of(&db, n_ord);
                let levels = |f: usize, row: &mut [f32]| {
                    read.push(f);
                    frames(f, row);
                };
                let mut host = LocalMean::new(levels, n_frames, n_ord);
                // What is held does not depend on the clip.
                assert_eq!(host.rows.len(), LOCAL_MEAN_ROWS * n_ord);
                for f in 0..n_frames {
                    host.next_residual(&mut residual);
                    let bits =
                        |values: &[f64]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                    assert_eq!(
                        bits(&residual),
                        bits(&expected[f * n_ord..(f + 1) * n_ord]),
                        "{n_frames} frames, frame {f}"
                    );
                }
            }
            // Every frame is analysed exactly once, in order.
            assert_eq!(read, (0..n_frames).collect::<Vec<_>>());
        }
    }

    #[test]
    fn side_by_side_dot_products_are_the_one_at_a_time_sums_bit_for_bit() {
        let n_ord = 37;
        let mut rng = ChaCha::new(b"phase-search", 3);
        // Bit counts that are and are not a multiple of `DOTS`.
        for n_bits in [1, 2, 3, 4, 5, 7, 8, 13, 96] {
            let folded: Vec<f64> = (0..n_bits * n_ord)
                .map(|_| f64::from(rng.u32()) / f64::from(u32::MAX) * 80.0 - 40.0)
                .collect();
            let chips: Vec<f32> = (0..n_bits * n_ord).map(|_| hard_chip(&mut rng)).collect();
            let mut corr = vec![f64::NAN; n_bits];
            for p in 0..n_bits {
                correlate_phase(&folded, &chips, n_ord, p, &mut corr);
                for (b, &found) in corr.iter().enumerate() {
                    let g = (b + n_bits - p) % n_bits;
                    let expected: f64 = folded[g * n_ord..(g + 1) * n_ord]
                        .iter()
                        .zip(&chips[b * n_ord..(b + 1) * n_ord])
                        .map(|(&r, &chip)| r * f64::from(chip))
                        .sum();
                    assert_eq!(
                        found.to_bits(),
                        expected.to_bits(),
                        "{n_bits} bits, phase {p}, bit {b}"
                    );
                }
            }
        }
    }

    #[test]
    fn readings_that_are_not_numbers_decode_to_no_payload_and_no_score() {
        let mut strongest = Strongest::new();
        for phase in 0..96 {
            strongest.offer(phase, &[f64::NAN; 96]);
        }
        let found = strongest.decode(PAYLOAD.len());
        assert!(found.payload.is_none());
        assert_eq!((found.score, found.sync_offset), (0.0, 0));
    }

    #[test]
    fn local_mean_residual_matches_a_direct_window_mean() {
        let (n_frames, n_ord) = (60, 5);
        let db = random_db(n_frames, n_ord);
        let mut host = LocalMean::new(rows_of(&db, n_ord), n_frames, n_ord);
        let mut residual = vec![0.0; n_ord];
        for f in 0..n_frames {
            host.next_residual(&mut residual);
            let lo = f.saturating_sub(LOCAL_MEAN_RADIUS);
            let hi = (f + LOCAL_MEAN_RADIUS + 1).min(n_frames);
            for (i, &r) in residual.iter().enumerate() {
                let sum: f64 = (lo..hi).map(|g| f64::from(db[g * n_ord + i])).sum();
                // The running sum is exact, so this is the same value to the bit.
                assert_eq!(
                    r,
                    f64::from(db[f * n_ord + i]) - sum / (hi - lo) as f64,
                    "frame {f} bin {i}"
                );
            }
        }
    }

    #[test]
    fn local_mean_window_is_clipped_at_both_ends_of_the_clip() {
        // 3 frames: every window is the whole clip, so the residuals sum to zero.
        let db = random_db(3, 4);
        let mut host = LocalMean::new(rows_of(&db, 4), 3, 4);
        let mut total = vec![0.0; 4];
        let mut residual = vec![0.0; 4];
        for _ in 0..3 {
            host.next_residual(&mut residual);
            total.iter_mut().zip(&residual).for_each(|(t, r)| *t += r);
        }
        assert!(total.iter().all(|t| t.abs() < 1e-9), "{total:?}");
    }

    #[test]
    fn identical_frames_leave_a_residual_of_exactly_zero() {
        let n_ord = 9;
        let frame: Vec<f32> = (0..n_ord)
            .map(|i| (-100.0 + i as f64 * 0.37) as f32)
            .collect();
        let db: Vec<f32> = frame.iter().copied().cycle().take(40 * n_ord).collect();
        let mut host = LocalMean::new(rows_of(&db, n_ord), 40, n_ord);
        let mut residual = vec![1.0; n_ord];
        for f in 0..40 {
            host.next_residual(&mut residual);
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
                |_: usize| -140.0f32,
                |i: usize| (-100.0 + i as f64 * 0.37) as f32,
            ] {
                let n_frames = n_bits + 30;
                let db: Vec<f32> = (0..n_frames * n_ord).map(|k| host(k % n_ord)).collect();
                let found = pick_phase(rows_of(&db, n_ord), n_frames, n_bits, &chips, payload_len);
                assert!(found.payload.is_none(), "payload_len {payload_len}");
                assert_eq!((found.score, found.sync_offset), (0.0, 0));
            }
        }
    }

    /// A frame sequence as the detector reads it: a host whose spectrum
    /// drifts over time, plus the ±3 dB chips for the framed `PAYLOAD`.
    fn marked_analysis(n_frames: usize) -> (Vec<f32>, Vec<f32>) {
        let (n_bits, n_ord) = (96, 300);
        let chips = chips(KEY, n_bits, n_ord);
        let bits = frame(&PAYLOAD);
        let step = 20.0 * libm::log10(f64::from(GAIN_UP));
        let mut db = vec![0.0f32; n_frames * n_ord];
        for f in 0..n_frames {
            let b = f % n_bits;
            let sign = if bits[b] == 1 { 1.0 } else { -1.0 };
            for i in 0..n_ord {
                let drift = 18.0 * libm::sin(f as f64 / 11.0 + i as f64 / 23.0);
                let host = 45.0 - 0.05 * i as f64 + drift;
                db[f * n_ord + i] = (host + sign * step * f64::from(chips[b * n_ord + i])) as f32;
            }
        }
        (db, chips)
    }

    /// Root-mean-square of what is left of the host after subtracting its
    /// estimate, over every frame and bin.
    fn residual_rms(db: &[f32], n_frames: usize, n_ord: usize, clip_wide: bool) -> f64 {
        let level = |f: usize, i: usize| f64::from(db[f * n_ord + i]);
        let mut host = LocalMean::new(rows_of(db, n_ord), n_frames, n_ord);
        let mut residual = vec![0.0; n_ord];
        let mut energy = 0.0;
        for f in 0..n_frames {
            // Both estimates are advanced, so the local one stays in step.
            host.next_residual(&mut residual);
            if clip_wide {
                for (i, r) in residual.iter_mut().enumerate() {
                    let mean = (0..n_frames).map(|g| level(g, i)).sum::<f64>() / n_frames as f64;
                    *r = level(f, i) - mean;
                }
            }
            energy += residual.iter().map(|r| r * r).sum::<f64>();
        }
        libm::sqrt(energy / (n_frames * n_ord) as f64)
    }

    #[test]
    fn a_drifting_host_decodes_and_leaves_far_less_host_than_the_clip_wide_mean() {
        let n_frames = 96 * 6;
        let (db, chips) = marked_analysis(n_frames);
        let found = pick_phase(rows_of(&db, 300), n_frames, 96, &chips, PAYLOAD.len());
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
        let found = pick_phase(
            rows_of(rest, 300),
            n_frames - skipped,
            96,
            &chips,
            PAYLOAD.len(),
        );
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
        db: &[f32],
        n_frames: usize,
        n_bits: usize,
        chips: &[f32],
        payload_len: usize,
    ) -> Detection {
        let n_ord = chips.len() / n_bits;
        let mut host = LocalMean::new(rows_of(db, n_ord), n_frames, n_ord);
        let mut frame = vec![0.0f64; n_ord];
        let mut table = vec![0.0f64; n_bits * n_bits];
        for f in 0..n_frames {
            host.next_residual(&mut frame);
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
            let folded = pick_phase(rows_of(&db, 300), n_frames, 96, &chips, PAYLOAD.len());
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
            pick_phase(rows_of(&db, 300), 96 * 3, 96, &chips, PAYLOAD.len())
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
