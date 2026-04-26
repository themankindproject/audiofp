//! 2-D peak picking on a magnitude spectrogram.
//!
//! [`PeakPicker`] finds local maxima inside a `(2·neighborhood_t+1) ×
//! (2·neighborhood_f+1)` box around each cell, filters by magnitude floor,
//! and applies a per-second target count so dense regions don't dominate.
//!
//! The 2-D rolling max is computed separably with Lemire's monotonic deque
//! along each axis, giving amortised O(N·M) over the whole spectrogram
//! independent of the neighbourhood size.

use alloc::collections::VecDeque;
use alloc::vec;
use alloc::vec::Vec;

/// One peak emitted by [`PeakPicker`].
///
/// `repr(C)` plus an explicit `_pad` field keeps the layout deterministic
/// (12 bytes, no implicit padding) so the struct is `bytemuck::Pod` and can
/// be stored directly in mmap'd files or shipped over a C ABI.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Peak {
    /// STFT frame index of the peak.
    pub t_frame: u32,
    /// FFT bin index of the peak.
    pub f_bin: u16,
    /// Explicit padding so the struct has no implicit gaps (required for
    /// `bytemuck::Pod`).
    pub _pad: u16,
    /// Magnitude at the peak.
    pub mag: f32,
}

/// Configuration for [`PeakPicker`].
#[derive(Clone, Debug)]
pub struct PeakPickerConfig {
    /// Half-width of the neighbourhood along the time axis. The full
    /// window is `2 * neighborhood_t + 1` frames.
    pub neighborhood_t: usize,
    /// Half-width of the neighbourhood along the frequency axis.
    pub neighborhood_f: usize,
    /// Floor on linear magnitude: cells below this are never candidates.
    pub min_magnitude: f32,
    /// Per-second cap on emitted peaks. Set to `0` to disable.
    pub target_per_sec: usize,
}

impl Default for PeakPickerConfig {
    fn default() -> Self {
        Self {
            neighborhood_t: 7,
            neighborhood_f: 7,
            min_magnitude: 1e-3,
            target_per_sec: 30,
        }
    }
}

/// 2-D peak picker.
///
/// # Example
///
/// ```
/// use audiofp::dsp::peaks::{PeakPicker, PeakPickerConfig};
///
/// // 8 frames × 8 bins, single peak at (3, 4).
/// let mut spec = vec![0.0_f32; 64];
/// spec[3 * 8 + 4] = 1.0;
///
/// let picker = PeakPicker::new(PeakPickerConfig {
///     neighborhood_t: 1,
///     neighborhood_f: 1,
///     min_magnitude: 0.1,
///     target_per_sec: 0, // disable adaptive thresholding
/// });
/// let peaks = picker.pick(&spec, 8, 8, 100.0);
/// assert_eq!(peaks.len(), 1);
/// assert_eq!((peaks[0].t_frame, peaks[0].f_bin), (3, 4));
/// ```
pub struct PeakPicker {
    cfg: PeakPickerConfig,
}

impl PeakPicker {
    /// Build a picker with the given config.
    #[must_use]
    pub fn new(cfg: PeakPickerConfig) -> Self {
        Self { cfg }
    }

    /// Borrow the configuration.
    #[must_use]
    pub fn config(&self) -> &PeakPickerConfig {
        &self.cfg
    }

    /// Pick peaks from a row-major `(n_frames, n_bins)` magnitude spectrogram.
    ///
    /// `frames_per_sec` is used only for the per-second adaptive threshold.
    /// Output is sorted by `(t_frame, f_bin)`.
    #[must_use]
    pub fn pick(
        &self,
        spec: &[f32],
        n_frames: usize,
        n_bins: usize,
        frames_per_sec: f32,
    ) -> Vec<Peak> {
        if n_frames == 0 || n_bins == 0 {
            return Vec::new();
        }
        assert_eq!(spec.len(), n_frames * n_bins, "spec length mismatch");

        let mut max_buf = vec![0.0_f32; spec.len()];
        rolling_max_2d(
            spec,
            n_frames,
            n_bins,
            self.cfg.neighborhood_t,
            self.cfg.neighborhood_f,
            &mut max_buf,
        );

        let mut candidates: Vec<Peak> = Vec::new();
        for t in 0..n_frames {
            for f in 0..n_bins {
                let idx = t * n_bins + f;
                let v = spec[idx];
                // A cell is a local maximum iff it equals the rolling max
                // value at its own location (and is above the floor).
                if v > self.cfg.min_magnitude && v >= max_buf[idx] {
                    candidates.push(Peak {
                        t_frame: t as u32,
                        f_bin: f as u16,
                        _pad: 0,
                        mag: v,
                    });
                }
            }
        }

        if self.cfg.target_per_sec > 0 && frames_per_sec > 0.0 && !candidates.is_empty() {
            candidates = adaptive_per_second(candidates, frames_per_sec, self.cfg.target_per_sec);
        }

        candidates.sort_unstable_by_key(|p| (p.t_frame, p.f_bin));
        candidates
    }
}

/// Keep the top `target` peaks per one-second bucket (by magnitude).
fn adaptive_per_second(mut peaks: Vec<Peak>, frames_per_sec: f32, target: usize) -> Vec<Peak> {
    // Sort by (bucket asc, mag desc) so we can stream-select per bucket.
    peaks.sort_unstable_by(|a, b| {
        let ba = (a.t_frame as f32 / frames_per_sec) as u32;
        let bb = (b.t_frame as f32 / frames_per_sec) as u32;
        ba.cmp(&bb).then_with(|| {
            b.mag
                .partial_cmp(&a.mag)
                .unwrap_or(core::cmp::Ordering::Equal)
        })
    });

    let mut kept = Vec::with_capacity(peaks.len());
    let mut current_bucket = u32::MAX;
    let mut count = 0usize;
    for p in peaks {
        let bucket = (p.t_frame as f32 / frames_per_sec) as u32;
        if bucket != current_bucket {
            current_bucket = bucket;
            count = 0;
        }
        if count < target {
            kept.push(p);
            count += 1;
        }
    }
    kept
}

/// 2-D rolling max, computed separably (max-along-cols → max-along-rows).
///
/// `output[r*n_cols + c] = max over the (2·kt+1) × (2·kf+1) box centred on (r, c)`,
/// clipped at the array boundary.
fn rolling_max_2d(
    input: &[f32],
    n_rows: usize,
    n_cols: usize,
    kt: usize,
    kf: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(input.len(), n_rows * n_cols);
    debug_assert_eq!(output.len(), n_rows * n_cols);

    let mut temp = vec![0.0_f32; n_rows * n_cols];
    for r in 0..n_rows {
        let row_in = &input[r * n_cols..(r + 1) * n_cols];
        let row_out = &mut temp[r * n_cols..(r + 1) * n_cols];
        rolling_max_1d(row_in, kf, row_out);
    }

    let mut col_in = vec![0.0_f32; n_rows];
    let mut col_out = vec![0.0_f32; n_rows];
    for c in 0..n_cols {
        for r in 0..n_rows {
            col_in[r] = temp[r * n_cols + c];
        }
        rolling_max_1d(&col_in, kt, &mut col_out);
        for r in 0..n_rows {
            output[r * n_cols + c] = col_out[r];
        }
    }
}

/// Lemire monotonic-deque sliding max with a half-window of `k`.
///
/// `output[i] = max(input[max(0, i-k) ..= min(n-1, i+k)])` for each `i`.
/// Total work is amortised O(n) — every index is pushed and popped at
/// most once.
fn rolling_max_1d(input: &[f32], k: usize, output: &mut [f32]) {
    let n = input.len();
    debug_assert_eq!(output.len(), n);
    if n == 0 {
        return;
    }

    let mut dq: VecDeque<usize> = VecDeque::with_capacity(n.min(2 * k + 1));

    // Forward pass: as we add input[j], settle output[j - k] when j >= k.
    for j in 0..n {
        while let Some(&back) = dq.back() {
            if input[back] <= input[j] {
                dq.pop_back();
            } else {
                break;
            }
        }
        dq.push_back(j);

        if j >= k {
            let i = j - k;
            let lower = i.saturating_sub(k);
            while let Some(&front) = dq.front() {
                if front < lower {
                    dq.pop_front();
                } else {
                    break;
                }
            }
            output[i] = input[*dq.front().unwrap()];
        }
    }

    // Tail pass: positions that didn't settle in the forward loop because
    // `j + k` would have exceeded the input length.
    let start = n.saturating_sub(k);
    for (i, slot) in output.iter_mut().enumerate().skip(start) {
        let lower = i.saturating_sub(k);
        while let Some(&front) = dq.front() {
            if front < lower {
                dq.pop_front();
            } else {
                break;
            }
        }
        *slot = input[*dq.front().unwrap()];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_max_1d(input: &[f32], k: usize) -> Vec<f32> {
        let n = input.len();
        (0..n)
            .map(|i| {
                let lo = i.saturating_sub(k);
                let hi = (i + k).min(n - 1);
                input[lo..=hi]
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .collect()
    }

    #[test]
    fn rolling_max_1d_matches_naive() {
        let inputs: &[&[f32]] = &[
            &[1.0, 2.0, 3.0, 2.0, 1.0],
            &[5.0, 1.0, 2.0, 4.0, 3.0],
            &[3.0, 3.0, 3.0, 3.0],
            &[1.0],
            &[],
        ];
        for &input in inputs {
            for k in 0..=4 {
                let mut got = vec![0.0; input.len()];
                rolling_max_1d(input, k, &mut got);
                let want = naive_max_1d(input, k);
                assert_eq!(got, want, "input={input:?}, k={k}");
            }
        }
    }

    #[test]
    fn single_peak_in_flat_zero_field() {
        let n_frames = 16;
        let n_bins = 16;
        let mut spec = vec![0.0_f32; n_frames * n_bins];
        spec[5 * n_bins + 7] = 0.9;

        let picker = PeakPicker::new(PeakPickerConfig {
            neighborhood_t: 2,
            neighborhood_f: 2,
            min_magnitude: 0.1,
            target_per_sec: 0,
        });
        let peaks = picker.pick(&spec, n_frames, n_bins, 100.0);
        assert_eq!(peaks.len(), 1);
        assert_eq!(peaks[0].t_frame, 5);
        assert_eq!(peaks[0].f_bin, 7);
        assert!((peaks[0].mag - 0.9).abs() < 1e-6);
    }

    #[test]
    fn min_magnitude_filters_low_energy() {
        let mut spec = vec![0.0_f32; 64];
        spec[10] = 0.05; // below floor
        spec[20] = 0.5; // above
        let picker = PeakPicker::new(PeakPickerConfig {
            neighborhood_t: 1,
            neighborhood_f: 1,
            min_magnitude: 0.1,
            target_per_sec: 0,
        });
        let peaks = picker.pick(&spec, 8, 8, 100.0);
        assert_eq!(peaks.len(), 1);
    }

    #[test]
    fn output_is_sorted_by_t_then_f() {
        // 4 isolated peaks in a 32x16 grid.
        let n_frames = 32;
        let n_bins = 16;
        let mut spec = vec![0.0_f32; n_frames * n_bins];
        spec[10 * n_bins + 8] = 1.0;
        spec[5 * n_bins + 12] = 0.7;
        spec[20 * n_bins + 4] = 0.5;
        spec[5 * n_bins + 2] = 0.4;

        let picker = PeakPicker::new(PeakPickerConfig {
            neighborhood_t: 1,
            neighborhood_f: 1,
            min_magnitude: 0.1,
            target_per_sec: 0,
        });
        let peaks = picker.pick(&spec, n_frames, n_bins, 100.0);
        assert_eq!(peaks.len(), 4);
        for w in peaks.windows(2) {
            assert!((w[0].t_frame, w[0].f_bin) <= (w[1].t_frame, w[1].f_bin));
        }
    }

    #[test]
    fn adaptive_per_second_caps_count() {
        // 10 well-separated peaks at frames 5, 10, …, 50 (column 4),
        // magnitudes 1..=10. neighborhood_t=2 keeps them all alive as
        // local maxima.
        let n_frames = 100;
        let n_bins = 8;
        let mut spec = vec![0.0_f32; n_frames * n_bins];
        for (i, t) in (5..=50).step_by(5).enumerate() {
            spec[t * n_bins + 4] = (i as f32) + 1.0;
        }

        let picker = PeakPicker::new(PeakPickerConfig {
            neighborhood_t: 2,
            neighborhood_f: 2,
            min_magnitude: 0.1,
            target_per_sec: 3,
        });

        // frames_per_sec=1.0 → bucket = t_frame, so every peak gets its own
        // bucket and the cap has no effect.
        let peaks_loose = picker.pick(&spec, n_frames, n_bins, 1.0);
        assert_eq!(peaks_loose.len(), 10);

        // frames_per_sec=100.0 → bucket = t/100 = 0 for all peaks (t≤50),
        // so all 10 fight over a single bucket and we keep the top 3.
        let peaks_tight = picker.pick(&spec, n_frames, n_bins, 100.0);
        assert_eq!(peaks_tight.len(), 3);
        let mut mags: Vec<f32> = peaks_tight.iter().map(|p| p.mag).collect();
        mags.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(mags, vec![8.0, 9.0, 10.0]);
    }

    #[test]
    fn empty_input_returns_empty() {
        let picker = PeakPicker::new(PeakPickerConfig::default());
        assert!(picker.pick(&[], 0, 0, 62.5).is_empty());
    }

    #[test]
    fn plateaus_emit_every_equal_cell_as_a_local_max() {
        // A 3×3 plateau of value 1.0 surrounded by zeros. Every cell in
        // the plateau is `>=` the rolling max within its neighbourhood,
        // so all 9 are picked.
        let n_frames = 9;
        let n_bins = 9;
        let mut spec = vec![0.0_f32; n_frames * n_bins];
        for t in 3..6 {
            for f in 3..6 {
                spec[t * n_bins + f] = 1.0;
            }
        }
        let picker = PeakPicker::new(PeakPickerConfig {
            neighborhood_t: 1,
            neighborhood_f: 1,
            min_magnitude: 0.1,
            target_per_sec: 0,
        });
        let peaks = picker.pick(&spec, n_frames, n_bins, 100.0);
        assert_eq!(peaks.len(), 9);
    }

    #[test]
    fn boundary_peak_at_corner_is_picked() {
        let mut spec = vec![0.0_f32; 16 * 16];
        spec[0] = 1.0;
        let picker = PeakPicker::new(PeakPickerConfig {
            neighborhood_t: 3,
            neighborhood_f: 3,
            min_magnitude: 0.1,
            target_per_sec: 0,
        });
        let peaks = picker.pick(&spec, 16, 16, 100.0);
        assert!(peaks.iter().any(|p| (p.t_frame, p.f_bin) == (0, 0)));
    }

    #[test]
    fn rolling_max_2d_matches_naive_brute_force() {
        // 8×8 input with deterministic xorshift values; verify against
        // the obvious O(N·M·K²) reference.
        let n_rows = 8;
        let n_cols = 8;
        let mut input = vec![0.0_f32; n_rows * n_cols];
        let mut x: u32 = 1;
        for v in input.iter_mut() {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *v = (x % 100) as f32;
        }

        let kt = 2;
        let kf = 2;
        let mut got = vec![0.0_f32; input.len()];
        rolling_max_2d(&input, n_rows, n_cols, kt, kf, &mut got);

        for r in 0..n_rows {
            for c in 0..n_cols {
                let r_lo = r.saturating_sub(kt);
                let r_hi = (r + kt).min(n_rows - 1);
                let c_lo = c.saturating_sub(kf);
                let c_hi = (c + kf).min(n_cols - 1);
                let mut want = f32::NEG_INFINITY;
                for rr in r_lo..=r_hi {
                    for cc in c_lo..=c_hi {
                        want = want.max(input[rr * n_cols + cc]);
                    }
                }
                assert_eq!(got[r * n_cols + c], want, "cell ({r}, {c})");
            }
        }
    }
}
