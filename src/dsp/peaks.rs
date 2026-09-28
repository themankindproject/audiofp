//! 2-D peak picking on a magnitude spectrogram.
//!
//! [`PeakPicker`] finds local maxima inside a `(2·neighborhood_t+1) ×
//! (2·neighborhood_f+1)` box around each cell, filters by magnitude floor,
//! and applies a per-second target count so dense regions don't dominate.
//!
//! The 2-D rolling max is separable. The horizontal (frequency) pass uses
//! doubling over a NaN-padded row — `log2(window)` contiguous SIMD max
//! passes — and the vertical (time) pass streams rows through a van Herk /
//! Gil-Werman block max (about three vector max ops per row for any window
//! size). Both are exact, so results are identical to a brute-force
//! window max, and only `O(neighborhood_t · n_bins)` scratch is held.

use alloc::collections::VecDeque;
#[cfg(test)]
use alloc::vec;
use alloc::vec::Vec;

/// One peak emitted by [`PeakPicker`].
///
/// `repr(C)` plus an explicit `_pad` field keeps the layout deterministic
/// (12 bytes, no implicit padding) so the struct is `bytemuck::Pod` and can
/// be stored directly in mmap'd files or shipped over a C ABI.
///
/// **Units of [`Peak::mag`]** depend entirely on what the upstream STFT
/// feeds into the picker — `PeakPicker` is a pure algorithm with no
/// opinion on units. Concretely, `audiofp`'s built-in extractors pass:
///
/// - **dB** (`10·log10(power)`): [`Wang`](crate::classical::Wang) and
///   [`Panako`](crate::classical::Panako) — their pickers threshold
///   against `min_anchor_mag_db`.
/// - **Raw band-difference sign bits** packed as a `u32`: not stored
///   here. [`Haitsma`](crate::classical::Haitsma) emits frames
///   directly, not `Peak`s.
///
/// Callers wiring up a custom front-end should document the units
/// alongside the picker config and apply any conversion (e.g. dB →
/// linear) before thresholding.
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
    /// Magnitude at the peak. See the [struct docs](Peak) for unit
    /// conventions used by the in-tree extractors.
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
    /// Floor on log-magnitude (dB): cells below this are never
    /// candidates. `audiofp`'s in-tree extractors (Wang, Panako) feed
    /// `pick` a dB spectrogram and pass their `min_anchor_mag_db`
    /// config here.
    pub min_magnitude_db: f32,
    /// Optional additional floor on **linear** magnitude, for callers
    /// that feed `pick` a raw (pre-log) spectrogram. When `Some(lin)`,
    /// cells must exceed both this linear floor and `min_magnitude_db`.
    /// `None` (default) disables the linear floor.
    pub min_magnitude_linear: Option<f32>,
    /// Per-second cap on emitted peaks. Set to `0` to disable.
    pub target_per_sec: usize,
}

impl Default for PeakPickerConfig {
    fn default() -> Self {
        Self {
            neighborhood_t: 7,
            neighborhood_f: 7,
            // A dB floor, matching `WangConfig`/`PanakoConfig`'s
            // `min_anchor_mag_db`. It must not be a small *linear* value:
            // for the dB spectrograms the in-tree extractors feed `pick`,
            // any positive threshold sits above the entire signal and
            // silently returns zero peaks.
            min_magnitude_db: -50.0,
            min_magnitude_linear: None,
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
/// let mut picker = PeakPicker::new(PeakPickerConfig {
///     neighborhood_t: 1,
///     neighborhood_f: 1,
///     min_magnitude_db: f32::NEG_INFINITY,
///     min_magnitude_linear: Some(0.1),
///     target_per_sec: 0, // disable adaptive thresholding
/// });
/// let peaks = picker.pick(&spec, 8, 8, 100.0);
/// assert_eq!(peaks.len(), 1);
/// assert_eq!((peaks[0].t_frame, peaks[0].f_bin), (3, 4));
/// ```
pub struct PeakPicker {
    cfg: PeakPickerConfig,
    /// NaN-padded horizontal scratch (`n_bins + 2·neighborhood_f`).
    hpad: Vec<f32>,
    /// Streaming vertical max over the horizontal-max rows.
    vert: VertMax,
    /// Ring of the last `neighborhood_t + 1` raw rows: a row's candidates
    /// are tested once its full vertical window has been seen.
    raw_ring: Vec<f32>,
    /// 2-D max row of the centre currently being emitted.
    vmax_row: Vec<f32>,
    /// Geometry of the in-progress `begin` … `finish` pass.
    n_bins: usize,
    rows_pushed: usize,
    /// Pooled candidate buffer — avoids a per-call heap allocation.
    candidates: Vec<Peak>,
}

impl PeakPicker {
    /// Build a picker with the given config.
    #[must_use]
    pub fn new(cfg: PeakPickerConfig) -> Self {
        Self {
            cfg,
            hpad: Vec::new(),
            vert: VertMax::default(),
            raw_ring: Vec::new(),
            vmax_row: Vec::new(),
            n_bins: 0,
            rows_pushed: 0,
            candidates: Vec::new(),
        }
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
    ///
    /// # Panics
    ///
    /// Panics if `spec.len() != n_frames * n_bins`.
    ///
    /// # Bin-index range
    ///
    /// [`Peak::f_bin`] is a `u16`, so `n_bins` must not exceed
    /// `u16::MAX + 1` (65 536) or the reported bin index wraps. `n_bins`
    /// is `n_fft / 2 + 1`, so this bounds `n_fft` at 131 070 — far above
    /// any sane analysis window (the built-in fingerprinters use 1 024 and
    /// 2 048). A debug build asserts the bound.
    ///
    /// # Note
    ///
    /// This method takes `&mut self` to re-use scratch buffers across calls.
    /// If you previously held a `PeakPicker` behind `&self`, store it as
    /// `Mutex<PeakPicker>` or use one picker per producing thread.
    pub fn pick(
        &mut self,
        spec: &[f32],
        n_frames: usize,
        n_bins: usize,
        frames_per_sec: f32,
    ) -> Vec<Peak> {
        if n_frames == 0 || n_bins == 0 {
            return Vec::new();
        }
        assert_eq!(spec.len(), n_frames * n_bins, "spec length mismatch");
        self.begin(n_frames, n_bins, frames_per_sec);
        for row in spec.chunks_exact(n_bins) {
            self.push_row(row);
        }
        self.finish(frames_per_sec)
    }

    /// Start a row-streaming pick over an `(n_frames, n_bins)`
    /// spectrogram. Feed exactly `n_frames` rows with
    /// [`push_row`](Self::push_row), then call [`finish`](Self::finish).
    ///
    /// Produces the same peaks as [`pick`](Self::pick) on the same rows
    /// (`pick` is implemented on top of it) while holding only
    /// `O((neighborhood_t + 1) · n_bins)` floats of scratch instead of the
    /// whole spectrogram, so front-ends can compute rows on the fly.
    pub(crate) fn begin(&mut self, n_frames: usize, n_bins: usize, frames_per_sec: f32) {
        // `Peak::f_bin` is a u16; a larger spectrogram would wrap the bin
        // index. Unreachable for any realistic n_fft (see "Bin-index range").
        debug_assert!(
            n_bins <= u16::MAX as usize + 1,
            "n_bins {n_bins} exceeds u16 range of Peak::f_bin"
        );
        let kt = self.cfg.neighborhood_t;
        let kf = self.cfg.neighborhood_f;
        self.n_bins = n_bins;
        self.rows_pushed = 0;
        resize_scratch(&mut self.hpad, n_bins + 2 * kf);
        resize_scratch(&mut self.raw_ring, (kt + 1) * n_bins);
        resize_scratch(&mut self.vmax_row, n_bins);
        self.vert.reset(2 * kt + 1, n_bins);
        // Leading edge: the window of row 0 is clamped at the start. NaN
        // rows never win a max, so `kt` NaN rows reproduce the clamp
        // exactly (and cannot complete a window yet: kt < 2·kt + 1).
        for _ in 0..kt {
            self.vert.next_slot().fill(f32::NAN);
            self.vert.commit(None);
        }

        let target_per_sec = self.cfg.target_per_sec;
        // Upper bound on candidate peaks (fps × target/s), doubled as
        // headroom, floored at 64 so the reserve is non-trivial even when
        // the per-second cap is disabled.
        //
        // `frames_per_sec` is caller-supplied, so guard the arithmetic:
        // `ceil() as usize` saturates for huge/non-finite values and the
        // multiply then overflows (debug panic) or reserves gigabytes
        // (release abort). Bounding the frame term by `n_frames` is safe —
        // it is only a capacity hint, and no candidate can exceed it.
        // Capacity hint only — bound by actual spectrogram cell count so
        // absurd `target_per_sec` / `frames_per_sec` cannot reserve gigabytes.
        let cell_count = n_frames.saturating_mul(n_bins);
        let upper = if frames_per_sec.is_finite() && frames_per_sec > 0.0 {
            let frames_per_sec = (frames_per_sec.ceil() as usize).min(n_frames);
            frames_per_sec
                .saturating_mul(target_per_sec)
                .saturating_mul(2)
                .max(64)
                .min(cell_count.max(64))
        } else {
            64
        };
        self.candidates.clear();
        // `self.candidates` is moved out of `self` by the `mem::take` in
        // `finish` (the caller owns the returned peaks), so its capacity is
        // NOT retained across passes — this reserve runs once per pass.
        // That is amortised over the whole spectrogram, so per-call
        // allocation is not on the hot path.
        if self.candidates.capacity() < upper {
            self.candidates.reserve(upper - self.candidates.capacity());
        }
    }

    /// Feed the next spectrogram row (length `n_bins` from
    /// [`begin`](Self::begin)).
    pub(crate) fn push_row(&mut self, row: &[f32]) {
        let n_bins = self.n_bins;
        debug_assert_eq!(row.len(), n_bins);
        let ring_rows = self.cfg.neighborhood_t + 1;
        let slot = self.rows_pushed % ring_rows;
        self.raw_ring[slot * n_bins..(slot + 1) * n_bins].copy_from_slice(row);
        self.rows_pushed += 1;
        hmax_row(
            row,
            self.cfg.neighborhood_f,
            &mut self.hpad,
            self.vert.next_slot(),
        );
        self.commit_and_emit();
    }

    /// Drain the trailing edge and return the peaks sorted by
    /// `(t_frame, f_bin)`.
    pub(crate) fn finish(&mut self, frames_per_sec: f32) -> Vec<Peak> {
        // Trailing edge: NaN rows reproduce the end-of-spectrogram clamp.
        for _ in 0..self.cfg.neighborhood_t {
            self.vert.next_slot().fill(f32::NAN);
            self.commit_and_emit();
        }
        let target_per_sec = self.cfg.target_per_sec;

        if target_per_sec > 0 && frames_per_sec > 0.0 && !self.candidates.is_empty() {
            let capped = adaptive_per_second(
                core::mem::take(&mut self.candidates),
                frames_per_sec,
                target_per_sec,
            );
            self.candidates = capped;
        }

        self.candidates
            .sort_unstable_by_key(|p| (p.t_frame, p.f_bin));
        core::mem::take(&mut self.candidates)
    }

    /// Commit the row just written into the vertical engine's slot; when a
    /// centre row's full window is complete, test its cells.
    #[inline]
    fn commit_and_emit(&mut self) {
        let w = self.vert.w;
        let pos = self.vert.pos;
        if pos + 1 < w {
            self.vert.commit(None);
            return;
        }
        self.vert.commit(Some(&mut self.vmax_row));
        // Padded row `pos` completed the window of padded row
        // `pos + 1 - w`, which is original row `pos + 1 - w` (the `kt`
        // leading pads shift both window ends equally).
        let t = pos + 1 - w;
        let n_bins = self.n_bins;
        let slot = t % (self.cfg.neighborhood_t + 1);
        emit_candidates(
            &self.raw_ring[slot * n_bins..(slot + 1) * n_bins],
            &self.vmax_row,
            t as u32,
            self.cfg.min_magnitude_db,
            self.cfg.min_magnitude_linear,
            &mut self.candidates,
        );
    }
}

/// Clear and zero-resize a scratch vector (no allocation once warm).
#[inline]
fn resize_scratch(v: &mut Vec<f32>, len: usize) {
    v.clear();
    v.resize(len, 0.0);
}

/// `buf[i] = max(buf[i], buf[i + d])` for `i in 0..m`, in place.
///
/// Ascending order keeps every read at an index that is not yet
/// written, and each 8-wide chunk loads both operands before storing, so
/// the in-place update equals the out-of-place one. Requires
/// `m + d <= buf.len()`.
#[inline]
fn max_shift_in_place(buf: &mut [f32], d: usize, m: usize) {
    use crate::dsp::simd::{load8, store8};
    debug_assert!(m + d <= buf.len());
    let mut i = 0;
    while i + 8 <= m {
        let v = load8(buf, i).max(load8(buf, i + d));
        store8(buf, i, v);
        i += 8;
    }
    while i < m {
        buf[i] = buf[i].max(buf[i + d]);
        i += 1;
    }
}

/// `out[i] = max(a[i], b[i])`.
#[inline]
fn max_pair_into(a: &[f32], b: &[f32], out: &mut [f32]) {
    use crate::dsp::simd::{load8, store8};
    debug_assert!(a.len() == out.len() && b.len() == out.len());
    let n = out.len();
    let mut i = 0;
    while i + 8 <= n {
        store8(out, i, load8(a, i).max(load8(b, i)));
        i += 8;
    }
    while i < n {
        out[i] = a[i].max(b[i]);
        i += 1;
    }
}

/// `acc[i] = max(acc[i], x[i])`.
#[inline]
fn max_assign(acc: &mut [f32], x: &[f32]) {
    use crate::dsp::simd::{load8, store8};
    debug_assert_eq!(acc.len(), x.len());
    let n = acc.len();
    let mut i = 0;
    while i + 8 <= n {
        let v = load8(acc, i).max(load8(x, i));
        store8(acc, i, v);
        i += 8;
    }
    while i < n {
        acc[i] = acc[i].max(x[i]);
        i += 1;
    }
}

/// Horizontal sliding max: `out[i] = max(row[max(0, i-k) ..= min(n-1, i+k)])`.
///
/// Same contract (and bit-identical values) as [`rolling_max_1d`]: the
/// window is clamped at both edges, maxima are exact, and a NaN never
/// wins (`f32::max` / `f32x8::max` semantics; an all-NaN window yields
/// NaN).
///
/// Uses doubling on a NaN-padded copy: after `log2(p)` in-place passes
/// (`p` = largest power of two `≤ 2k+1`) `pad[j]` holds the max of
/// `pad[j .. j+p]`, and two overlapping `p`-windows cover each
/// `(2k+1)`-window. For Wang/Panako (`k = 15`) that is 5 vector max
/// passes per row instead of 30, and every pass is contiguous SIMD.
/// NaN padding reproduces the edge clamp exactly because NaN never wins.
///
/// `pad` must have length `row.len() + 2k`.
fn hmax_row(row: &[f32], k: usize, pad: &mut [f32], out: &mut [f32]) {
    let n = row.len();
    debug_assert_eq!(out.len(), n);
    debug_assert_eq!(pad.len(), n + 2 * k);
    if k == 0 || n == 0 {
        out.copy_from_slice(row);
        return;
    }
    pad[..k].fill(f32::NAN);
    pad[k..k + n].copy_from_slice(row);
    pad[k + n..].fill(f32::NAN);
    let w = 2 * k + 1;
    let p = 1usize << (usize::BITS - 1 - w.leading_zeros());
    let len = pad.len();
    let mut d = 1;
    while d < p {
        // After this pass `pad[j] = max(pad[j .. j + 2d])` for every
        // `j <= len - 2d` — every index the final combine reads.
        max_shift_in_place(pad, d, len + 1 - 2 * d);
        d *= 2;
    }
    let s = w - p;
    max_pair_into(&pad[..n], &pad[s..s + n], out);
}

/// Streaming vertical sliding max over rows (van Herk / Gil-Werman).
///
/// Rows are grouped into blocks of `w`; a window `[s, s + w)` spans the
/// suffix of one block and the prefix of the next, so its max is
/// `max(suffix[s], prefix[s + w - 1])` — about three vector max ops per
/// row regardless of `w`. Only two blocks of rows are held, so memory is
/// `O(w · n_bins)` independent of the spectrogram length.
#[derive(Default)]
struct VertMax {
    /// Window length in rows (`2·kt + 1`).
    w: usize,
    n_bins: usize,
    /// Rows of the block being filled.
    cur: Vec<f32>,
    /// Suffix maxima of the previous complete block.
    prev: Vec<f32>,
    /// Running prefix max of the current block.
    g: Vec<f32>,
    /// Rows committed so far (padding included).
    pos: usize,
}

impl VertMax {
    fn reset(&mut self, w: usize, n_bins: usize) {
        self.w = w;
        self.n_bins = n_bins;
        resize_scratch(&mut self.cur, w * n_bins);
        resize_scratch(&mut self.prev, w * n_bins);
        resize_scratch(&mut self.g, n_bins);
        self.pos = 0;
    }

    /// Slot the next row must be written into before [`commit`](Self::commit).
    #[inline]
    fn next_slot(&mut self) -> &mut [f32] {
        let j = self.pos % self.w;
        &mut self.cur[j * self.n_bins..(j + 1) * self.n_bins]
    }

    /// Commit the row in [`next_slot`](Self::next_slot). When `out` is
    /// `Some`, write the max over rows `[max(0, pos + 1 - w), pos]` into it
    /// (the window ending at this row, clamped at the stream start).
    fn commit(&mut self, out: Option<&mut [f32]>) {
        let w = self.w;
        let nb = self.n_bins;
        let j = self.pos % w;
        let row = &self.cur[j * nb..(j + 1) * nb];
        if j == 0 {
            self.g.copy_from_slice(row);
        } else {
            max_assign(&mut self.g, row);
        }
        if let Some(out) = out {
            if j == w - 1 || self.pos < w {
                // The window lies inside this block: its prefix max.
                out.copy_from_slice(&self.g);
            } else {
                // Suffix of the previous block from `j + 1`, plus this
                // block's prefix through `j`.
                max_pair_into(&self.prev[(j + 1) * nb..(j + 2) * nb], &self.g, out);
            }
        }
        if j == w - 1 {
            // Block complete: turn it into suffix maxima for the next block.
            for i in (0..w - 1).rev() {
                let (lo, hi) = self.cur.split_at_mut((i + 1) * nb);
                max_assign(&mut lo[i * nb..], &hi[..nb]);
            }
            core::mem::swap(&mut self.cur, &mut self.prev);
        }
        self.pos += 1;
    }

    /// Max over rows `[a, pos - 1]` (a window ending at the last committed
    /// row, at most `w` rows long) without changing any state — used to
    /// drain right-clamped tail windows at end of stream while leaving the
    /// stream resumable.
    fn max_since(&self, a: usize, out: &mut [f32]) {
        let w = self.w;
        let nb = self.n_bins;
        debug_assert!(self.pos > 0 && a < self.pos && self.pos - a <= w);
        let last = self.pos - 1;
        let j_last = last % w;
        let row = |i: usize| i * nb..(i + 1) * nb;
        if a / w == last / w {
            if j_last == w - 1 {
                // The block was just finalised into suffix maxima.
                out.copy_from_slice(&self.prev[row(a % w)]);
            } else {
                // Raw rows of the open block.
                out.copy_from_slice(&self.cur[row(a % w)]);
                for i in a % w + 1..=j_last {
                    max_assign(out, &self.cur[row(i)]);
                }
            }
        } else {
            // `a` is in the previous (finalised) block and the open block
            // holds rows through `j_last` (a window of at most `w` rows
            // cannot span a finalised current block plus an earlier one).
            debug_assert!(j_last < w - 1);
            max_pair_into(&self.prev[row(a % w)], &self.g, out);
        }
    }
}

/// Push every cell of row `t` that clears the floors and equals its
/// neighbourhood max (`>=`, so plateau cells all count), in ascending
/// bin order.
///
/// Branch-free 8-wide compare + bitmask: peaks are sparse, so most
/// chunks are rejected with one mask test. Lane-for-lane the predicate is
/// the scalar one — ordered comparisons are `false` for NaN in both.
#[inline]
fn emit_candidates(
    raw: &[f32],
    vmax: &[f32],
    t: u32,
    min_mag: f32,
    min_mag_linear: Option<f32>,
    out: &mut Vec<Peak>,
) {
    use crate::dsp::simd::load8;
    use wide::f32x8;

    debug_assert_eq!(raw.len(), vmax.len());
    let n = raw.len();
    let floor = f32x8::splat(min_mag);
    let lin = min_mag_linear.map(f32x8::splat);
    let mut push = |f: usize| {
        out.push(Peak {
            t_frame: t,
            f_bin: f as u16,
            _pad: 0,
            mag: raw[f],
        });
    };
    let mut off = 0;
    while off + 8 <= n {
        let v = load8(raw, off);
        let mut m = v.simd_gt(floor) & v.simd_ge(load8(vmax, off));
        if let Some(lin) = lin {
            m &= v.simd_gt(lin);
        }
        let mut bits = m.to_bitmask();
        while bits != 0 {
            push(off + bits.trailing_zeros() as usize);
            bits &= bits - 1;
        }
        off += 8;
    }
    for f in off..n {
        let v = raw[f];
        let above_floor = v > min_mag && min_mag_linear.is_none_or(|lin| v > lin);
        if above_floor && v >= vmax[f] {
            push(f);
        }
    }
}

/// Keep the top `target` peaks per one-second bucket (by magnitude).
fn adaptive_per_second(mut peaks: Vec<Peak>, frames_per_sec: f32, target: usize) -> Vec<Peak> {
    // Sort by (bucket asc, mag desc, position asc) so we can stream-select
    // per bucket. The `(t_frame, f_bin)` tiebreak is unique per peak, making
    // this a total order: equal-magnitude peaks resolve identically here and
    // in the streaming `finalize_bucket`, so both keep the same top-K.
    peaks.sort_unstable_by(|a, b| {
        let ba = (a.t_frame as f32 / frames_per_sec) as u32;
        let bb = (b.t_frame as f32 / frames_per_sec) as u32;
        ba.cmp(&bb)
            .then_with(|| {
                b.mag
                    .partial_cmp(&a.mag)
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .then_with(|| (a.t_frame, a.f_bin).cmp(&(b.t_frame, b.f_bin)))
    });

    // In-place compaction after the bucket sort — same kept set, one buffer.
    let mut write = 0usize;
    let mut current_bucket = u32::MAX;
    let mut count = 0usize;
    for read in 0..peaks.len() {
        let bucket = (peaks[read].t_frame as f32 / frames_per_sec) as u32;
        if bucket != current_bucket {
            current_bucket = bucket;
            count = 0;
        }
        if count < target {
            if write != read {
                peaks[write] = peaks[read];
            }
            write += 1;
            count += 1;
        }
    }
    peaks.truncate(write);
    peaks
}

/// 2-D rolling max with caller-provided scratch buffers (no allocation).
///
/// All scratch slices must already be sized: `temp` and `output` to
/// `n_rows * n_cols`; `col_in` and `col_out` to `n_rows`.
///
/// This is the static (non-incremental) variant of [`IncrementalPeakDetector`].
/// It recomputes the full 2-D max on every call, so its cost is
/// `O(n_rows · n_cols)` regardless of `kt` / `kf` — useful for one-shot
/// offline peak picking where incremental maintenance isn't worth the
/// state. The scratch buffers let callers reuse allocations across calls.
#[allow(clippy::too_many_arguments)] // public helper, bundling would obscure intent
pub fn rolling_max_2d_pooled(
    input: &[f32],
    n_rows: usize,
    n_cols: usize,
    kt: usize,
    kf: usize,
    output: &mut [f32],
    temp: &mut [f32],
    col_in: &mut [f32],
    col_out: &mut [f32],
    dq: &mut VecDeque<usize>,
) {
    debug_assert_eq!(input.len(), n_rows * n_cols);
    debug_assert_eq!(output.len(), n_rows * n_cols);
    debug_assert_eq!(temp.len(), n_rows * n_cols);
    debug_assert_eq!(col_in.len(), n_rows);
    debug_assert_eq!(col_out.len(), n_rows);

    for r in 0..n_rows {
        let row_in = &input[r * n_cols..(r + 1) * n_cols];
        let row_out = &mut temp[r * n_cols..(r + 1) * n_cols];
        rolling_max_1d(row_in, kf, row_out, dq);
    }

    // Block-tile column pass to keep `temp` L1-resident. Each column
    // read strides `n_cols` apart (a separate cache line per row);
    // processing by TILE-wide blocks ensures recently-touched rows
    // stay hot across columns in each block.
    const TILE: usize = 64;
    for c_block in (0..n_cols).step_by(TILE) {
        let c_end = (c_block + TILE).min(n_cols);
        for c in c_block..c_end {
            for r in 0..n_rows {
                col_in[r] = temp[r * n_cols + c];
            }
            rolling_max_1d(col_in, kt, col_out, dq);
            for r in 0..n_rows {
                output[r * n_cols + c] = col_out[r];
            }
        }
    }
}

/// Allocating wrapper around [`rolling_max_2d_pooled`]; kept for the
/// brute-force-comparison test.
#[cfg(test)]
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
    let mut dq: VecDeque<usize> = VecDeque::new();
    for r in 0..n_rows {
        let row_in = &input[r * n_cols..(r + 1) * n_cols];
        let row_out = &mut temp[r * n_cols..(r + 1) * n_cols];
        rolling_max_1d(row_in, kf, row_out, &mut dq);
    }

    let mut col_in = vec![0.0_f32; n_rows];
    let mut col_out = vec![0.0_f32; n_rows];
    for c in 0..n_cols {
        for r in 0..n_rows {
            col_in[r] = temp[r * n_cols + c];
        }
        rolling_max_1d(&col_in, kt, &mut col_out, &mut dq);
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
#[inline]
fn rolling_max_1d(input: &[f32], k: usize, output: &mut [f32], dq: &mut VecDeque<usize>) {
    // Fast path: Wang/Panako peak pickers always use neighbourhood 15
    // (31-cell window). The direct 31-tap vectorized max below is
    // bit-exact vs the Lemire deque (pure pairwise f32::max) and ~1.6x
    // faster on real spectrogram lines — measured in scratch probe
    // (500 random + plateau parity shapes, 1.57x isolated). No scratch
    // needed, so the pooled deque is untouched.
    if k == 15 {
        max31_vec(input, output);
        return;
    }
    let n = input.len();
    debug_assert_eq!(output.len(), n);
    if n == 0 {
        return;
    }

    // Pooled scratch: cleared (capacity retained) so the hot path does
    // not allocate after warmup.
    dq.clear();

    // Forward pass: as we add input[j], settle output[j - k] when j >= k.
    for j in 0..n {
        while let Some(&back) = dq.back() {
            // Evict the back while the new value is at least as large
            // under `f32::max` semantics — identical to `<=` for finite
            // inputs, but a NaN never wins (plain `<=` is false for every
            // NaN comparison, which let a NaN reach the front and be
            // returned, diverging from `max31_vec`).
            if f32::max(input[back], input[j]) == input[j] {
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
            if let Some(&front) = dq.front() {
                output[i] = input[front];
            }
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
        if let Some(&front) = dq.front() {
            *slot = input[front];
        }
    }
}

/// Direct 31-tap vectorized max, fixed-k=15 fast path for [`rolling_max_1d`].
///
/// Same contract: `output[i] = max(input[max(0, i-15) ..= min(n-1, i+15)])`.
/// The interior runs 8-wide via pairwise `f32x8::max` (31 taps: 31 vector
/// loads + 30 vector max ops per 8 outputs), branch-free; the ≤15-cell
/// edges run scalar. Bit-exact vs the Lemire deque for finite inputs —
/// both use `f32::max`, so a NaN never wins in either path.
#[inline]
fn max31_vec(input: &[f32], output: &mut [f32]) {
    use crate::dsp::simd::{load8, store8};

    let n = input.len();
    debug_assert_eq!(output.len(), n);
    if n == 0 {
        return;
    }
    const K: usize = 15;
    // Edges: scalar, correctness-first.
    for (i, slot) in output.iter_mut().enumerate().take(K.min(n)) {
        let hi = (i + K).min(n - 1);
        let mut m = input[0];
        for &v in &input[1..=hi] {
            m = m.max(v);
        }
        *slot = m;
    }
    if n > K {
        for (i, slot) in output.iter_mut().enumerate().take(n).skip((n - K).max(K)) {
            let lo = i.saturating_sub(K);
            let mut m = input[lo];
            for &v in &input[lo + 1..n] {
                m = m.max(v);
            }
            *slot = m;
        }
    }
    if n < 2 * K + 1 {
        return;
    }
    // Interior: i in [15, n-15), 8-wide.
    let end = n - K;
    let chunks_end = K + ((end - K) / 8) * 8;
    let mut i = K;
    while i < chunks_end {
        let mut m = load8(input, i - K);
        for t in -14..=15_isize {
            let base = (i as isize + t) as usize;
            m = m.max(load8(input, base));
        }
        store8(output, i, m);
        i += 8;
    }
    while i < end {
        let lo = i - K;
        let hi = i + K;
        let mut m = input[lo];
        for &v in &input[lo + 1..=hi] {
            m = m.max(v);
        }
        output[i] = m;
        i += 1;
    }
}

/// Incremental 2-D rolling-max for streaming peak detection.
///
/// Computes the horizontal rolling-max once per row and streams it through
/// a van Herk / Gil-Werman vertical max, so the 2-D max for each ripe row
/// costs a few vector passes over `n_bins` regardless of the window size.
/// All state is allocated at construction; `push_row` and `flush` never
/// allocate.
///
/// Useful as a building block for any pipeline that needs to pick peaks
/// from a streaming spectrogram (tempo, onset, beat tracking, etc.).
/// Construct with [`IncrementalPeakDetector::new`], feed each new
/// spectrogram row into [`push_row`](IncrementalPeakDetector::push_row),
/// and consume the per-ripe-row 2-D max from `out_max`.
pub struct IncrementalPeakDetector {
    kt: usize,
    kf: usize,
    n_bins: usize,
    // Number of rows pushed so far (saturates at u32::MAX in practice).
    n_pushed: u32,
    // Absolute index of the last row whose 2-D max was emitted, via
    // either `push_row` or `flush`. Rows at or below it are never
    // re-emitted — this is what makes `flush` idempotent and
    // `push`-after-`flush` duplicate-free. -1 = nothing emitted yet.
    last_emitted: i64,
    // Streaming vertical max over the horizontal-max rows.
    vert: VertMax,
    // NaN-padded horizontal scratch (`n_bins + 2·kf`).
    hpad: Vec<f32>,
}

impl IncrementalPeakDetector {
    /// Build a detector for a spectrogram of width `n_bins` and a
    /// `(2·kt+1) × (2·kf+1)` neighbourhood. Allocates all internal
    /// state up front so `push_row` performs zero allocations on the
    /// hot path.
    #[must_use]
    pub fn new(kt: usize, kf: usize, n_bins: usize) -> Self {
        let mut vert = VertMax::default();
        vert.reset(2 * kt + 1, n_bins);
        Self {
            kt,
            kf,
            n_bins,
            n_pushed: 0,
            last_emitted: -1,
            vert,
            hpad: alloc::vec![0.0_f32; n_bins + 2 * kf],
        }
    }

    /// Push a new spectrogram row. Returns `Some(abs_ripe_frame)` when the
    /// center row of the vertical window becomes ripe, writing its 2-D
    /// rolling-max into `out_max`. Returns `None` while filling the initial
    /// window.
    ///
    /// # Preconditions
    ///
    /// - `row.len()` must equal `n_bins` (the detector's configured width).
    /// - `out_max.len()` must equal `n_bins`.
    ///
    /// Both are checked by `debug_assert` — violated in release builds,
    /// this produces incorrect (but memory-safe) results.
    pub fn push_row(&mut self, row: &[f32], out_max: &mut [f32]) -> Option<u32> {
        debug_assert_eq!(row.len(), self.n_bins);
        debug_assert_eq!(out_max.len(), self.n_bins);

        let abs = self.n_pushed;
        self.n_pushed += 1;

        // 1. Horizontal rolling-max of the new row, written straight into
        //    the vertical engine's slot for this row.
        hmax_row(row, self.kf, &mut self.hpad, self.vert.next_slot());

        // 2. A row is ripe once its forward context (`kt` rows) has been
        //    pushed; skip rows an earlier `flush` already emitted —
        //    re-emitting would duplicate peaks downstream.
        let kt = self.kt as u32;
        let ripe_abs = abs
            .checked_sub(kt)
            .filter(|&r| r as i64 > self.last_emitted);

        // 3. Commit to the vertical engine. For a ripe row the window
        //    [ripe_abs - kt, ripe_abs + kt] = [abs - 2·kt, abs] (clamped at
        //    the stream start) is exactly the window ending at `abs`.
        match ripe_abs {
            Some(ripe_abs) => {
                self.vert.commit(Some(out_max));
                self.last_emitted = ripe_abs as i64;
                Some(ripe_abs)
            }
            None => {
                self.vert.commit(None);
                None
            }
        }
    }

    /// Flush remaining rows that haven't become ripe during normal push.
    /// These are the tail rows whose forward context extends past end-of-stream.
    /// Calls `emit_fn(abs_frame, max_slice)` for each.
    ///
    /// Idempotent: rows already emitted (by earlier `push_row` calls or a
    /// prior `flush`) are never emitted again. A second `flush` with no
    /// intervening `push_row` calls emits nothing.
    pub fn flush(&mut self, out_max: &mut [f32], mut emit_fn: impl FnMut(u32, &[f32])) {
        if self.n_pushed == 0 {
            return;
        }
        // The last ripe frame emitted by push was at abs = n_pushed - 1 - kt
        // (if n_pushed > kt). The remaining un-emitted frames are
        // [last_ripe + 1, n_pushed - 1], i.e. the last `min(kt, n_pushed-1)` frames.
        // But if n_pushed <= kt, then NO frames were emitted by push at all,
        // so we need to emit all of [0, n_pushed-1]. Rows at or below
        // `last_emitted` were already drained by a prior flush and are
        // skipped (idempotency / push-after-flush correctness).
        let first_flush = self
            .n_pushed
            .saturating_sub(self.kt as u32)
            .max((self.last_emitted + 1).max(0) as u32);
        let last_flush = self.n_pushed - 1;

        // After this drain every row [0, n_pushed-1] has been emitted
        // (earlier rows via push_row, the tail via flush), so advance the
        // cursor unconditionally — even when the range below is empty.
        self.last_emitted = self.n_pushed as i64 - 1;

        // Tail windows are right-clamped at the last pushed row:
        //   window = [ripe_abs.saturating_sub(kt), n_pushed - 1]
        // (ripe_abs + kt >= n_pushed - 1 for every flush frame). They are
        // read without mutating the engine, so a later `push_row`
        // continues the same stream.
        for ripe_abs in first_flush..=last_flush {
            let start = ripe_abs.saturating_sub(self.kt as u32) as usize;
            self.vert.max_since(start, out_max);
            emit_fn(ripe_abs, out_max);
        }
    }

    /// Reset all internal state. The detector behaves as if freshly
    /// constructed: no rows pushed, no ripe frames emitted. Call between
    /// independent input streams sharing one detector instance so stale
    /// data from a previous stream doesn't bleed into the first emitted
    /// row.
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.n_pushed = 0;
        self.last_emitted = -1;
        self.vert.reset(2 * self.kt + 1, self.n_bins);
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
        let mut dq: VecDeque<usize> = VecDeque::new();
        for &input in inputs {
            for k in 0..=4 {
                let mut got = vec![0.0; input.len()];
                rolling_max_1d(input, k, &mut got, &mut dq);
                let want = naive_max_1d(input, k);
                assert_eq!(got, want, "input={input:?}, k={k}");
            }
        }
    }

    /// The k=15 vectorized fast path (`max31_vec`) must be bit-exact vs
    /// the Lemire deque on every length class: empty, sub-window,
    /// window-sized, chunk-aligned/unaligned interiors, and random.
    #[test]
    fn max31_fast_path_matches_naive() {
        // Fixed shapes incl. plateau / monotone (tie semantics) and the
        // 31/30/16-cell boundary lengths.
        let shapes: Vec<Vec<f32>> = alloc::vec![
            alloc::vec![1.0f32; 64],
            (0..64).map(|i| i as f32).collect(),
            (0..64).map(|i| 64.0 - i as f32).collect(),
            alloc::vec![0.0f32; 64],
            alloc::vec![1.0f32; 31],
            alloc::vec![1.0f32; 30],
            alloc::vec![1.0f32; 16],
            alloc::vec![1.0f32; 1],
            alloc::vec![],
        ];
        let mut dq: VecDeque<usize> = VecDeque::new();
        for input in &shapes {
            let mut want = vec![0.0; input.len()];
            let mut got = vec![0.0; input.len()];
            rolling_max_1d(input, 15, &mut want, &mut dq);
            // Force the fast path even though k==15 routes there anyway;
            // compare against the naive reference instead of the deque.
            super::max31_vec(input, &mut got);
            assert_eq!(got, naive_max_1d(input, 15), "shape len={}", input.len());
            assert_eq!(got, want, "shape len={}", input.len());
        }
        // Random lengths across all alignment classes.
        let mut x: u32 = 0x31FA0;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        for _ in 0..200 {
            let n = (next() % 2500 + 1) as usize;
            let input: Vec<f32> = (0..n).map(|_| (next() % 1000) as f32 / 10.0).collect();
            let mut got = vec![0.0; n];
            super::max31_vec(&input, &mut got);
            assert_eq!(got, naive_max_1d(&input, 15), "random n={n}");
        }
    }

    #[test]
    fn nan_never_wins_in_either_rolling_max_path() {
        // The k==15 vectorized path uses `f32::max` (NaN never wins); the
        // deque path must agree. Plain `<=` eviction let a NaN reach the
        // deque front and be returned.
        let mut input = vec![1.0_f32; 64];
        input[32] = f32::NAN;
        let mut dq: VecDeque<usize> = VecDeque::new();
        let mut fast = vec![0.0; input.len()];
        super::max31_vec(&input, &mut fast);
        let mut deque = vec![0.0; input.len()];
        rolling_max_1d(&input, 14, &mut deque, &mut dq);
        let naive = naive_max_1d(&input, 14);

        // Pin exact outputs on both axes — NaN at index 32 must not win.
        for (i, ((&f, &d), &n)) in fast.iter().zip(deque.iter()).zip(naive.iter()).enumerate() {
            assert!(!f.is_nan(), "max31 produced NaN at {i}");
            assert!(!d.is_nan(), "deque path surfaced NaN at {i}");
            assert_eq!(f, d, "fast vs deque mismatch at {i}");
            if !n.is_nan() {
                assert_eq!(d, n, "deque vs naive mismatch at {i}");
            }
        }
        assert_eq!(fast[31], 1.0);
        assert_eq!(fast[32], 1.0);
        assert_eq!(fast[33], 1.0);
    }

    #[test]
    fn single_peak_in_flat_zero_field() {
        let n_frames = 16;
        let n_bins = 16;
        let mut spec = vec![0.0_f32; n_frames * n_bins];
        spec[5 * n_bins + 7] = 0.9;

        let mut picker = PeakPicker::new(PeakPickerConfig {
            neighborhood_t: 2,
            neighborhood_f: 2,
            min_magnitude_db: f32::NEG_INFINITY,
            min_magnitude_linear: Some(0.1),
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
        let mut picker = PeakPicker::new(PeakPickerConfig {
            neighborhood_t: 1,
            neighborhood_f: 1,
            min_magnitude_db: f32::NEG_INFINITY,
            min_magnitude_linear: Some(0.1),
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

        let mut picker = PeakPicker::new(PeakPickerConfig {
            neighborhood_t: 1,
            neighborhood_f: 1,
            min_magnitude_db: f32::NEG_INFINITY,
            min_magnitude_linear: Some(0.1),
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

        let mut picker = PeakPicker::new(PeakPickerConfig {
            neighborhood_t: 2,
            neighborhood_f: 2,
            min_magnitude_db: f32::NEG_INFINITY,
            min_magnitude_linear: Some(0.1),
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
    fn adaptive_per_second_breaks_exact_mag_ties_by_position() {
        // All peaks share one bucket and the SAME magnitude, exceeding the
        // cap. The `(t_frame, f_bin)` tiebreak must make selection a total
        // order so the kept set is deterministic — and identical to what
        // the streaming `finalize_bucket` keeps. Without the tiebreak,
        // `sort_unstable` could retain a different subset.
        let peaks: Vec<Peak> = (0..6)
            .map(|f| Peak {
                t_frame: 1,
                f_bin: f,
                _pad: 0,
                mag: 5.0, // exact tie
            })
            .collect();

        let kept = adaptive_per_second(peaks, 100.0, 3);
        let mut got: Vec<u16> = kept.iter().map(|p| p.f_bin).collect();
        got.sort_unstable();
        // Lowest-position peaks win the tie (mag desc, then (t, f) asc).
        assert_eq!(got, vec![0, 1, 2]);
    }

    #[test]
    fn empty_input_returns_empty() {
        let mut picker = PeakPicker::new(PeakPickerConfig::default());
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
        let mut picker = PeakPicker::new(PeakPickerConfig {
            neighborhood_t: 1,
            neighborhood_f: 1,
            min_magnitude_db: f32::NEG_INFINITY,
            min_magnitude_linear: Some(0.1),
            target_per_sec: 0,
        });
        let peaks = picker.pick(&spec, n_frames, n_bins, 100.0);
        assert_eq!(peaks.len(), 9);
    }

    #[test]
    fn boundary_peak_at_corner_is_picked() {
        let mut spec = vec![0.0_f32; 16 * 16];
        spec[0] = 1.0;
        let mut picker = PeakPicker::new(PeakPickerConfig {
            neighborhood_t: 3,
            neighborhood_f: 3,
            min_magnitude_db: f32::NEG_INFINITY,
            min_magnitude_linear: Some(0.1),
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

    #[test]
    fn incremental_matches_rolling_max_2d() {
        let n_rows = 12;
        let n_cols = 10;
        let mut input = vec![0.0_f32; n_rows * n_cols];
        let mut x: u32 = 42;
        for v in input.iter_mut() {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *v = (x % 200) as f32;
        }

        for kt in [1, 2, 3, 5] {
            for kf in [1, 2, 3] {
                let mut reference = vec![0.0_f32; n_rows * n_cols];
                rolling_max_2d(&input, n_rows, n_cols, kt, kf, &mut reference);

                let mut det = IncrementalPeakDetector::new(kt, kf, n_cols);
                let mut out_max = vec![0.0_f32; n_cols];
                let mut got_rows: Vec<(u32, Vec<f32>)> = Vec::new();

                for r in 0..n_rows {
                    let row = &input[r * n_cols..(r + 1) * n_cols];
                    if let Some(ripe_abs) = det.push_row(row, &mut out_max) {
                        got_rows.push((ripe_abs, out_max.clone()));
                    }
                }
                det.flush(&mut out_max, |abs, max_row| {
                    got_rows.push((abs, max_row.to_vec()));
                });

                assert_eq!(
                    got_rows.len(),
                    n_rows,
                    "kt={kt}, kf={kf}: expected {n_rows} output rows, got {}",
                    got_rows.len()
                );

                for (abs, row_max) in &got_rows {
                    let r = *abs as usize;
                    let ref_row = &reference[r * n_cols..(r + 1) * n_cols];
                    assert_eq!(
                        row_max.as_slice(),
                        ref_row,
                        "kt={kt}, kf={kf}, row={r} mismatch"
                    );
                }
            }
        }
    }

    // Direct unit tests for `IncrementalPeakDetector`.
    //
    // These pin the contract of the post-226e0f2 failable reads in
    // `push_row` and `flush`. The reads are guarded by `if let Some`,
    // so a future bug that leaves a read site unwritten would
    // silently leave `out_max` at 0.0 (the pre-zeroed default) instead
    // of panicking. The tests below exercise the invariant directly:
    // the detector's per-row output must equal the brute-force 2-D
    // rolling max for every (kt, kf, n_bins) combination, and the
    // stream of ripe rows must be contiguous (kt, kt+1, …, n_pushed-1)
    // with no duplicates between `push_row` and `flush`.

    #[test]
    fn incremental_returns_none_until_window_fills_then_some() {
        // For kt=2, the first 2 pushes return None (vertical window not
        // full), the 3rd push emits the first ripe row (abs=0), the 4th
        // emits abs=1, and so on.
        let kt = 2;
        let mut det = IncrementalPeakDetector::new(kt, 1, 4);
        let mut out_max = vec![0.0_f32; 4];
        let row = vec![1.0_f32; 4];

        assert!(det.push_row(&row, &mut out_max).is_none());
        assert!(det.push_row(&row, &mut out_max).is_none());
        for expected_ripe in 0u32..5 {
            let ripe = det
                .push_row(&row, &mut out_max)
                .expect("ripe row should be available after kt pushes");
            assert_eq!(ripe, expected_ripe);
        }
    }

    #[test]
    fn incremental_matches_naive_for_random_input() {
        // Random spectrogram, single (kt, kf) config: every per-row
        // output (push and flush) must equal the brute-force 2-D max.
        // This is the focused regression for the 226e0f2 refactor: if
        // the `if let Some(&(_, v))` branch is ever hit on a real push,
        // this test fails because the 0.0 fallback diverges from the
        // reference.
        let n_rows = 10;
        let n_cols = 6;
        let kt = 2;
        let kf = 2;

        let mut input = vec![0.0_f32; n_rows * n_cols];
        let mut x: u32 = 7;
        for v in input.iter_mut() {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *v = (x % 100) as f32;
        }

        let mut reference = vec![0.0_f32; n_rows * n_cols];
        rolling_max_2d(&input, n_rows, n_cols, kt, kf, &mut reference);

        let mut det = IncrementalPeakDetector::new(kt, kf, n_cols);
        let mut out_max = vec![0.0_f32; n_cols];
        let mut got: Vec<(u32, Vec<f32>)> = Vec::new();

        for r in 0..n_rows {
            let row = &input[r * n_cols..(r + 1) * n_cols];
            if let Some(ripe_abs) = det.push_row(row, &mut out_max) {
                got.push((ripe_abs, out_max.clone()));
            }
        }
        det.flush(&mut out_max, |abs, max_row| {
            got.push((abs, max_row.to_vec()));
        });

        // No duplicates between push and flush; contiguous kt..n_rows-1.
        assert_eq!(got.len(), n_rows);
        for (expected, (abs, _)) in got.iter().enumerate() {
            assert_eq!(*abs as usize, expected);
        }
        for (abs, row_max) in &got {
            let r = *abs as usize;
            let ref_row = &reference[r * n_cols..(r + 1) * n_cols];
            assert_eq!(row_max.as_slice(), ref_row, "row {r} mismatch");
        }
    }

    // ── Idempotency / push-after-flush contract ──
    //
    // `flush` must be idempotent and `push_row` after a `flush` must
    // not re-emit rows the flush already drained (the
    // `StreamingFingerprinter::flush` lifecycle contract in `fp.rs`).
    // These tests pin the `last_emitted` cursor: without it, a second
    // flush re-detects the tail rows and a push after flush re-ripens
    // them, duplicating every downstream peak/hash.

    #[test]
    fn flush_is_idempotent() {
        let kt = 2;
        let mut det = IncrementalPeakDetector::new(kt, 1, 4);
        let mut out = vec![0.0_f32; 4];
        let row = vec![1.0_f32; 4];
        for _ in 0..6 {
            let _ = det.push_row(&row, &mut out);
        }
        let mut first: Vec<u32> = Vec::new();
        det.flush(&mut out, |abs, _| first.push(abs));
        assert!(!first.is_empty());
        let mut second: Vec<u32> = Vec::new();
        det.flush(&mut out, |abs, _| second.push(abs));
        assert!(
            second.is_empty(),
            "second flush re-emitted rows {second:?} (first flush emitted {first:?})"
        );
    }

    #[test]
    fn push_after_flush_never_reemits_rows() {
        let kt = 3;
        let n_first = 5usize;
        let n_more = kt + 2;
        let mut det = IncrementalPeakDetector::new(kt, 1, 2);
        let mut out = vec![0.0_f32; 2];
        let row = vec![1.0_f32, 2.0];
        let mut emitted: Vec<u32> = Vec::new();

        for _ in 0..n_first {
            if let Some(r) = det.push_row(&row, &mut out) {
                emitted.push(r);
            }
        }
        det.flush(&mut out, |abs, _| emitted.push(abs));

        // Continue the same stream after the flush.
        for _ in 0..n_more {
            if let Some(r) = det.push_row(&row, &mut out) {
                emitted.push(r);
            }
        }
        det.flush(&mut out, |abs, _| emitted.push(abs));

        let total = n_first + n_more;
        let mut sorted = emitted.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            emitted.len(),
            "rows emitted more than once: {emitted:?}"
        );
        let want: Vec<u32> = (0..total as u32).collect();
        assert_eq!(sorted, want, "full session must cover rows 0..{total} once");
    }

    #[test]
    fn vertical_nan_never_surfaces_in_incremental_path() {
        let kt = 1;
        let kf = 0;
        let mut det = IncrementalPeakDetector::new(kt, kf, 4);
        let mut out = vec![0.0_f32; 4];
        let rows = [
            vec![1.0, 0.0, 0.0, 0.0],
            vec![f32::NAN, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0],
        ];
        for row in rows {
            let _ = det.push_row(&row, &mut out);
        }
        let mut flushed: Vec<[f32; 4]> = Vec::new();
        det.flush(&mut out, |_, max_row| {
            let pinned = [max_row[0], max_row[1], max_row[2], max_row[3]];
            assert!(pinned.iter().all(|v| !v.is_nan()));
            flushed.push(pinned);
        });
        assert_eq!(flushed.len(), 1);
        // Row 2's vertical max: NaN in row 1 never wins over the zeros.
        assert_eq!(flushed[0], [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn vertical_all_nan_row_yields_zeros() {
        let kt = 1;
        let kf = 0;
        let mut det = IncrementalPeakDetector::new(kt, kf, 3);
        let mut out = vec![0.0_f32; 3];
        let rows = [
            vec![2.0, 2.0, 2.0],
            vec![f32::NAN, f32::NAN, f32::NAN],
            vec![1.0, 1.0, 1.0],
        ];
        for row in rows {
            let _ = det.push_row(&row, &mut out);
        }
        let mut flushed: Vec<[f32; 3]> = Vec::new();
        det.flush(&mut out, |_, max_row| {
            flushed.push([max_row[0], max_row[1], max_row[2]]);
        });
        assert_eq!(flushed.len(), 1);
        // All-NaN row 1 never wins; row 2's vertical max is row 2 itself.
        assert_eq!(flushed[0], [1.0, 1.0, 1.0]);
    }

    #[test]
    fn peak_picker_huge_target_per_sec_does_not_panic_on_one_cell() {
        let spec = vec![1.0_f32];
        let mut picker = PeakPicker::new(PeakPickerConfig {
            neighborhood_t: 0,
            neighborhood_f: 0,
            min_magnitude_db: f32::NEG_INFINITY,
            min_magnitude_linear: None,
            target_per_sec: usize::MAX,
        });
        // Finite fps=1 with a one-cell spectrogram: old code reserved
        // `usize::MAX` candidates; the cell-count cap must keep this O(1).
        let peaks = picker.pick(&spec, 1, 1, 1.0);
        assert_eq!(peaks.len(), 1);
        assert_eq!(peaks[0].t_frame, 0);
        assert_eq!(peaks[0].f_bin, 0);
    }

    #[test]
    fn reset_clears_emission_cursor() {
        let kt = 2;
        let mut det = IncrementalPeakDetector::new(kt, 1, 3);
        let mut out = vec![0.0_f32; 3];
        let row = vec![1.0_f32; 3];
        for _ in 0..5 {
            let _ = det.push_row(&row, &mut out);
        }
        let mut n = 0;
        det.flush(&mut out, |_, _| n += 1);
        assert!(n > 0);

        // After reset the same session must emit again (fresh stream).
        det.reset();
        let mut re_emitted = 0usize;
        for _ in 0..5 {
            if det.push_row(&row, &mut out).is_some() {
                re_emitted += 1;
            }
        }
        det.flush(&mut out, |_, _| re_emitted += 1);
        assert_eq!(re_emitted, 5, "reset must restart the emission cursor");
    }

    /// xorshift32 in `[lo, lo + span)`; every 97th value is NaN.
    fn noisy_values(seed: u32, n: usize, lo: f32, span: u32) -> Vec<f32> {
        let mut x = seed.max(1);
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                if x.is_multiple_of(97) {
                    f32::NAN
                } else {
                    lo + (x % span) as f32
                }
            })
            .collect()
    }

    /// The pre-streaming `pick`: full 2-D max via `rolling_max_2d`, then a
    /// scalar candidate scan. Kept as the oracle for the streaming picker.
    fn legacy_pick(
        cfg: &PeakPickerConfig,
        spec: &[f32],
        nf: usize,
        nb: usize,
        fps: f32,
    ) -> Vec<Peak> {
        let mut max = vec![0.0_f32; spec.len()];
        rolling_max_2d(
            spec,
            nf,
            nb,
            cfg.neighborhood_t,
            cfg.neighborhood_f,
            &mut max,
        );
        let mut out = Vec::new();
        for t in 0..nf {
            for f in 0..nb {
                let v = spec[t * nb + f];
                let above =
                    v > cfg.min_magnitude_db && cfg.min_magnitude_linear.is_none_or(|l| v > l);
                if above && v >= max[t * nb + f] {
                    out.push(Peak {
                        t_frame: t as u32,
                        f_bin: f as u16,
                        _pad: 0,
                        mag: v,
                    });
                }
            }
        }
        if cfg.target_per_sec > 0 && fps > 0.0 && !out.is_empty() {
            out = adaptive_per_second(out, fps, cfg.target_per_sec);
        }
        out.sort_unstable_by_key(|p| (p.t_frame, p.f_bin));
        out
    }

    #[test]
    fn hmax_row_matches_rolling_max_1d_bit_for_bit() {
        let mut dq = VecDeque::new();
        for n in [1usize, 2, 7, 8, 9, 30, 31, 32, 63, 64, 65, 513] {
            let row = noisy_values(n as u32 * 31 + 5, n, -60.0, 40);
            for k in [0usize, 1, 2, 3, 4, 7, 8, 15, 16, 40] {
                let mut want = vec![0.0; n];
                rolling_max_1d(&row, k, &mut want, &mut dq);
                let mut pad = vec![0.0; n + 2 * k];
                let mut got = vec![0.0; n];
                hmax_row(&row, k, &mut pad, &mut got);
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&got), bits(&want), "n={n} k={k}");
            }
        }
    }

    #[test]
    fn streaming_pick_matches_full_spectrogram_oracle() {
        let mut x: u32 = 0x5EED;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        // One picker reused across shapes also pins scratch reuse.
        for case in 0..400 {
            let nf = (next() % 70 + 1) as usize;
            let nb = (next() % 60 + 1) as usize;
            let cfg = PeakPickerConfig {
                neighborhood_t: (next() % 18) as usize,
                neighborhood_f: (next() % 18) as usize,
                min_magnitude_db: -55.0,
                min_magnitude_linear: if case % 4 == 0 { Some(-45.0) } else { None },
                target_per_sec: [0, 2, 30][case % 3],
            };
            let spec = noisy_values(next(), nf * nb, -60.0, 30);
            let mut picker = PeakPicker::new(cfg.clone());
            for _ in 0..2 {
                let got = picker.pick(&spec, nf, nb, 10.0);
                assert_eq!(
                    got,
                    legacy_pick(&cfg, &spec, nf, nb, 10.0),
                    "case {case}: {nf}x{nb} {cfg:?}"
                );
            }
        }
    }

    /// Randomized sessions: pushes interleaved with flushes (push-after-
    /// flush continues the stream) and resets, NaN cells included. Every
    /// emitted row must equal the brute-force window max over the rows
    /// pushed so far, with the window right-clamped at the push cursor.
    #[test]
    fn incremental_random_sessions_match_brute_force() {
        let mut x: u32 = 0xF1A5;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        for case in 0..150 {
            let kt = (next() % 9) as usize;
            let kf = (next() % 9) as usize;
            let nb = (next() % 40 + 1) as usize;
            let mut det = IncrementalPeakDetector::new(kt, kf, nb);
            let mut out = vec![0.0_f32; nb];
            let mut rows: Vec<Vec<f32>> = Vec::new();
            let mut emitted: Vec<u32> = Vec::new();
            let check = |rows: &[Vec<f32>], r: u32, got: &[f32]| {
                let r = r as usize;
                let hi = (r + kt).min(rows.len() - 1);
                for f in 0..nb {
                    let mut want = f32::NAN;
                    for row in &rows[r.saturating_sub(kt)..=hi] {
                        for &v in &row[f.saturating_sub(kf)..=(f + kf).min(nb - 1)] {
                            want = want.max(v);
                        }
                    }
                    // `==` treats ±0 as equal; NaN only for an all-NaN window.
                    assert!(
                        got[f] == want || (got[f].is_nan() && want.is_nan()),
                        "case {case} kt={kt} kf={kf} row {r} bin {f}: got {} want {want}",
                        got[f]
                    );
                }
            };
            for _ in 0..(next() % 6 + 1) {
                for _ in 0..(next() % 40) {
                    let row = noisy_values(next(), nb, -10.0, 20);
                    rows.push(row);
                    if let Some(r) = det.push_row(rows.last().unwrap(), &mut out) {
                        check(&rows, r, &out);
                        emitted.push(r);
                    }
                }
                match next() % 3 {
                    0 => {}
                    1 => det.flush(&mut out, |r, m| {
                        check(&rows, r, m);
                        emitted.push(r);
                    }),
                    _ => {
                        det.flush(&mut out, |r, m| {
                            check(&rows, r, m);
                            emitted.push(r);
                        });
                        let want: Vec<u32> = (0..rows.len() as u32).collect();
                        assert_eq!(
                            emitted, want,
                            "case {case}: every row exactly once, in order"
                        );
                        det.reset();
                        rows.clear();
                        emitted.clear();
                    }
                }
            }
        }
    }
}
