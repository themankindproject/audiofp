//! Shared streaming-extraction engine for the classical fingerprinters.
//!
//! `StreamingWang` and `StreamingPanako` run the same pipeline — STFT
//! front-end, rolling log-power spectrogram, incremental peak detection,
//! per-second adaptive bucket finalisation, deferred anchor emission —
//! and differ only in three knobs:
//!
//! - the target-zone time bound (Wang inclusive `dt ≤ zone_t`, Panako
//!   strict `dt < zone_t`),
//! - the per-anchor target cap (Wang keeps top-K, Panako keeps a
//!   2·fan_out soft cap with weakest-magnitude eviction),
//! - the hash-emission closure (Wang linear top-K pairs, Panako
//!   pairwise (b, c) heap).
//!
//! This module owns the shared skeleton once, so the two extractors
//! become thin wrappers that configure the knobs and supply their emit
//! closure. Output parity is pinned by the existing
//! `streaming_offline_*` tests in each file.
//!
//! # Continuous-stream frame limit
//!
//! Frame indices are `u32` throughout — `Peak::t_frame`,
//! `WangHash::t_anchor`, and `PanakoHash::t_anchor`/`t_b`/`t_c` are all
//! `u32` because they are part of the `bytemuck::Pod` wire layout (8 bytes
//! for Wang, 16 for Panako). Widening them would grow the persisted hash
//! format and force a serialization version bump, which 0.4.0 explicitly
//! declined.
//!
//! The practical consequence is a bound on how long a *single* stream may
//! run without [`StreamCore::reset`]:
//!
//! | Algorithm     | Frame rate   | `u32` frame budget      |
//! |---------------|--------------|-------------------------|
//! | Wang / Panako | 62.5 fps     | ~795 days of audio      |
//! | Haitsma       | 78.125 fps   | ~636 days of audio      |
//!
//! Past that point the frame counter wraps: debug builds panic on the
//! overflowing `+= 1`, release builds silently emit hashes with wrapped
//! `t_anchor` values. Long-lived capture processes (24/7 broadcast
//! monitoring) should call `reset()` on a segment boundary — e.g. hourly
//! or daily — which is normal practice anyway, since matching operates on
//! bounded windows rather than an unbounded stream. Re-creating the
//! streamer, or calling `reset()`, returns the counters to zero.

use alloc::vec;
use alloc::vec::Vec;

use crate::TimestampMs;
use crate::dsp::peaks::{IncrementalPeakDetector, Peak};
use crate::dsp::power_to_db_wide;
use crate::dsp::stft::{ShortTimeFFT, StftConfig};
use crate::dsp::windows::WindowKind;
use crate::pcm;

/// A target-zone comparison policy.
#[derive(Clone, Copy)]
pub(crate) enum Zone {
    /// Wang: `dt ≤ target_zone_t` and `|df| ≤ target_zone_f`.
    Inclusive,
    /// Panako: `dt < target_zone_t` and `|df| < target_zone_f`.
    Strict,
}

/// One anchor awaiting finalisation, with its accumulated targets.
pub(crate) struct PendingAnchor {
    pub(crate) peak: Peak,
    pub(crate) targets: Vec<Peak>,
}

/// Peak/zone configuration — the parts of `WangConfig` / `PanakoConfig`
/// the shared pipeline needs.
#[derive(Clone, Copy)]
pub(crate) struct PeakCfg {
    pub(crate) min_anchor_mag_db: f32,
    pub(crate) target_zone_t: i32,
    pub(crate) target_zone_f: i32,
    pub(crate) fan_out: usize,
    pub(crate) peaks_per_sec: usize,
    pub(crate) max_pending_anchors: Option<usize>,
    pub(crate) max_push_samples: Option<usize>,
}

/// Shared streaming pipeline state. The two classical extractors embed
/// this and delegate `push`/`flush` to it, supplying only the
/// per-algorithm emission closure.
pub(crate) struct StreamCore<F> {
    pub(crate) stft: ShortTimeFFT,
    pub(crate) sample_carry: Vec<f32>,

    // Rolling log-power spectrogram window (ring buffer, row-major).
    //
    // audit C2: `append_frame_scratch_row` used to memmove the whole
    // window down one row at every frame past capacity (~4 MB/s of
    // `copy_within` at 62.5 fps for Wang's 31×513×4B window). The buffer
    // is now a ring of rows: the new row overwrites the oldest, and rows
    // are addressed logically (`spec_tail + row` mod capacity) so there
    // is no data movement at all. Each row stays contiguous in memory,
    // so the bin loops are unchanged apart from the base offset.
    pub(crate) spec: Vec<f32>,
    pub(crate) spec_n_rows: usize,
    pub(crate) spec_n_bins: usize,
    /// Ring index of the logical row 0 (the oldest row in `spec`).
    pub(crate) spec_tail: usize,
    pub(crate) spec_first_frame: u32,

    pub(crate) n_frames_total: u32,
    pub(crate) last_pd_frame: i32,

    pub(crate) peak_det: IncrementalPeakDetector,
    pub(crate) peak_row_max: Vec<f32>,
    pub(crate) frame_scratch: Vec<f32>,

    // Per-second adaptive thresholding. Sorted Vec — bounded (≤ 3 entries
    // in steady state), so linear/binary search is faster than a tree.
    pub(crate) bucket_pending: Vec<(u32, Vec<Peak>)>,
    pub(crate) last_finalized_bucket: i32,

    pub(crate) pending_anchors: alloc::collections::VecDeque<PendingAnchor>,

    /// Pooled scratch for `finalize_buckets` / `flush`.
    pub(crate) to_finalize: Vec<u32>,

    /// Pooled output buffer for the emit path.
    pub(crate) emitted: Vec<(TimestampMs, F)>,

    // Per-algorithm constants.
    pub(crate) n_fft: usize,
    pub(crate) hop: usize,
    pub(crate) frames_per_sec: f32,
    pub(crate) neighborhood: usize,
    pub(crate) log_floor_power: f32,
    pub(crate) zone: Zone,
    /// Free-list of per-anchor target buffers. `finalize_bucket` pops (or
    /// allocates on first use); `emit_anchor` implementations return the
    /// consumed anchor's buffer here. This closes the last steady-state
    /// allocation in the streaming pipeline: without it every anchor pays
    /// one `Vec::new()` + growth. Pool size is naturally bounded (one entry
    /// per anchor that ever lived concurrently — the pending-anchor
    /// high-water mark); capacities are bounded by construction
    /// (`insert_top_target` caps at `fan_out`, Panako at `2·fan_out`).
    pub(crate) targets_pool: Vec<Vec<Peak>>,
    /// Free-list of per-bucket peak buffers. `push_peak` pops (or allocates
    /// on first use) when a new second-bucket starts; `finalize_bucket`
    /// returns the consumed buffer after extracting its peaks. Same
    /// steady-state rationale as [`targets_pool`](Self::targets_pool), but
    /// demand-grown (no prefill): bucket concurrency is tiny (≤ a handful
    /// alive at once — one per unfinalized second), so warmup stabilizes it
    /// immediately, unlike the bursty anchor path.
    pub(crate) bucket_pool: Vec<Vec<Peak>>,
    /// Test-only pool diagnostics: pops served from the pool vs fresh
    /// allocations (steady-state zero-alloc means ~all hits after warmup).
    #[cfg(test)]
    pub(crate) pool_hits: u64,
    /// Test-only pool diagnostics (see [`pool_hits`](Self::pool_hits)).
    #[cfg(test)]
    pub(crate) pool_misses: u64,
}

/// Push `peak` into the per-second bucket map, creating the bucket entry
/// if absent. Shared by the steady-state peak scan and the flush drain
/// so the bucket-insertion order stays identical on both paths.
///
/// New buckets pop their peak buffer from `bucket_pool` (or allocate on
/// first use) instead of `vec![peak]` — the steady-state zero-alloc half
/// of the bucket lifecycle (`finalize_bucket` returns the buffer after
/// extracting its peaks).
fn push_peak(
    bucket_pending: &mut Vec<(u32, Vec<Peak>)>,
    bucket_pool: &mut Vec<Vec<Peak>>,
    bucket: u32,
    peak: Peak,
) {
    match bucket_pending.binary_search_by_key(&bucket, |e| e.0) {
        Ok(idx) => bucket_pending[idx].1.push(peak),
        Err(idx) => {
            let mut peaks = bucket_pool.pop().unwrap_or_default();
            peaks.push(peak);
            bucket_pending.insert(idx, (bucket, peaks));
        }
    }
}

/// Maximum retained entries for [`StreamCore::targets_pool`](StreamCore::targets_pool).
///
/// Flush emits every pending anchor while creating few new ones, so the
/// pool would balloon to the whole backlog on every stream end and reallocate
/// its own backing Vec whenever a flush exceeds the previous high-water.
/// Capping keeps pool memory bounded (~`MAX × 300 B` worst case) and pool
/// pushes allocation-free in steady state; excess buffers are simply dropped
/// (their memory is freed — the rare flush-time case they cover is not hot).
pub(crate) const TARGETS_POOL_MAX: usize = 256;

/// Maximum retained entries for the bucket free-list (see
/// [`TARGETS_POOL_MAX`]). Concurrent unfinalized buckets fit in a handful;
/// anything beyond that is flush-time excess.
pub(crate) const BUCKET_POOL_MAX: usize = 32;

/// Prefill size for [`StreamCore::targets_pool`](StreamCore::targets_pool).
///
/// Rationale: in steady state most pooled buffers are checked out inside
/// live pending anchors, so the free count oscillates near zero; bucket
/// finalization creates anchors in bursts (one bucket's peaks at once) that
/// would outrun a demand-grown pool and allocate fresh buffers every burst.
/// Pre-seeding covers ~2× the observed concurrent-anchor high-water mark
/// (~36 on representative audio) with slack, so bursts pop pre-reserved
/// buffers instead of allocating. Pathological peak density far beyond
/// warmup input can still outgrow the pool — the same caveat as every other
/// amortised buffer in the pipeline (`emitted`, `pending_anchors`, carry).
pub(crate) const TARGETS_POOL_PREFILL: usize = 64;

impl<F> StreamCore<F> {
    /// - `pool_buf_cap`: capacity reserved per pre-seeded target buffer —
    ///   must equal the pop-time `reserve` in `finalize_bucket`
    ///   (`2·fan_out + 2`, covering both algorithms' transient overshoot)
    ///   so pre-seeded AND recycled buffers never regrow.
    /// - `pool_prefill`: number of target buffers to pre-seed (see
    ///   [`TARGETS_POOL_PREFILL`]).
    // 8 construction scalars (all used, no meaningful subgrouping that
    // wouldn't just move the arity into a config struct for 3 call sites).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        n_fft: usize,
        hop: usize,
        sample_rate: u32,
        neighborhood: usize,
        log_floor_power: f32,
        zone: Zone,
        pool_buf_cap: usize,
        pool_prefill: usize,
    ) -> Self {
        let stft = ShortTimeFFT::new(StftConfig {
            n_fft,
            hop,
            window: WindowKind::Hann,
            center: false,
        });
        let n_bins = stft.n_bins();
        let window_capacity = 2 * neighborhood + 1;
        Self {
            stft,
            sample_carry: Vec::new(),
            spec: vec![0.0_f32; window_capacity * n_bins],
            spec_n_rows: 0,
            spec_n_bins: n_bins,
            spec_tail: 0,
            spec_first_frame: 0,
            n_frames_total: 0,
            last_pd_frame: -1,
            peak_det: IncrementalPeakDetector::new(neighborhood, neighborhood, n_bins),
            peak_row_max: vec![0.0_f32; n_bins],
            frame_scratch: vec![0.0_f32; n_bins],
            bucket_pending: Vec::new(),
            last_finalized_bucket: -1,
            pending_anchors: alloc::collections::VecDeque::new(),
            to_finalize: Vec::new(),
            emitted: Vec::new(),
            targets_pool: (0..pool_prefill)
                .map(|_| Vec::with_capacity(pool_buf_cap))
                .collect(),
            bucket_pool: Vec::new(),
            #[cfg(test)]
            pool_hits: 0,
            #[cfg(test)]
            pool_misses: 0,
            n_fft,
            hop,
            frames_per_sec: sample_rate as f32 / hop as f32,
            neighborhood,
            log_floor_power,
            zone,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.sample_carry.clear();
        self.peak_det.reset();
        self.spec_n_rows = 0;
        self.spec_tail = 0;
        self.spec_first_frame = 0;
        self.n_frames_total = 0;
        self.last_pd_frame = -1;
        self.bucket_pending.clear();
        self.last_finalized_bucket = -1;
        self.pending_anchors.clear();
        self.to_finalize.clear();
        self.emitted.clear();
    }

    /// Capacity of the spec ring buffer, in rows.
    #[inline]
    fn spec_capacity(&self) -> usize {
        2 * self.neighborhood + 1
    }

    /// Base offset of logical row `row` (0 = oldest) inside `spec`.
    ///
    /// Rows live at contiguous `n_bins`-long slots addressed by
    /// `(spec_tail + row) % capacity`; no data movement ever occurs,
    /// even when the ring wraps (audit C2).
    #[inline]
    fn spec_row_base(&self, row: usize) -> usize {
        (self.spec_tail + row) % self.spec_capacity() * self.spec_n_bins
    }

    /// Append `self.frame_scratch` to the rolling spec ring, overwriting
    /// the oldest row at capacity. Avoids both the per-frame `Vec::clone`
    /// and the per-frame full-window memmove (audit C2).
    fn append_frame_scratch_row(&mut self) {
        let n_bins = self.spec_n_bins;
        debug_assert_eq!(self.frame_scratch.len(), n_bins);
        let cap = self.spec_capacity();
        if self.spec_n_rows == cap {
            // Drop the oldest row by advancing the ring — O(1), no
            // `copy_within` of the (cap-1) older rows.
            self.spec_tail = (self.spec_tail + 1) % cap;
            self.spec_first_frame += 1;
        } else {
            self.spec_n_rows += 1;
        }
        let dst_start = self.spec_row_base(self.spec_n_rows - 1);
        // Disjoint borrow: `self.spec` (mut) and `self.frame_scratch`
        // (shared) are different fields of `self`, so this is sound.
        self.spec[dst_start..dst_start + n_bins].copy_from_slice(&self.frame_scratch);
    }

    /// Peak-detect rows `[from_row, to_row]` (spec-relative) and push
    /// survivors into `bucket_pending`.
    fn detect_rows_range(&mut self, cfg: PeakCfg, from_row: usize, to_row: usize) {
        if self.spec_n_rows == 0 || from_row > to_row {
            return;
        }
        let n_bins = self.spec_n_bins;
        let spec_tail = self.spec_tail;
        let spec_cap = self.spec_capacity();
        for row in from_row..=to_row {
            if row >= self.spec_n_rows {
                break;
            }
            let abs_f = self.spec_first_frame + row as u32;
            let bucket = (abs_f as f32 / self.frames_per_sec) as u32;
            // Ring addressing: logical row → slot offset (audit C2).
            let row_start = (spec_tail + row) % spec_cap * n_bins;
            let spec_row = &self.spec[row_start..row_start + n_bins];
            let peak_max = &self.peak_row_max[..n_bins];
            for bin in 0..n_bins {
                let v = spec_row[bin];
                if v > cfg.min_anchor_mag_db && v >= peak_max[bin] {
                    let peak = Peak {
                        t_frame: abs_f,
                        f_bin: bin as u16,
                        _pad: 0,
                        mag: v,
                    };
                    // Disjoint field borrows (`bucket_pending` vs
                    // `bucket_pool`) — no conflict.
                    push_peak(
                        &mut self.bucket_pending,
                        &mut self.bucket_pool,
                        bucket,
                        peak,
                    );
                }
            }
        }
    }

    /// Finalise one bucket: apply per-second adaptive threshold (top
    /// `peaks_per_sec` by magnitude), then for each surviving peak in
    /// `(t, f)` order, grow target lists of older anchors and register
    /// the peak as a new anchor.
    fn finalize_bucket(
        &mut self,
        cfg: PeakCfg,
        bucket: u32,
        mut add_target: impl FnMut(&mut Vec<Peak>, Peak, i32, i32, PeakCfg),
    ) {
        let mut peaks = match self.bucket_pending.binary_search_by_key(&bucket, |e| e.0) {
            Ok(idx) => self.bucket_pending.remove(idx).1,
            Err(_) => return,
        };
        // Sort by mag desc, then `(t, f)` ascending. The positional
        // tiebreak is unique per peak, so equal-magnitude peaks at the
        // truncation boundary resolve identically to the offline
        // `adaptive_per_second`.
        peaks.sort_unstable_by(|a, b| {
            b.mag
                .partial_cmp(&a.mag)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then_with(|| (a.t_frame, a.f_bin).cmp(&(b.t_frame, b.f_bin)))
        });
        peaks.truncate(cfg.peaks_per_sec);
        // Re-sort by `(t, f)` so downstream iteration matches the offline
        // hash builder's order.
        peaks.sort_unstable_by_key(|p| (p.t_frame, p.f_bin));

        let target_zone_t = cfg.target_zone_t;
        let target_zone_f = cfg.target_zone_f;

        // `drain` (not by-value consume) so the buffer survives for the
        // bucket pool below. `Peak` is `Copy`, so yielded items behave
        // identically to the old `for peak in peaks`.
        for peak in peaks.drain(..) {
            // Add as TARGET to older anchors whose zone covers it.
            for anchor in self.pending_anchors.iter_mut() {
                let dt = peak.t_frame as i32 - anchor.peak.t_frame as i32;
                let in_time = match self.zone {
                    Zone::Inclusive => dt >= 1 && dt <= target_zone_t,
                    Zone::Strict => dt >= 1 && dt < target_zone_t,
                };
                if !in_time {
                    continue;
                }
                let df = peak.f_bin as i32 - anchor.peak.f_bin as i32;
                let in_freq = match self.zone {
                    Zone::Inclusive => df.abs() <= target_zone_f,
                    Zone::Strict => df.abs() < target_zone_f,
                };
                if !in_freq {
                    continue;
                }
                add_target(&mut anchor.targets, peak, dt, df, cfg);
            }
            // Register this peak as a new ANCHOR, reusing a pooled target
            // buffer when one is available (steady-state zero-alloc).
            // If a hard cap is configured, evict oldest anchors first
            // so memory stays bounded under adversarial / dense input —
            // evicted buffers go back to the pool (no allocation even
            // under pressure).
            if let Some(limit) = cfg.max_pending_anchors {
                while self.pending_anchors.len() >= limit {
                    if let Some(evicted) = self.pending_anchors.pop_front() {
                        // Capped (see TARGETS_POOL_MAX): excess is dropped.
                        if self.targets_pool.len() < TARGETS_POOL_MAX {
                            self.targets_pool.push(evicted.targets);
                        }
                    }
                }
            }
            #[cfg(test)]
            let pool_len_before = self.targets_pool.len();
            let mut targets = self.targets_pool.pop().unwrap_or_default();
            // Pooled buffers arrive with the previous anchor's targets still
            // in them — `clear` keeps the capacity and drops the stale
            // contents (reusing them would corrupt hashes AND can underflow
            // `dt` arithmetic downstream).
            targets.clear();
            // Guarantee growth-free accumulation: `insert_top_target`
            // (Wang) does `Vec::insert`, which reallocs when `len == cap`
            // even though logical length never exceeds `fan_out` (the
            // transient insert needs one spare slot); Panako's `add_target`
            // pushes up to `2·fan_out + 1` transiently. Reserving
            // `2·fan_out + 2` covers both with negligible waste, and
            // `reserve` is a no-op for buffers that already converged.
            targets.reserve(2 * cfg.fan_out + 2);
            #[cfg(test)]
            {
                // The pop hit iff the pool was non-empty (a hit allocates
                // nothing; a miss allocates one fresh buffer below... i.e.
                // the `unwrap_or_default` itself).
                if pool_len_before > 0 {
                    self.pool_hits += 1;
                } else {
                    self.pool_misses += 1;
                }
            }
            self.pending_anchors
                .push_back(PendingAnchor { peak, targets });
        }
        // Return the drained peaks buffer to the bucket pool (empty, capacity
        // retained) — the steady-state zero-alloc half of the bucket
        // lifecycle (`push_peak` pops from this pool for new buckets).
        // Capped (see BUCKET_POOL_MAX): excess is dropped.
        if self.bucket_pool.len() < BUCKET_POOL_MAX {
            self.bucket_pool.push(peaks);
        }
        self.last_finalized_bucket = bucket as i32;
    }

    /// Finalise every bucket whose ALL frames have been peak-detected.
    /// Conservative: bucket B is finalisable iff `bucket(last_pd_frame) > B`.
    fn finalize_buckets(
        &mut self,
        cfg: PeakCfg,
        add_target: &mut impl FnMut(&mut Vec<Peak>, Peak, i32, i32, PeakCfg),
    ) {
        if self.last_pd_frame < 0 {
            return;
        }
        let current_bucket = (self.last_pd_frame as f32 / self.frames_per_sec) as i32;
        // Collect into the pooled buffer instead of allocating a fresh
        // `Vec` on every `push`. `bucket_pending` is bounded (≤ 3 in
        // steady state), so the buffer never grows after warmup.
        //
        // The index-based loop (rather than `drain(..)`) sidesteps the
        // borrow conflict: `drain` would hold `&mut self.to_finalize`
        // across the loop body where `self.finalize_bucket` needs
        // `&mut self`. Indexing a `Copy` element produces a `u32` by
        // value, so the immutable borrow of `to_finalize` ends before
        // the mutable call begins.
        self.to_finalize.clear();
        self.to_finalize.extend(
            self.bucket_pending.iter().map(|e| e.0).filter(|&b| {
                (b as i32) > self.last_finalized_bucket && (b as i32) < current_bucket
            }),
        );
        let n = self.to_finalize.len();
        for i in 0..n {
            let bucket = self.to_finalize[i];
            self.finalize_bucket(cfg, bucket, &mut *add_target);
        }
        self.to_finalize.clear();
    }

    /// Pop anchors whose target zone is fully observed, build hashes from
    /// their accumulated targets via `emit_anchor`, and push into
    /// `self.emitted`.
    ///
    /// `emit_anchor` returns the consumed anchor's target buffer, which is
    /// recycled into [`targets_pool`](Self::targets_pool) — the steady-state
    /// zero-allocation half of the anchor lifecycle (`finalize_bucket`
    /// pops from the pool when creating anchors).
    pub(crate) fn emit_finalized_anchors(
        &mut self,
        cfg: PeakCfg,
        emit_anchor: impl FnMut(PendingAnchor, PeakCfg, &mut Vec<(TimestampMs, F)>) -> Vec<Peak>,
    ) {
        // Pop-and-push pattern: take the front anchor, decide whether its
        // target zone is fully observed, and if not put it back. This avoids
        // an `unwrap` after a separate `front()` peek and stays a clean
        // `while let` over the pop result.
        //
        // Temporarily take `emitted` to split the borrow: the loop body
        // needs `&self` (for `emit_anchor`) and `&mut emitted`.
        let mut emitted = core::mem::take(&mut self.emitted);
        let mut emit_anchor = emit_anchor;
        while let Some(anchor) = self.pending_anchors.pop_front() {
            let last_dt = match self.zone {
                // Wang's target zone is inclusive (`dt ≤ target_zone_t`),
                // so the last possible target frame is exactly `t + zone_t`.
                Zone::Inclusive => cfg.target_zone_t as u32,
                // Panako uses strict `dt < target_zone_t`.
                Zone::Strict => cfg.target_zone_t as u32 - 1,
            };
            let last_target_frame = anchor.peak.t_frame + last_dt;
            let last_target_bucket = (last_target_frame as f32 / self.frames_per_sec) as i32;
            if self.last_finalized_bucket < last_target_bucket {
                self.pending_anchors.push_front(anchor);
                break;
            }
            let freed = emit_anchor(anchor, cfg, &mut emitted);
            // Recycle the consumed anchor's target buffer. `mut` access to
            // the pool while `emitted` is taken is disjoint — no borrow
            // conflict. Capped (see TARGETS_POOL_MAX): excess is dropped.
            if self.targets_pool.len() < TARGETS_POOL_MAX {
                self.targets_pool.push(freed);
            }
        }
        self.emitted = emitted;
    }

    /// Common processing for `push` and `push_with`: advance the STFT,
    /// detect peaks, finalise buckets, emit ready anchors into
    /// `self.emitted`.
    pub(crate) fn process_push_samples(
        &mut self,
        samples: &[f32],
        cfg: PeakCfg,
        mut add_target: impl FnMut(&mut Vec<Peak>, Peak, i32, i32, PeakCfg),
        emit_anchor: impl FnMut(PendingAnchor, PeakCfg, &mut Vec<(TimestampMs, F)>) -> Vec<Peak>,
    ) {
        let samples = pcm::truncate_push(samples, cfg.max_push_samples);
        pcm::extend_sanitized(&mut self.sample_carry, samples);

        let n_fft = self.n_fft;
        let hop = self.hop;
        let mut off = 0usize;
        while self.sample_carry.len() - off >= n_fft {
            self.stft
                .process_frame_power(
                    &self.sample_carry[off..off + n_fft],
                    &mut self.frame_scratch,
                )
                .expect("frame_scratch is sized n_bins and frames are exactly n_fft");
            power_to_db_wide(&mut self.frame_scratch, self.log_floor_power);
            self.append_frame_scratch_row();

            self.n_frames_total += 1;
            off += hop;

            if let Some(ripe_abs) = self
                .peak_det
                .push_row(&self.frame_scratch, &mut self.peak_row_max)
            {
                let row_idx = (ripe_abs - self.spec_first_frame) as usize;
                self.detect_rows_range(cfg, row_idx, row_idx);
                self.last_pd_frame = ripe_abs as i32;
            }
        }

        if off > 0 {
            self.sample_carry.drain(0..off);
        }

        self.finalize_buckets(cfg, &mut add_target);
        self.emit_finalized_anchors(cfg, emit_anchor);
    }

    /// Common processing for `flush` and `flush_with`: drain remaining
    /// peaks from the incremental detector, finalise all buckets, emit
    /// all anchors into `self.emitted`.
    pub(crate) fn process_flush(
        &mut self,
        cfg: PeakCfg,
        mut add_target: impl FnMut(&mut Vec<Peak>, Peak, i32, i32, PeakCfg),
        mut emit_anchor: impl FnMut(PendingAnchor, PeakCfg, &mut Vec<(TimestampMs, F)>) -> Vec<Peak>,
    ) {
        let n_bins = self.spec_n_bins;
        let spec = &self.spec;
        let spec_tail = self.spec_tail;
        let spec_cap = 2 * self.neighborhood + 1;
        let spec_first_frame = self.spec_first_frame;
        let bucket_pending = &mut self.bucket_pending;
        let bucket_pool = &mut self.bucket_pool;
        let last_pd = &mut self.last_pd_frame;
        let min_mag = cfg.min_anchor_mag_db;
        let frames_per_sec = self.frames_per_sec;

        self.peak_det
            .flush(&mut self.peak_row_max, |ripe_abs, max_row| {
                let row_idx = (ripe_abs - spec_first_frame) as usize;
                let bucket = (ripe_abs as f32 / frames_per_sec) as u32;
                for (bin, &row_max) in max_row.iter().enumerate() {
                    let idx = (spec_tail + row_idx) % spec_cap * n_bins + bin;
                    let v = spec[idx];
                    if v > min_mag && v >= row_max {
                        let peak = Peak {
                            t_frame: ripe_abs,
                            f_bin: bin as u16,
                            _pad: 0,
                            mag: v,
                        };
                        push_peak(bucket_pending, bucket_pool, bucket, peak);
                    }
                }
                *last_pd = ripe_abs as i32;
            });

        self.to_finalize.clear();
        self.to_finalize
            .extend(self.bucket_pending.iter().map(|e| e.0));
        let n = self.to_finalize.len();
        // Finalize + emit INTERLEAVED per bucket (not finalize-all then
        // emit-all): each bucket's anchors are emitted — returning their
        // target buffers to the pool — before the next bucket allocates new
        // ones, so the flush reuses the same pooled working set as steady
        // pushes instead of bursting fresh allocations. Output-identical to
        // emit-all-at-end: `pending_anchors` is FIFO, per-bucket finalize
        // appends newer anchors at the back, and each emit-all drains in
        // order — the global emission sequence is unchanged. (Also bounds
        // `pending_anchors` length to one bucket's burst rather than the
        // whole remainder, avoiding a flush-time VecDeque doubling.)
        //
        // Flush drains *all* pending anchors unconditionally — the zone
        // check is a liveness optimization for steady-state `push`, but at
        // end-of-stream every anchor's full lookahead has been observed, so
        // the remaining ones must emit (and the queue must empty).
        let mut emitted = core::mem::take(&mut self.emitted);
        for i in 0..n {
            let bucket = self.to_finalize[i];
            self.finalize_bucket(cfg, bucket, &mut add_target);
            while let Some(anchor) = self.pending_anchors.pop_front() {
                let freed = emit_anchor(anchor, cfg, &mut emitted);
                // Capped (see TARGETS_POOL_MAX): excess is dropped.
                if self.targets_pool.len() < TARGETS_POOL_MAX {
                    self.targets_pool.push(freed);
                }
            }
        }
        self.to_finalize.clear();
        self.emitted = emitted;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_ring_overwrites_oldest_row_at_capacity() {
        let mut core = StreamCore::<u32>::new(256, 64, 8_000, 2, -80.0, Zone::Inclusive, 8, 0);
        assert_eq!(core.spec_capacity(), 5);
        let n_bins = core.spec_n_bins;
        core.frame_scratch = alloc::vec![0.0f32; n_bins];

        // Push 12 rows into a 5-row ring: rows 0..7 must be gone,
        // rows 8..12 must be readable as logical rows 0..4.
        for f in 0..12u32 {
            core.frame_scratch.fill(f as f32 / 10.0);
            core.append_frame_scratch_row();
        }

        assert_eq!(core.spec_n_rows, 5, "ring stays at capacity");
        assert_eq!(core.spec_first_frame, 7, "seven oldest frames dropped");
        for row in 0..5usize {
            let base = core.spec_row_base(row);
            let expect = (7 + row) as f32 / 10.0;
            assert!(
                core.spec[base..base + n_bins].iter().all(|&v| v == expect),
                "logical row {row} must hold frame data {expect}"
            );
        }
    }

    #[test]
    fn spec_ring_reset_clears_tail() {
        let mut core = StreamCore::<u32>::new(256, 64, 8_000, 2, -80.0, Zone::Inclusive, 8, 0);
        core.frame_scratch = alloc::vec![1.0f32; core.spec_n_bins];
        for _ in 0..12 {
            core.append_frame_scratch_row();
        }
        assert_eq!(core.spec_first_frame, 7);
        core.reset();
        assert_eq!(core.spec_first_frame, 0);
        assert_eq!(core.spec_n_rows, 0);
        assert_eq!(core.spec_tail, 0, "ring tail must restart at slot 0");

        // And appending after reset still lands in slot 0.
        core.frame_scratch.fill(2.0);
        core.append_frame_scratch_row();
        assert_eq!(core.spec_row_base(0), 0);
        assert!(
            core.spec[..core.spec_n_bins].iter().all(|&v| v == 2.0),
            "post-reset append must write the first slot"
        );
    }
}
