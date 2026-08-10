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

    // Rolling log-power spectrogram window (contiguous, row-major).
    pub(crate) spec: Vec<f32>,
    pub(crate) spec_n_rows: usize,
    pub(crate) spec_n_bins: usize,
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
}

impl<F> StreamCore<F> {
    pub(crate) fn new(
        n_fft: usize,
        hop: usize,
        sample_rate: u32,
        neighborhood: usize,
        log_floor_power: f32,
        zone: Zone,
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
        self.spec_first_frame = 0;
        self.n_frames_total = 0;
        self.last_pd_frame = -1;
        self.bucket_pending.clear();
        self.last_finalized_bucket = -1;
        self.pending_anchors.clear();
        self.to_finalize.clear();
        self.emitted.clear();
    }

    /// Append `self.frame_scratch` to the rolling spec buffer, dropping
    /// the oldest row if at capacity. Avoids the per-frame `Vec::clone`.
    fn append_frame_scratch_row(&mut self) {
        let n_bins = self.spec_n_bins;
        debug_assert_eq!(self.frame_scratch.len(), n_bins);
        let cap = 2 * self.neighborhood + 1;
        if self.spec_n_rows == cap {
            self.spec.copy_within(n_bins.., 0);
            self.spec_first_frame += 1;
            self.spec_n_rows -= 1;
        }
        let dst_start = self.spec_n_rows * n_bins;
        // Disjoint borrow: `self.spec` (mut) and `self.frame_scratch`
        // (shared) are different fields of `self`, so this is sound.
        self.spec[dst_start..dst_start + n_bins].copy_from_slice(&self.frame_scratch);
        self.spec_n_rows += 1;
    }

    /// Peak-detect rows `[from_row, to_row]` (spec-relative) and push
    /// survivors into `bucket_pending`.
    fn detect_rows_range(&mut self, cfg: PeakCfg, from_row: usize, to_row: usize) {
        if self.spec_n_rows == 0 || from_row > to_row {
            return;
        }
        let n_bins = self.spec_n_bins;
        for row in from_row..=to_row {
            if row >= self.spec_n_rows {
                break;
            }
            let abs_f = self.spec_first_frame + row as u32;
            let bucket = (abs_f as f32 / self.frames_per_sec) as u32;
            let row_start = row * n_bins;
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
                    match self.bucket_pending.binary_search_by_key(&bucket, |e| e.0) {
                        Ok(idx) => self.bucket_pending[idx].1.push(peak),
                        Err(idx) => self.bucket_pending.insert(idx, (bucket, vec![peak])),
                    }
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

        for peak in peaks {
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
            // Register this peak as a new ANCHOR.
            // If a hard cap is configured, evict oldest anchors first
            // so memory stays bounded under adversarial / dense input.
            if let Some(limit) = cfg.max_pending_anchors {
                while self.pending_anchors.len() >= limit {
                    self.pending_anchors.pop_front();
                }
            }
            self.pending_anchors.push_back(PendingAnchor {
                peak,
                targets: Vec::new(),
            });
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
    pub(crate) fn emit_finalized_anchors(
        &mut self,
        cfg: PeakCfg,
        emit_anchor: impl FnMut(&PendingAnchor, PeakCfg, &mut Vec<(TimestampMs, F)>),
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
            emit_anchor(&anchor, cfg, &mut emitted);
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
        emit_anchor: impl FnMut(&PendingAnchor, PeakCfg, &mut Vec<(TimestampMs, F)>),
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
        emit_anchor: impl FnMut(&PendingAnchor, PeakCfg, &mut Vec<(TimestampMs, F)>),
    ) {
        let n_bins = self.spec_n_bins;
        let spec = &self.spec;
        let spec_first_frame = self.spec_first_frame;
        let bucket_pending = &mut self.bucket_pending;
        let last_pd = &mut self.last_pd_frame;
        let min_mag = cfg.min_anchor_mag_db;
        let frames_per_sec = self.frames_per_sec;

        self.peak_det
            .flush(&mut self.peak_row_max, |ripe_abs, max_row| {
                let row_idx = (ripe_abs - spec_first_frame) as usize;
                let bucket = (ripe_abs as f32 / frames_per_sec) as u32;
                for (bin, &row_max) in max_row.iter().enumerate() {
                    let idx = row_idx * n_bins + bin;
                    let v = spec[idx];
                    if v > min_mag && v >= row_max {
                        let peak = Peak {
                            t_frame: ripe_abs,
                            f_bin: bin as u16,
                            _pad: 0,
                            mag: v,
                        };
                        match bucket_pending.binary_search_by_key(&bucket, |e| e.0) {
                            Ok(idx) => bucket_pending[idx].1.push(peak),
                            Err(idx) => bucket_pending.insert(idx, (bucket, vec![peak])),
                        }
                    }
                }
                *last_pd = ripe_abs as i32;
            });

        self.to_finalize.clear();
        self.to_finalize
            .extend(self.bucket_pending.iter().map(|e| e.0));
        let n = self.to_finalize.len();
        for i in 0..n {
            let bucket = self.to_finalize[i];
            self.finalize_bucket(cfg, bucket, &mut add_target);
        }
        self.to_finalize.clear();

        // Flush drains *all* pending anchors unconditionally — the zone
        // check is a liveness optimization for steady-state `push`, but at
        // end-of-stream every anchor's full lookahead has been observed, so
        // the remaining ones must emit (and the queue must empty).
        let mut emitted = core::mem::take(&mut self.emitted);
        let mut emit_anchor = emit_anchor;
        while let Some(anchor) = self.pending_anchors.pop_front() {
            emit_anchor(&anchor, cfg, &mut emitted);
        }
        self.emitted = emitted;
    }
}
