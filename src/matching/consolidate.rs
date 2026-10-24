//! Shared vote-consolidation helpers for Wang and Panako matchers/indexes.
//!
//! Centralises peak/plateau selection and neighbourhood sums so the 1:1
//! matcher paths and index paths cannot drift.

extern crate alloc;

use alloc::vec::Vec;

/// Bounds `(start, end)` (inclusive) of the **first** connected maximal
/// plateau at `peak_val` in a dense consolidated histogram; `(0, 0)` when
/// there is none. Taking the first connected run means disconnected equal
/// peaks never yield a midpoint through a zero-valued gap (audit F02).
#[must_use]
pub(crate) fn first_connected_plateau(consolidated: &[u32], peak_val: u32) -> (usize, usize) {
    if peak_val == 0 {
        return (0, 0);
    }
    match consolidated.iter().position(|&v| v == peak_val) {
        Some(start) => {
            let len = consolidated[start..]
                .iter()
                .take_while(|&&v| v == peak_val)
                .count();
            (start, start + len - 1)
        }
        None => (0, 0),
    }
}

/// Reported offset for a consolidated peak: the most-voted raw offset in
/// the plateau's ±`tol` window; when several offsets tie for the most votes,
/// the integer nearest their mean (ties to the lower one). `None` when the
/// window holds no votes.
///
/// The plateau centre alone is a biased estimate: one background vote at
/// `δ* − 2` turns a lone spike's 3-wide plateau into a 1-wide one at
/// `δ* − 1`, and a vote at `δ* − 1` makes it 2 wide with a floored midpoint
/// of `δ* − 1`. Votes split evenly across `δ* ± 1` still report `δ*`.
/// Scores and prominence keep coming from the plateau.
#[must_use]
pub(crate) fn mode_offset(votes: impl Iterator<Item = (i64, u32)>) -> Option<i64> {
    let (mut best, mut sum, mut n) = (0u32, 0i128, 0i128);
    for (d, v) in votes {
        if v > best {
            (best, sum, n) = (v, i128::from(d), 1);
        } else if v == best && v > 0 {
            sum += i128::from(d);
            n += 1;
        }
    }
    if best == 0 {
        return None;
    }
    // Nearest integer to `sum / n`, ties to the lower: ceil((2·sum − n) / 2n).
    let num = 2 * sum - n;
    let den = 2 * n;
    Some((-(-num).div_euclid(den)) as i64)
}

/// [`mode_offset`] over a dense histogram window `raw[lo..=hi]`, as an index.
#[must_use]
pub(crate) fn dense_mode_offset(raw: &[u32], lo: usize, hi: usize) -> Option<usize> {
    let hi = hi.min(raw.len().saturating_sub(1));
    mode_offset((lo..=hi).map(|i| (i as i64, raw[i]))).map(|i| i as usize)
}

/// Peak vote count and offset for sparse Wang bins using a ±`tol` box
/// filter over every integer centre in `[dmin, dmax]`.
///
/// Returns `(peak_votes, peak_center_offset, peak_mode_offset,
/// dense_consolidated_sum)`: the plateau centre (scoring), the most-voted
/// raw offset near it (the reported offset, see [`mode_offset`]), and the
/// sum of consolidated values across the whole dense span (for prominence
/// parity with [`WangMatcher`](super::WangMatcher)).
#[must_use]
pub(crate) fn sparse_wang_box_peak(
    bins: &[(i64, u32)],
    tol: i64,
    dmin: i64,
    dmax: i64,
) -> (u32, i64, i64, u64) {
    if bins.is_empty() || dmax < dmin {
        return (0, dmin, dmin, 0);
    }

    // The sorted bins produce two individually sorted event streams:
    // enter at d-tol, leave at d+tol+1. Merge them without materializing
    // events or visiting the (possibly billions of) empty frame positions.
    let end = dmax + 1;
    let (mut enter, mut leave) = (0usize, 0usize);
    let mut current = 0u64;
    let mut position = dmin;
    let mut peak = 0u32;
    let (mut best_start, mut best_end) = (dmin, dmin);
    let mut dense_sum = 0u64;
    while position < end {
        while enter < bins.len() && bins[enter].0 - tol <= position {
            current += u64::from(bins[enter].1);
            enter += 1;
        }
        while leave < bins.len() && bins[leave].0 + tol < position {
            current -= u64::from(bins[leave].1);
            leave += 1;
        }
        let next_enter = bins.get(enter).map_or(end, |&(d, _)| d - tol);
        let next_leave = bins.get(leave).map_or(end, |&(d, _)| d + tol + 1);
        let next = next_enter.min(next_leave).min(end);
        let value = current.min(u64::from(u32::MAX)) as u32;
        dense_sum += u64::from(value) * (next - position) as u64;
        if value > peak {
            peak = value;
            best_start = position;
            best_end = next - 1;
        } else if value == peak && position == best_end + 1 {
            // Zero-net event positions do not split a connected plateau.
            best_end = next - 1;
        }
        position = next;
    }
    let center = best_start + (best_end - best_start) / 2;
    let (lo, hi) = (best_start - tol, best_end + tol);
    let start = bins.partition_point(|&(d, _)| d < lo);
    let end = bins.partition_point(|&(d, _)| d <= hi);
    let mode = mode_offset(bins[start..end].iter().copied()).unwrap_or(center);
    (peak, center, mode, dense_sum)
}

/// Sum of consolidated vote counts across every integer centre in the dense
/// span — used for Wang prominence background parity.
#[cfg(test)]
#[must_use]
pub(crate) fn dense_wang_consolidated_sum(
    bins: &[(i64, u32)],
    tol: i64,
    dmin: i64,
    dmax: i64,
) -> u64 {
    if bins.is_empty() || dmax < dmin {
        return 0;
    }
    let mut total = 0u64;
    for &(d, c) in bins {
        let win_lo = (d - tol).max(dmin);
        let win_hi = (d + tol).min(dmax);
        if win_hi >= win_lo {
            let centers = (win_hi - win_lo + 1) as u64;
            total += (c as u64) * centers;
        }
    }
    total
}

/// Reusable buffers for [`panako_hough_peak`]: callers that consolidate
/// many accumulators (one per candidate in `PanakoIndex::query`) keep one
/// and pay the allocations once.
#[derive(Default)]
pub(crate) struct HoughScratch {
    rows: Vec<(u32, usize, usize)>,
    prefix: Vec<u64>,
    by_offset: Vec<(i64, u32)>,
    /// `consolidated[i]`: neighbourhood vote sum around `bin_vec[i]`.
    pub(crate) consolidated: Vec<u32>,
}

/// Consolidate a Panako `(scale_bin, offset) → votes` accumulator sorted by
/// key and find its peak: returns `(peak_votes, peak_idx)`, the first bin
/// with the largest neighbourhood sum (±1 scale bin, ±`tol` frames).
/// `scratch.consolidated` is left aligned with `bin_vec` for prominence.
///
/// Shared by `PanakoMatcher` and `PanakoIndex::query`, which previously
/// carried their own copies of this loop.
pub(crate) fn panako_hough_peak(
    bin_vec: &[((u32, i64), u32)],
    tol: i64,
    scratch: &mut HoughScratch,
) -> (u32, usize) {
    panako_scale_rows_into(bin_vec, &mut scratch.rows);
    panako_vote_prefix_into(bin_vec, &mut scratch.prefix);
    scratch.consolidated.clear();
    let mut peak_votes = 0u32;
    let mut peak_idx = 0usize;
    for i in 0..bin_vec.len() {
        let neigh = panako_neighborhood_sum(bin_vec, &scratch.rows, &scratch.prefix, i, tol);
        scratch.consolidated.push(neigh);
        if neigh > peak_votes {
            peak_votes = neigh;
            peak_idx = i;
        }
    }
    (peak_votes, peak_idx)
}

/// Panako prominence: `peak / (mean_background + 1)` on the consolidated
/// accumulator, where the background is every occupied bin **outside the
/// peak's reach** — more than 2 scale bins or more than `2·tol` frames
/// from it.
///
/// A bin inside that window shares votes with the peak through
/// consolidation (both neighbourhoods cover the same raw bins). Counting
/// those as background — as the generic [`compute_prominence`] over every
/// occupied bin did — made a clean sparse match measure its own spread:
/// 20 jittered triplets with no noise scored 0.95, under the default
/// `min_prominence` of 5, while the same evidence passed `WangMatcher`.
///
/// [`compute_prominence`]: super::compute_prominence
#[must_use]
pub(crate) fn panako_prominence(
    bin_vec: &[((u32, i64), u32)],
    consolidated: &[u32],
    peak_idx: usize,
    tol: i64,
) -> f32 {
    let peak = consolidated[peak_idx] as f32;
    if peak == 0.0 {
        return 0.0;
    }
    let ((s_bin, off_key), _) = bin_vec[peak_idx];
    let reach = tol.saturating_mul(2);
    let (mut sum, mut n) = (0u64, 0u64);
    for (&((s, o), _), &c) in bin_vec.iter().zip(consolidated) {
        if s.abs_diff(s_bin) > 2 || o.abs_diff(off_key) > reach as u64 {
            sum += u64::from(c);
            n += 1;
        }
    }
    let mean_rest = if n > 0 { sum as f32 / n as f32 } else { 0.0 };
    peak / (mean_rest + 1.0)
}

/// Reported offset for the Panako peak at `bin_vec[peak_idx]`: the
/// [`mode_offset`] of its consolidation neighbourhood (±1 scale bin,
/// ±`tol` frames), with votes at one offset summed across scale bins.
pub(crate) fn panako_peak_offset(
    bin_vec: &[((u32, i64), u32)],
    peak_idx: usize,
    tol: i64,
    scratch: &mut HoughScratch,
) -> i64 {
    let ((s_bin, off_key), _) = bin_vec[peak_idx];
    let (lo, hi) = (off_key.saturating_sub(tol), off_key.saturating_add(tol));
    let by_offset = &mut scratch.by_offset;
    by_offset.clear();
    for &((s, o), v) in bin_vec {
        if s.abs_diff(s_bin) <= 1 && (lo..=hi).contains(&o) {
            by_offset.push((o, v));
        }
    }
    by_offset.sort_unstable_by_key(|&(o, _)| o);
    let mut merged = 0usize;
    for i in 0..by_offset.len() {
        if merged > 0 && by_offset[merged - 1].0 == by_offset[i].0 {
            by_offset[merged - 1].1 += by_offset[i].1;
        } else {
            by_offset[merged] = by_offset[i];
            merged += 1;
        }
    }
    by_offset.truncate(merged);
    mode_offset(by_offset.iter().copied()).unwrap_or(off_key)
}

/// Run-length count a key-sorted vote list into `(key, votes)` bins, in key
/// order. Equal to collecting a `key → count` map and sorting its entries,
/// without the map.
pub(crate) fn count_sorted<K: Copy + PartialEq>(sorted: &[K], out: &mut Vec<(K, u32)>) {
    out.clear();
    for &k in sorted {
        match out.last_mut() {
            Some((last, n)) if *last == k => *n += 1,
            _ => out.push((k, 1)),
        }
    }
}

/// `(scale_bin, start, end)` row bounds for a `bin_vec` sorted by
/// `(scale_bin, offset)`.
fn panako_scale_rows_into(bin_vec: &[((u32, i64), u32)], rows: &mut Vec<(u32, usize, usize)>) {
    rows.clear();
    let mut i = 0usize;
    while i < bin_vec.len() {
        let s = bin_vec[i].0.0;
        let start = i;
        while i < bin_vec.len() && bin_vec[i].0.0 == s {
            i += 1;
        }
        rows.push((s, start, i));
    }
}

/// Neighbourhood vote sum for Panako 2-D consolidation: scale ±1 and offset
/// ±`tol` around `center_i`, scanning each scale row independently so an
/// early break within one row does not skip valid neighbours on adjacent
/// scale rows (audit F04).
#[must_use]
fn panako_neighborhood_sum(
    bin_vec: &[((u32, i64), u32)],
    rows: &[(u32, usize, usize)],
    prefix: &[u64],
    center_i: usize,
    tol: i64,
) -> u32 {
    let (s_bin, off_key) = bin_vec[center_i].0;
    let mut sum = 0u64;
    for ns in [s_bin.checked_sub(1), Some(s_bin), s_bin.checked_add(1)]
        .into_iter()
        .flatten()
    {
        if let Ok(row_idx) = rows.binary_search_by_key(&ns, |&(s, _, _)| s) {
            let (_, start, end) = rows[row_idx];
            let row = &bin_vec[start..end];
            let lo = off_key.saturating_sub(tol);
            let hi = off_key.saturating_add(tol);
            let i0 = row.partition_point(|&((_, no), _)| no < lo);
            let i1 = row.partition_point(|&((_, no), _)| no <= hi);
            sum += prefix[start + i1] - prefix[start + i0];
        }
    }
    sum.min(u64::from(u32::MAX)) as u32
}

fn panako_vote_prefix_into(bins: &[((u32, i64), u32)], prefix: &mut Vec<u64>) {
    prefix.clear();
    prefix.reserve(bins.len() + 1);
    prefix.push(0);
    let mut total = 0u64;
    for &(_, votes) in bins {
        total += u64::from(votes);
        prefix.push(total);
    }
}

/// Naive O(B²) Panako neighbourhood oracle for regression tests.
#[cfg(test)]
pub(crate) fn panako_neighborhood_sum_naive(
    bin_vec: &[((u32, i64), u32)],
    center_i: usize,
    tol: i64,
) -> u32 {
    let (s_bin, off_key) = bin_vec[center_i].0;
    let mut sum = 0u32;
    for &((ns, no), v) in bin_vec.iter() {
        if ns.abs_diff(s_bin) <= 1 && (no - off_key).abs() <= tol {
            sum += v;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn panako_neighborhood_efficient_matches_naive() {
        let bins = vec![
            ((0u32, -10i64), 1u32),
            ((0, 0), 3),
            ((0, 10), 1),
            ((1, -5), 2),
            ((1, 0), 3),
            ((1, 5), 1),
            ((2, 0), 2),
        ];
        let mut scratch = HoughScratch::default();
        let (peak, peak_idx) = panako_hough_peak(&bins, 1, &mut scratch);
        for i in 0..bins.len() {
            assert_eq!(
                scratch.consolidated[i],
                panako_neighborhood_sum_naive(&bins, i, 1),
                "mismatch at center index {i}"
            );
        }
        assert_eq!(peak, scratch.consolidated[peak_idx]);
        assert_eq!(peak, *scratch.consolidated.iter().max().unwrap());
    }

    #[test]
    fn mode_offset_resists_background_votes_and_splits_ties() {
        // Lone spike with a background vote two frames below: the plateau
        // collapses to δ* − 1, the mode does not.
        assert_eq!(mode_offset([(998, 1), (1000, 40)].into_iter()), Some(1000));
        // Even split: the mean of the tied offsets.
        assert_eq!(mode_offset([(0, 3), (2, 3)].into_iter()), Some(1));
        // Half-way between two integers: the lower one.
        assert_eq!(mode_offset([(999, 20), (1000, 20)].into_iter()), Some(999));
        assert_eq!(mode_offset([(-3, 5), (-2, 5)].into_iter()), Some(-3));
        assert_eq!(mode_offset([(-3, 0)].into_iter()), None);
        assert_eq!(dense_mode_offset(&[0, 1, 0, 40, 0], 0, 9), Some(3));
    }

    #[test]
    fn sparse_wang_jitter_peak_at_unoccupied_center() {
        let bins = vec![(0i64, 3u32), (2, 3)];
        let (peak, center, mode, _) = sparse_wang_box_peak(&bins, 1, -10, 10);
        assert_eq!(peak, 6);
        assert_eq!(center, 1);
        assert_eq!(mode, 1, "an even split reports the midpoint");
    }

    #[test]
    fn sparse_peak_searches_past_lower_local_maximum() {
        assert_eq!(
            sparse_wang_box_peak(&[(0, 1), (10, 9)], 0, -5, 20),
            (9, 10, 10, 10)
        );
    }

    #[test]
    fn sparse_peak_negative_plateau_rounds_like_dense_indices() {
        assert_eq!(sparse_wang_box_peak(&[(-4, 3)], 1, -4, 10), (3, -4, -4, 6));
    }

    #[test]
    fn sparse_peak_large_span_does_not_materialize_empty_frames() {
        assert_eq!(
            sparse_wang_box_peak(&[(4_000_000_000, 9)], 1, -4_000_000_000, 4_000_000_001),
            (9, 4_000_000_000, 4_000_000_000, 27)
        );
    }

    #[test]
    fn sparse_peak_matches_dense_randomized_oracle() {
        let mut seed = 17u32;
        for _ in 0..200 {
            let bins: Vec<_> = (-20i64..=20)
                .filter_map(|d| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    let c = seed % 7;
                    (c > 3).then_some((d, c))
                })
                .collect();
            for tol in [0i64, 1, 4, 100] {
                let dense: Vec<u32> = (-20i64..=20)
                    .map(|x| {
                        bins.iter()
                            .filter(|(d, _)| (x - d).abs() <= tol)
                            .map(|(_, c)| c)
                            .sum()
                    })
                    .collect();
                let peak = *dense.iter().max().unwrap();
                let start = dense.iter().position(|&v| v == peak).unwrap();
                let end = start + dense[start..].iter().take_while(|&&v| v == peak).count() - 1;
                let center = -20 + ((start + end) / 2) as i64;
                let (lo, hi) = (-20 + start as i64 - tol, -20 + end as i64 + tol);
                let in_window: Vec<_> =
                    bins.iter().filter(|(d, _)| (lo..=hi).contains(d)).collect();
                let top = in_window.iter().map(|(_, c)| *c).max().unwrap();
                let tied: Vec<f64> = in_window
                    .iter()
                    .filter(|(_, c)| *c == top)
                    .map(|(d, _)| *d as f64)
                    .collect();
                let mean = tied.iter().sum::<f64>() / tied.len() as f64;
                let mode = (mean - 0.5).ceil() as i64;
                let expected = (
                    peak,
                    center,
                    mode,
                    dense.iter().map(|&v| u64::from(v)).sum(),
                );
                assert_eq!(sparse_wang_box_peak(&bins, tol, -20, 20), expected);
                assert_eq!(dense_wang_consolidated_sum(&bins, tol, -20, 20), expected.3);
            }
        }
    }
}
