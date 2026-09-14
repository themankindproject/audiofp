//! Shared vote-consolidation helpers for Wang and Panako matchers/indexes.
//!
//! Centralises peak/plateau selection and neighbourhood sums so the 1:1
//! matcher paths and index paths cannot drift.

extern crate alloc;

use alloc::vec::Vec;

/// Centre index of the **first** connected maximal plateau at `peak_val`
/// in a dense consolidated histogram.
///
/// Ordinary single-plateau peaks keep the same median-centre behaviour as
/// before; disconnected equal peaks no longer pick a midpoint through a
/// zero-valued gap (audit F02).
#[must_use]
pub(crate) fn first_connected_plateau_center(consolidated: &[u32], peak_val: u32) -> usize {
    if peak_val == 0 {
        return 0;
    }
    let mut i = 0usize;
    while i < consolidated.len() {
        if consolidated[i] == peak_val {
            let start = i;
            while i < consolidated.len() && consolidated[i] == peak_val {
                i += 1;
            }
            let end = i - 1;
            return (start + end) / 2;
        }
        i += 1;
    }
    consolidated
        .iter()
        .position(|&v| v == peak_val)
        .unwrap_or(0)
}

/// Peak vote count and offset for sparse Wang bins using a ±`tol` box
/// filter over every integer centre in `[dmin, dmax]`.
///
/// Returns `(peak_votes, peak_center_offset, dense_consolidated_sum)` where
/// `dense_consolidated_sum` is the sum of consolidated values across the
/// whole dense span (for prominence parity with [`WangMatcher`](super::WangMatcher)).
#[must_use]
pub(crate) fn sparse_wang_box_peak(
    bins: &[(i64, u32)],
    tol: i64,
    dmin: i64,
    dmax: i64,
) -> (u32, i64, u64) {
    if bins.is_empty() || dmax < dmin {
        return (0, dmin, 0);
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
    (peak, best_start + (best_end - best_start) / 2, dense_sum)
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

/// `(scale_bin, start, end)` row bounds for a `bin_vec` sorted by
/// `(scale_bin, offset)`.
#[must_use]
pub(crate) fn panako_scale_rows(bin_vec: &[((u32, i64), u32)]) -> Vec<(u32, usize, usize)> {
    let mut rows = Vec::new();
    let mut i = 0usize;
    while i < bin_vec.len() {
        let s = bin_vec[i].0.0;
        let start = i;
        while i < bin_vec.len() && bin_vec[i].0.0 == s {
            i += 1;
        }
        rows.push((s, start, i));
    }
    rows
}

/// Neighbourhood vote sum for Panako 2-D consolidation: scale ±1 and offset
/// ±`tol` around `center_i`, scanning each scale row independently so an
/// early break within one row does not skip valid neighbours on adjacent
/// scale rows (audit F04).
#[must_use]
pub(crate) fn panako_neighborhood_sum(
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

pub(crate) fn panako_vote_prefix(bins: &[((u32, i64), u32)]) -> Vec<u64> {
    let mut prefix = Vec::with_capacity(bins.len() + 1);
    prefix.push(0);
    let mut total = 0u64;
    for &(_, votes) in bins {
        total += u64::from(votes);
        prefix.push(total);
    }
    prefix
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
        let rows = panako_scale_rows(&bins);
        let prefix = panako_vote_prefix(&bins);
        for i in 0..bins.len() {
            assert_eq!(
                panako_neighborhood_sum(&bins, &rows, &prefix, i, 1),
                panako_neighborhood_sum_naive(&bins, i, 1),
                "mismatch at center index {i}"
            );
        }
    }

    #[test]
    fn sparse_wang_jitter_peak_at_unoccupied_center() {
        let bins = vec![(0i64, 3u32), (2, 3)];
        let (peak, center, _) = sparse_wang_box_peak(&bins, 1, -10, 10);
        assert_eq!(peak, 6);
        assert_eq!(center, 1);
    }

    #[test]
    fn sparse_peak_searches_past_lower_local_maximum() {
        assert_eq!(
            sparse_wang_box_peak(&[(0, 1), (10, 9)], 0, -5, 20),
            (9, 10, 10)
        );
    }

    #[test]
    fn sparse_peak_negative_plateau_rounds_like_dense_indices() {
        assert_eq!(sparse_wang_box_peak(&[(-4, 3)], 1, -4, 10), (3, -4, 6));
    }

    #[test]
    fn sparse_peak_large_span_does_not_materialize_empty_frames() {
        assert_eq!(
            sparse_wang_box_peak(&[(4_000_000_000, 9)], 1, -4_000_000_000, 4_000_000_001),
            (9, 4_000_000_000, 27)
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
                let expected = (
                    peak,
                    -20 + ((start + end) / 2) as i64,
                    dense.iter().map(|&v| u64::from(v)).sum(),
                );
                assert_eq!(sparse_wang_box_peak(&bins, tol, -20, 20), expected);
                assert_eq!(dense_wang_consolidated_sum(&bins, tol, -20, 20), expected.2);
            }
        }
    }
}
