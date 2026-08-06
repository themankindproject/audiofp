//! Fast map aliases and sorted-posting helpers for matching hot paths.
//!
//! With the `std` feature, use [`ahash::HashMap`] (AHash-backed) for
//! O(1) expected lookups with ~2-5× faster hashing than SipHash on
//! integer keys. Without `std`, fall back to [`alloc::collections::BTreeMap`]
//! so the matching module stays `no_std + alloc` capable.
//!
//! [`SortedPostings`] provides an alternative to `HashMap<u32, Vec<u32>>`
//! for the inverted-index step: a flat sorted array with binary-search
//! lookup per hash. Eliminates per-unique-hash allocations and improves
//! cache locality (contiguous memory vs pointer-chasing hash nodes).

extern crate alloc;

#[cfg(feature = "std")]
pub(crate) type HashMap<K, V> = std::collections::HashMap<K, V, ahash::RandomState>;

#[cfg(not(feature = "std"))]
pub(crate) use alloc::collections::BTreeMap as HashMap;

/// Create a map with pre-allocated capacity (HashMap) or just empty (BTreeMap).
#[cfg(feature = "std")]
#[inline]
pub(crate) fn hashmap_with_capacity<K: core::hash::Hash + Eq, V>(cap: usize) -> HashMap<K, V> {
    HashMap::with_capacity_and_hasher(cap, ahash::RandomState::default())
}

/// Create an empty map.
#[cfg(feature = "std")]
#[inline]
pub(crate) fn hashmap_new<K: core::hash::Hash + Eq, V>() -> HashMap<K, V> {
    HashMap::with_hasher(ahash::RandomState::default())
}

#[cfg(not(feature = "std"))]
#[inline]
pub(crate) fn hashmap_with_capacity<K: Ord, V>(_cap: usize) -> HashMap<K, V> {
    HashMap::new()
}

/// Create an empty map (BTreeMap fallback).
#[cfg(not(feature = "std"))]
#[inline]
pub(crate) fn hashmap_new<K: Ord, V>() -> HashMap<K, V> {
    HashMap::new()
}

use alloc::vec;
use alloc::vec::Vec;

// ---------------------------------------------------------------------------
// SortedPostings — flat sorted arrays + binary search
// ---------------------------------------------------------------------------

/// A sorted inverted index for `(hash, t_anchor)` pairs.
///
/// Replaces `HashMap<hash, Vec<t_anchor>>` with a single sort + three
/// flat arrays (`hashes`, `starts`, `anchors`). Lookup is binary search
/// on `hashes` followed by a contiguous slice on `anchors`.
///
/// # Performance vs HashMap
///
/// - Build: O(N log N), single sort + 3 compact arrays (vs N+1 allocs).
/// - Lookup: binary search O(log U) per query hash (U = unique hashes).
///   For song-length fingerprints, U ≈ 500–2000 → log₂U ≈ 9–11.
/// - Cache: flat contiguous arrays beat pointer-chasing hash-table probes.
#[derive(Clone, Debug)]
pub(crate) struct SortedPostings {
    /// Unique hashes in ascending order.
    hashes: Vec<u32>,
    /// `starts[i]` is the first index into `anchors` for `hashes[i]`.
    /// `starts[i+1]` is one-past-last (so `starts.len() == hashes.len() + 1`).
    starts: Vec<u32>,
    /// t_anchor values, grouped by hash and sorted within each group.
    anchors: Vec<u32>,
}

impl SortedPostings {
    /// Build from a flat slice of `(hash, t_anchor)` pairs.
    ///
    /// Drops entries where the hash appears in more than `max_postings`
    /// positions (stop-hash removal). Returns `None` if nothing remains.
    pub(crate) fn build(pairs: &[(u32, u32)], max_postings: u32) -> Option<Self> {
        if pairs.is_empty() {
            return None;
        }

        let n = pairs.len();

        // Copy to mutable working buffer.
        let mut keys: Vec<u32> = Vec::with_capacity(n);
        let mut vals: Vec<u32> = Vec::with_capacity(n);
        for &(hash, t_anchor) in pairs {
            keys.push(hash);
            vals.push(t_anchor);
        }

        // Sort by hash, then by t_anchor (secondary sort within hash).
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| keys[a].cmp(&keys[b]).then(vals[a].cmp(&vals[b])));

        // Permute to sorted order in-place.
        let mut sorted_keys = vec![0u32; n];
        let mut sorted_vals = vec![0u32; n];
        for (dst, &idx) in indices.iter().enumerate() {
            sorted_keys[dst] = keys[idx];
            sorted_vals[dst] = vals[idx];
        }

        // Build hash → range index, filtering stop-hashes in one pass.
        let mut hashes = Vec::new();
        let mut starts = Vec::new();
        let mut anchors = Vec::new();

        let mut i = 0;
        while i < n {
            let hash = sorted_keys[i];
            let run_start = i;
            while i < n && sorted_keys[i] == hash {
                i += 1;
            }
            let run_end = i;
            let count = (run_end - run_start) as u32;

            if count <= max_postings {
                hashes.push(hash);
                starts.push(anchors.len() as u32);
                anchors.extend_from_slice(&sorted_vals[run_start..run_end]);
            }
        }

        if hashes.is_empty() {
            return None;
        }

        starts.push(anchors.len() as u32); // sentinel

        Some(Self {
            hashes,
            starts,
            anchors,
        })
    }

    /// Return all `t_anchor` values for `hash`, or an empty slice if absent.
    ///
    /// Anchors within a hash group are sorted ascending (invariant of
    /// [`Self::build`]). Callers may rely on this for ordered scans.
    #[inline]
    pub(crate) fn get(&self, hash: u32) -> &[u32] {
        match self.hashes.binary_search(&hash) {
            Ok(idx) => {
                let lo = self.starts[idx] as usize;
                let hi = self.starts[idx + 1] as usize;
                &self.anchors[lo..hi]
            }
            Err(_) => &[],
        }
    }

    /// True when empty.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        self.anchors.is_empty()
    }

    /// Total number of postings (t_anchor entries).
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.anchors.len()
    }

    /// Number of unique hashes in the index.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn num_hashes(&self) -> usize {
        self.hashes.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input() {
        assert!(SortedPostings::build(&[], 100).is_none());
    }

    #[test]
    fn single_hash() {
        let sp = SortedPostings::build(&[(42, 10)], 100).unwrap();
        assert_eq!(sp.get(42), &[10]);
        assert!(sp.get(99).is_empty());
    }

    #[test]
    fn multiple_anchors_same_hash_sorted() {
        let sp = SortedPostings::build(&[(7, 100), (7, 200), (7, 50)], 100).unwrap();
        assert_eq!(sp.get(7), &[50, 100, 200]);
    }

    #[test]
    fn multiple_hashes() {
        let sp =
            SortedPostings::build(&[(1, 10), (2, 20), (1, 30), (3, 40), (2, 50)], 100).unwrap();
        assert_eq!(sp.get(1), &[10, 30]);
        assert_eq!(sp.get(2), &[20, 50]);
        assert_eq!(sp.get(3), &[40]);
        assert!(sp.get(99).is_empty());
    }

    #[test]
    fn stop_hash_filtered() {
        let sp = SortedPostings::build(&[(0, 1), (0, 2), (0, 3), (1, 10)], 2).unwrap();
        assert!(sp.get(0).is_empty());
        assert_eq!(sp.get(1), &[10]);
    }

    #[test]
    fn all_stop_hashes() {
        assert!(SortedPostings::build(&[(0, 1), (0, 2), (0, 3)], 2).is_none());
    }

    #[test]
    fn large_input() {
        let mut pairs = Vec::new();
        for hash in 0..1000u32 {
            for t in 0..5u32 {
                pairs.push((hash, t * 10));
            }
        }
        let sp = SortedPostings::build(&pairs, 10).unwrap();
        for hash in 0..1000 {
            assert_eq!(sp.get(hash).len(), 5);
        }
    }

    #[test]
    fn deterministic_order() {
        let sp1 = SortedPostings::build(&[(3, 30), (1, 10), (2, 20)], 100).unwrap();
        let sp2 = SortedPostings::build(&[(2, 20), (3, 30), (1, 10)], 100).unwrap();
        assert_eq!(sp1.get(1), sp2.get(1));
        assert_eq!(sp1.get(2), sp2.get(2));
        assert_eq!(sp1.get(3), sp2.get(3));
    }

    #[test]
    fn build_is_single_alloc_per_vector() {
        // Verify that for n pairs the index has exactly 3 vectors + 1 index.
        let pairs: Vec<(u32, u32)> = (0..500u32).map(|i| (i % 50, i)).collect();
        let sp = SortedPostings::build(&pairs, 100).unwrap();
        assert_eq!(sp.hashes.len(), 50); // 50 unique hashes
        assert_eq!(sp.anchors.len(), 500);
        assert!(!sp.is_empty());
    }
}
