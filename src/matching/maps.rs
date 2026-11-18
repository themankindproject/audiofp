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

/// Set alias matching [`HashMap`]: AHash-backed under `std`, `BTreeSet`
/// otherwise.
#[cfg(feature = "std")]
pub(crate) type HashSet<K> = std::collections::HashSet<K, ahash::RandomState>;

#[cfg(not(feature = "std"))]
pub(crate) use alloc::collections::BTreeSet as HashSet;

/// Create an empty set.
#[cfg(feature = "std")]
#[inline]
pub(crate) fn hashset_new<K: core::hash::Hash + Eq>() -> HashSet<K> {
    HashSet::with_hasher(ahash::RandomState::default())
}

#[cfg(not(feature = "std"))]
#[inline]
pub(crate) fn hashset_new<K: Ord>() -> HashSet<K> {
    HashSet::new()
}

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

use alloc::vec::Vec;

/// A posting list that stores a single posting inline.
///
/// `P` is the posting tuple: `(ref_id, position)` for the Wang and Haitsma
/// indexes, `(ref_id, t_anchor, t_b, t_c)` for Panako.
///
/// Measured on 300-reference catalogs, 82 % (Wang) and 85 % (Panako) of
/// hash keys have exactly one posting. As a `Vec<P>` each of those costs a
/// separate heap allocation (rounded up by the allocator) on top of the
/// `Vec` header in the map slot. Here the singleton lives in the slot
/// itself, so the common case allocates nothing and reading it needs no
/// pointer chase. The enum is exactly the size of a `Vec` for both posting
/// sizes on 64-bit targets (24 bytes); on 32-bit targets it is for
/// `(ref_id, position)` and 8 bytes larger for Panako's 16-byte posting.
///
/// Behaviour is exactly that of the `Vec` it replaces: postings are kept in
/// insertion order, `retain` preserves the order of survivors, and iteration
/// yields postings in the same order, so index queries are bit-identical.
/// An emptied list is `Empty`; whether it stays in the map is the caller's
/// choice.
#[derive(Clone, Debug)]
pub(crate) enum Postings<P> {
    /// No postings (a list emptied by `remove`).
    Empty,
    /// Exactly one posting, stored inline.
    One(P),
    /// Two or more postings, on the heap.
    Many(Vec<P>),
}

impl<P> Default for Postings<P> {
    #[inline]
    fn default() -> Self {
        Postings::Empty
    }
}

impl<P: Copy> Postings<P> {
    /// Append a posting.
    #[inline]
    pub(crate) fn push(&mut self, p: P) {
        match self {
            Postings::Empty => *self = Postings::One(p),
            Postings::One(first) => *self = Postings::Many(alloc::vec![*first, p]),
            Postings::Many(v) => v.push(p),
        }
    }

    /// The postings as a slice, in insertion order.
    #[inline]
    pub(crate) fn as_slice(&self) -> &[P] {
        match self {
            Postings::Empty => &[],
            Postings::One(p) => core::slice::from_ref(p),
            Postings::Many(v) => v,
        }
    }

    /// Number of postings.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// `true` if the list holds no postings.
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    /// Iterate postings by reference, in insertion order.
    #[inline]
    pub(crate) fn iter(&self) -> core::slice::Iter<'_, P> {
        self.as_slice().iter()
    }

    /// Keep only the postings for which `keep` returns `true`, preserving
    /// order (same contract as [`Vec::retain`]).
    ///
    /// A spilled list gives its heap block back as it shrinks: none or one
    /// survivor returns to the inline forms, and a block left more than
    /// half unused is trimmed to fit. A catalog purge would otherwise keep
    /// every surviving list at its peak size.
    #[inline]
    pub(crate) fn retain(&mut self, mut keep: impl FnMut(&P) -> bool) {
        match self {
            Postings::Empty => {}
            Postings::One(p) => {
                if !keep(p) {
                    *self = Postings::Empty;
                }
            }
            Postings::Many(v) => {
                v.retain(|p| keep(p));
                match v[..] {
                    [] => *self = Postings::Empty,
                    [only] => *self = Postings::One(only),
                    _ if v.capacity() > 2 * v.len() => v.shrink_to_fit(),
                    _ => {}
                }
            }
        }
    }

    /// Heap bytes owned by this list (0 unless spilled to a `Vec`).
    #[inline]
    pub(crate) fn heap_bytes(&self) -> usize {
        match self {
            Postings::Many(v) => v.capacity() * core::mem::size_of::<P>(),
            _ => 0,
        }
    }
}

impl<'a, P: Copy> IntoIterator for &'a Postings<P> {
    type Item = &'a P;
    type IntoIter = core::slice::Iter<'a, P>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// The inline singleton must not grow the map slot beyond a `Vec` header.
const _: () = assert!(
    core::mem::size_of::<Postings<(u32, u32)>>() == core::mem::size_of::<Vec<(u32, u32)>>(),
    "Postings singleton must not grow the map slot beyond a Vec header"
);
// A 16-byte Panako posting fits in a 64-bit `Vec` header's niche (24 bytes)
// but not in a 32-bit one (12 bytes); there the enum is 20 bytes and each
// singleton still saves a heap allocation, so only 64-bit targets can hold
// it to the `Vec` size.
#[cfg(target_pointer_width = "64")]
const _: () = assert!(
    core::mem::size_of::<Postings<(u32, u32, u32, u32)>>()
        == core::mem::size_of::<Vec<(u32, u32, u32, u32)>>(),
    "Panako Postings must fit in a Vec header on 64-bit targets"
);

/// One-bit-per-slot membership filter over `u32` keys.
///
/// [`KeyFilter::may_contain`] is `true` for every key the filter was built
/// from, and for a small share of the others (about one in
/// `slots_per_key`). It is used to skip work an exact lookup would only
/// have found pointless: the reference entries no query key can hit, the
/// probes no reference frame can answer.
#[derive(Clone, Debug)]
pub(crate) struct KeyFilter {
    words: Vec<u64>,
    /// `32 − log2(slots)`: a slot is the top bits of a multiplicative hash
    /// of the key.
    shift: u32,
}

impl KeyFilter {
    /// Build a filter with `slots_per_key` slots per key, rounded up to a
    /// power of two between 64 and 2^24 slots (2 MiB).
    pub(crate) fn new(keys: impl ExactSizeIterator<Item = u32>, slots_per_key: usize) -> Self {
        let slots = keys
            .len()
            .saturating_mul(slots_per_key)
            .clamp(64, 1 << 24)
            .next_power_of_two();
        let mut filter = Self {
            words: alloc::vec![0; slots / 64],
            shift: 32 - slots.trailing_zeros(),
        };
        for key in keys {
            let slot = filter.slot(key);
            filter.words[slot / 64] |= 1 << (slot % 64);
        }
        filter
    }

    #[inline]
    fn slot(&self, key: u32) -> usize {
        (key.wrapping_mul(0x9E37_79B1) >> self.shift) as usize
    }

    /// `false` only if `key` is not one of the keys the filter was built
    /// from.
    #[inline]
    pub(crate) fn may_contain(&self, key: u32) -> bool {
        let slot = self.slot(key);
        self.words[slot / 64] & (1 << (slot % 64)) != 0
    }
}

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
    #[cfg(test)]
    pub(crate) fn build(pairs: &[(u32, u32)], max_postings: u32) -> Option<Self> {
        Self::build_sorted(pairs.to_vec(), max_postings)
    }

    /// Build from [`WangHash`] anchors without an intermediate projection
    /// buffer in the caller (audit perf #6).
    ///
    /// With `wanted`, only hashes the filter may contain are indexed. A
    /// kept hash keeps every one of its postings, so its stop-hash count —
    /// and with it every lookup of a wanted hash — is the same as in the
    /// full index.
    pub(crate) fn build_from_wang(
        hashes: &[crate::classical::WangHash],
        max_postings: u32,
        wanted: Option<&KeyFilter>,
    ) -> Option<Self> {
        let pair = |h: &crate::classical::WangHash| (h.hash, h.t_anchor);
        let sorted: Vec<(u32, u32)> = match wanted {
            None => hashes.iter().map(pair).collect(),
            Some(wanted) => hashes
                .iter()
                .filter(|h| wanted.may_contain(h.hash))
                .map(pair)
                .collect(),
        };
        Self::build_sorted(sorted, max_postings)
    }

    fn build_sorted(mut sorted: Vec<(u32, u32)>, max_postings: u32) -> Option<Self> {
        sorted.sort_unstable();

        // One run per hash; a run longer than `max_postings` is a stop-hash.
        let kept_runs = || {
            sorted
                .chunk_by(|a, b| a.0 == b.0)
                .filter(|run| run.len() as u64 <= u64::from(max_postings))
        };
        // Counted first so the three arrays are allocated once, exactly.
        let (n_hashes, n_anchors) = kept_runs().fold((0, 0), |(hashes, anchors), run| {
            (hashes + 1, anchors + run.len())
        });
        if n_hashes == 0 {
            return None;
        }

        let mut hashes = Vec::with_capacity(n_hashes);
        let mut starts = Vec::with_capacity(n_hashes + 1);
        let mut anchors = Vec::with_capacity(n_anchors);
        for run in kept_runs() {
            hashes.push(run[0].0);
            starts.push(anchors.len() as u32);
            anchors.extend(run.iter().map(|&(_, t_anchor)| t_anchor));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retain_gives_a_shrunken_list_its_heap_block_back() {
        let mut list: Postings<(u32, u32)> = Postings::default();
        for i in 0..100 {
            list.push((i, i));
        }
        let peak = list.heap_bytes();
        assert!(peak >= 100 * 8);

        // Left more than half unused: trimmed.
        list.retain(|&(r, _)| r < 40);
        assert_eq!(list.len(), 40);
        assert!(list.heap_bytes() < peak / 2, "{}", list.heap_bytes());

        // One survivor, then none: back to the inline forms.
        list.retain(|&(r, _)| r == 7);
        assert!(matches!(list, Postings::One((7, 7))));
        assert_eq!(list.heap_bytes(), 0);
        list.retain(|_| false);
        assert!(matches!(list, Postings::Empty));

        // A list that stays more than half full keeps its block, and the
        // survivors keep their order.
        let mut list: Postings<u32> = Postings::default();
        for i in 0..8 {
            list.push(i);
        }
        let block = list.heap_bytes();
        list.retain(|&p| p != 3);
        assert_eq!(list.as_slice(), &[0, 1, 2, 4, 5, 6, 7]);
        assert_eq!(list.heap_bytes(), block);
    }

    #[test]
    fn key_filter_never_misses_and_passes_few_others() {
        let mut x: u32 = 0x1234_5678;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        for n in [0_usize, 1, 5, 1_000, 50_000] {
            let keys: Vec<u32> = (0..n).map(|_| next()).collect();
            let filter = KeyFilter::new(keys.iter().copied(), 16);
            assert!(keys.iter().all(|&key| filter.may_contain(key)), "n = {n}");
            let strangers = (0..20_000).filter(|_| filter.may_contain(next())).count();
            assert!(
                strangers < 20_000 / 8,
                "n = {n}: {strangers} of 20000 passed"
            );
        }
    }

    /// Indexing only the hashes a query carries must answer every lookup of
    /// those hashes exactly as the full index does, stop-hashes included.
    #[test]
    fn filtered_build_answers_wanted_lookups_like_the_full_one() {
        use crate::classical::WangHash;

        let mut x: u32 = 0x9E37_79B9;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        for max_postings in [0_u32, 1, 3, u32::MAX] {
            let reference: Vec<WangHash> = (0..2_000)
                .map(|_| WangHash {
                    hash: next() % 600,
                    t_anchor: next() % 5_000,
                })
                .collect();
            let wanted_keys: Vec<u32> = (0..40).map(|_| next() % 700).collect();
            let wanted = KeyFilter::new(wanted_keys.iter().copied(), 16);

            let full = SortedPostings::build_from_wang(&reference, max_postings, None);
            let filtered = SortedPostings::build_from_wang(&reference, max_postings, Some(&wanted));
            let get = |index: &Option<SortedPostings>, hash| {
                index.as_ref().map_or(Vec::new(), |i| i.get(hash).to_vec())
            };
            for &key in &wanted_keys {
                assert_eq!(get(&filtered, key), get(&full, key), "hash {key}");
            }
            if let (Some(full), Some(filtered)) = (&full, &filtered) {
                assert!(filtered.len() <= full.len());
            }
        }
    }

    #[test]
    fn build_allocates_each_array_exactly() {
        let pairs: Vec<(u32, u32)> = (0..500_u32).map(|i| (i % 50, i)).collect();
        let sp = SortedPostings::build(&pairs, 100).unwrap();
        assert_eq!(sp.hashes.capacity(), 50);
        assert_eq!(sp.starts.capacity(), 51);
        assert_eq!(sp.anchors.capacity(), 500);
    }

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

    #[test]
    fn build_matches_naive_reference_on_random_input() {
        // Regression test for the in-place-sort build path: compare
        // against a naive BTreeMap grouping (the obvious reference
        // semantics) across randomized inputs, duplicate hashes,
        // duplicate pairs, and several stop-hash thresholds.
        let mut x: u32 = 0x9E37_79B9;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };

        for max_postings in [0u32, 1, 2, 3, 10, u32::MAX] {
            for n in [1usize, 2, 7, 64, 1000] {
                let pairs: Vec<(u32, u32)> = (0..n)
                    .map(|_| {
                        // Small hash space forces collisions and runs.
                        (next() % 20, next() % 500)
                    })
                    .collect();

                // Naive reference: group by hash, sort anchors, drop
                // stop-hashes.
                let mut groups: alloc::collections::BTreeMap<u32, Vec<u32>> =
                    alloc::collections::BTreeMap::new();
                for &(h, t) in &pairs {
                    groups.entry(h).or_default().push(t);
                }
                let expected: Vec<(u32, Vec<u32>)> = groups
                    .into_iter()
                    .filter(|(_, v)| v.len() as u32 <= max_postings)
                    .map(|(h, mut v)| {
                        v.sort_unstable();
                        (h, v)
                    })
                    .collect();

                let sp = SortedPostings::build(&pairs, max_postings);
                if expected.is_empty() {
                    assert!(sp.is_none(), "n={n} max_postings={max_postings}");
                    continue;
                }
                let sp = sp.expect("non-empty reference must build");
                assert_eq!(
                    sp.num_hashes(),
                    expected.len(),
                    "n={n} max_postings={max_postings}"
                );
                for (h, anchors) in &expected {
                    assert_eq!(
                        sp.get(*h),
                        anchors.as_slice(),
                        "hash={h} n={n} max_postings={max_postings}"
                    );
                }
            }
        }
    }

    /// `Postings` must behave exactly like the `Vec` it replaced under any
    /// interleaving of push and retain (order, contents, emptiness).
    #[test]
    fn postings_matches_vec_under_random_push_retain() {
        let mut x: u32 = 0xA11CE;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        for _ in 0..2000 {
            let mut p: Postings<(u32, u32)> = Postings::default();
            let mut v: Vec<(u32, u32)> = Vec::new();
            for _ in 0..(next() % 12) {
                if next() % 3 == 0 {
                    let kill = next() % 4;
                    p.retain(|&(r, _)| r != kill);
                    v.retain(|&(r, _)| r != kill);
                } else {
                    let e = (next() % 4, next());
                    p.push(e);
                    v.push(e);
                }
                assert_eq!(p.as_slice(), v.as_slice());
                assert_eq!(p.len(), v.len());
                assert_eq!(p.is_empty(), v.is_empty());
                assert_eq!((&p).into_iter().copied().collect::<Vec<_>>(), v);
            }
        }
        // A singleton owns no heap; a spilled list reports its capacity.
        let mut p: Postings<(u32, u32)> = Postings::default();
        p.push((1, 2));
        assert_eq!(p.heap_bytes(), 0);
        p.push((3, 4));
        assert!(p.heap_bytes() >= 2 * core::mem::size_of::<(u32, u32)>());
    }
}
