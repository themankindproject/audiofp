//! 1:N matching functions and optional transient in-memory indexes.
//!
//! All functions operate on `&[Fingerprint]` slices. The optional
//! `WangIndex`/`HaitsmaIndex`/`PanakoIndex` are matching
//! *accelerators* that hold a combined inverted index in RAM for the
//! lifetime of the struct. They are never serialised, never persisted,
//! and carry no file handles.
//!
//! # Parallel / large 1:N
//!
//! [`match_best`] and [`match_ranked`] are sequential pairwise loops.
//! With the `rayon` feature, [`par_match_best`] and [`par_match_ranked`]
//! evaluate every reference in parallel and return **deterministically
//! identical** results to their sequential counterparts (ties resolve to
//! the lowest reference id; ranking order is preserved). The `rayon`
//! feature also parallelises batch *fingerprinting*
//! (`fingerprint_batch_parallel`). For very large catalogs, prefer the
//! in-memory index types ([`WangIndex`], [`HaitsmaIndex`],
//! [`PanakoIndex`]) which amortise one inverted-index build across many
//! queries.

use crate::matching::{MatchResult, Matcher};

use alloc::vec::Vec;

use crate::matching::maps::{HashMap, Postings, hashmap_new};

/// Hard cap on the **total** votes a single [`WangIndex::query`] or
/// [`PanakoIndex::query`] may accumulate across all references (audit
/// §4.1 #6, 67-6).
///
/// One flat-list entry is 16 bytes, so the vote list is bounded at 160 MB,
/// and its capacity is clamped to this value so it cannot pre-allocate past
/// it either; the in-place unstable sort adds no buffer. A legitimate
/// query — a few thousand hashes, each hitting stop-hash-pruned postings —
/// lands orders of magnitude below it. (A separate per-reference cap of the
/// same size used to exist; with the per-query cap equal to it, it could
/// never bind.)
const MAX_VOTES_PER_QUERY: usize = 10_000_000;

/// Find the single best-matching reference.
///
/// Returns `Some((index, result))` if any reference clears its
/// decision threshold, or `None` if no reference matched.
///
/// Scans every reference and keeps the highest-ranked `is_match` result
/// (by [`match_result_compare_desc`](crate::matching::match_result_compare_desc)
/// — score desc, then prominence desc; a tie keeps the incumbent, i.e.
/// the lowest reference id, because the scan is ascending). There is
/// **no** early exit on a perfect score: a later reference can also score
/// `1.0` with a higher prominence, so terminating early would return a
/// different answer than [`match_ranked`] /
/// [`par_match_best`](https://docs.rs/audiofp/latest/audiofp/matching/fn.par_match_best.html)
/// (requires `rayon`). For the full ranking of every reference use
/// [`match_ranked`]; for large
/// catalogs prefer the index types.
///
/// # Performance
///
/// `O(N × match_one_cost)` — every reference is scored independently.
#[must_use]
pub fn match_best<M: Matcher>(
    matcher: &M,
    query: &M::Fingerprint,
    refs: &[M::Fingerprint],
) -> Option<(usize, MatchResult)> {
    let mut best: Option<(usize, MatchResult)> = None;
    for (i, r) in refs.iter().enumerate() {
        let result = matcher.match_one(query, r);
        if !result.is_match {
            continue;
        }
        track_best(&mut best, i, result);
    }
    best
}

/// Score every reference and return all results sorted by descending
/// score (ties broken by descending prominence).
///
/// Empty reference list → empty result. Unmatched references appear
/// with `is_match == false` — the caller can inspect their scores.
///
/// For large 1:N catalogs, use the in-memory index types
/// ([`WangIndex`], [`HaitsmaIndex`], [`PanakoIndex`]) which amortise
/// the inverted-index build across queries. Pairwise [`match_ranked`]
/// always evaluates every reference.
#[must_use]
pub fn match_ranked<M: Matcher>(
    matcher: &M,
    query: &M::Fingerprint,
    refs: &[M::Fingerprint],
) -> Vec<(usize, MatchResult)> {
    let mut results: Vec<(usize, MatchResult)> = refs
        .iter()
        .enumerate()
        .map(|(i, r)| (i, matcher.match_one(query, r)))
        .collect();
    results.sort_by(|a, b| crate::matching::match_result_compare_desc(&a.1, &b.1));
    results
}

/// Rayon-parallel [`match_best`] (requires the `rayon` feature, audit
/// C5).
///
/// Every reference is scored in parallel, then the best match is picked
/// deterministically: higher score, then higher prominence, then the
/// lowest reference id — identical to the sequential scan (which keeps
/// the first-encountered winner on ties, i.e. the lowest id, since it
/// iterates ascending). Like the sequential scan, this evaluates every
/// reference (no perfect-score early exit), so the two are
/// element-for-element equivalent.
///
/// # Panics
///
/// Panics if the rayon thread pool panics while scoring references.
#[cfg(feature = "rayon")]
#[must_use]
pub fn par_match_best<M>(
    matcher: &M,
    query: &M::Fingerprint,
    refs: &[M::Fingerprint],
) -> Option<(usize, MatchResult)>
where
    M: Matcher + Sync,
    M::Fingerprint: Sync,
{
    use rayon::prelude::*;

    refs.par_iter()
        .enumerate()
        .map(|(i, r)| (i, matcher.match_one(query, r)))
        .filter(|(_, res)| res.is_match)
        .min_by(|(ia, a), (ib, b)| {
            crate::matching::match_result_compare_desc(a, b).then_with(|| ia.cmp(ib))
        })
}

/// Rayon-parallel [`match_ranked`] (requires the `rayon` feature, audit
/// C5).
///
/// Every reference is scored in parallel and the results are returned
/// sorted by descending score (ties broken by descending prominence).
/// `rayon`'s ordered collect preserves the reference order, and the
/// stable sort keeps ties in reference order, so the output is
/// element-for-element identical to the sequential [`match_ranked`].
///
/// # Panics
///
/// Panics if the rayon thread pool panics while scoring references.
#[cfg(feature = "rayon")]
#[must_use]
pub fn par_match_ranked<M>(
    matcher: &M,
    query: &M::Fingerprint,
    refs: &[M::Fingerprint],
) -> Vec<(usize, MatchResult)>
where
    M: Matcher + Sync,
    M::Fingerprint: Sync,
{
    use rayon::prelude::*;

    let mut results: Vec<(usize, MatchResult)> = refs
        .par_iter()
        .enumerate()
        .map(|(i, r)| (i, matcher.match_one(query, r)))
        .collect();
    results.sort_by(|a, b| crate::matching::match_result_compare_desc(&a.1, &b.1));
    results
}

/// Approximate per-node overhead (bytes) of the `no_std` `BTreeMap`
/// fallback, per entry. Used only by `estimated_bytes` diagnostics.
#[cfg(not(feature = "std"))]
const BTREEMAP_NODE_OVERHEAD: usize = 48;

/// Heap bytes of `map`'s own table (not the values' heap), for
/// `estimated_bytes`.
///
/// On `std` the map is a SwissTable: `capacity()` is 7/8 of a power-of-two
/// bucket count, and each bucket holds one `(K, V)` plus one control byte,
/// with one SIMD group of trailing control bytes. Reconstructing that
/// layout matches the allocation the table really makes; the previous flat
/// "40 bytes of overhead per slot" overstated an index by 1.6–2.2×. The
/// `no_std` `BTreeMap` fallback has no capacity, so entries are counted
/// with a per-node overhead estimate.
fn map_heap_bytes<K, V>(map: &HashMap<K, V>) -> usize {
    #[cfg(feature = "std")]
    {
        let capacity = map.capacity();
        if capacity == 0 {
            return 0;
        }
        let buckets = (capacity * 8 / 7).next_power_of_two();
        buckets * (size_of::<(K, V)>() + 1) + 16
    }
    #[cfg(not(feature = "std"))]
    {
        map.len() * (size_of::<K>() + size_of::<V>() + BTREEMAP_NODE_OVERHEAD)
    }
}

/// Assign a reference id for `insert`: reuse the most-recently-vacated slot
/// (LIFO) when one exists so steady enroll/erase churn does not grow the id
/// space, else append. Returns the id as `u32` for posting storage.
///
/// # Panics
///
/// Panics if the catalog already holds `u32::MAX` live references (same
/// bound [`WangIndex::build`] enforces).
#[inline]
fn assign_ref_id(fps_len: usize, vacant: &mut Vec<u32>) -> (usize, u32) {
    if let Some(id) = vacant.pop() {
        (id as usize, id)
    } else {
        assert!(
            fps_len < u32::MAX as usize,
            "reference count exceeds u32::MAX"
        );
        (fps_len, fps_len as u32)
    }
}

/// Track a candidate result into `best`, keeping the higher-ranked one.
///
/// Ranking is [`match_result_compare_desc`](crate::matching::match_result_compare_desc)
/// — score desc, then prominence desc — so a perfect score does **not**
/// terminate the scan: a later reference can share `score == 1.0` with a
/// higher prominence and must win. Callers therefore always evaluate every
/// candidate, which is what makes `match_best`, `match_ranked`, and
/// `par_match_best` agree.
///
/// [`match_result_compare_desc`]: crate::matching::match_result_compare_desc
#[inline]
fn track_best(best: &mut Option<(usize, MatchResult)>, ref_id: usize, result: MatchResult) {
    let better = match best {
        None => true,
        Some((_, b)) => {
            crate::matching::match_result_compare_desc(&result, b) == core::cmp::Ordering::Less
        }
    };
    if better {
        *best = Some((ref_id, result));
    }
}

/// Stop-list every key whose posting list exceeds `limit` (TF-IDF-style
/// stop-hash removal, shared by the three `build`s): drop it from `map` and
/// return the tombstones, so a later `insert` cannot resurrect it with a
/// short list.
fn stop_list<P: Copy>(map: &mut HashMap<u32, Postings<P>>, limit: u32) -> HashMap<u32, ()> {
    let mut stopped = hashmap_new();
    map.retain(|key, list| {
        let keep = (list.len() as u32) <= limit;
        if !keep {
            stopped.insert(*key, ());
        }
        keep
    });
    stopped
}

/// Append one reference's `(key, posting)` pairs, skipping stop-listed
/// keys, then stop-list the keys these appends pushed over `limit` (shared
/// by the three `insert`s; `touched` is the caller's reused scratch).
///
/// Single-lookup insert: `entry` hashes once, and stop-hash bookkeeping
/// rides on the post-push length while the list is already borrowed:
///
/// - Under a fixed limit every pre-existing list satisfies the policy
///   (build retained, prior inserts pruned), and appends only grow. So a
///   list that violates the policy at the end of this insert must have
///   crossed the limit during one of its pushes (lengths grow 1 by 1),
///   exactly when it is held here.
/// - The key is recorded each time a post-push length exceeds the limit;
///   re-recording is harmless (pruning is idempotent). In the common case
///   nothing exceeds it, `touched` stays empty, and the prune pass visits
///   zero keys.
fn insert_postings<P: Copy>(
    map: &mut HashMap<u32, Postings<P>>,
    stopped: &mut HashMap<u32, ()>,
    touched: &mut Vec<u32>,
    postings: impl ExactSizeIterator<Item = (u32, P)>,
    limit: u32,
) {
    touched.clear();
    touched.reserve(postings.len().min(1024));
    let limit = limit as usize;
    for (key, posting) in postings {
        if stopped.contains_key(&key) {
            continue;
        }
        let list = map.entry(key).or_default();
        list.push(posting);
        if list.len() > limit {
            touched.push(key);
        }
    }
    for key in touched.drain(..) {
        if map.get(&key).is_some_and(|list| list.len() > limit) {
            stopped.insert(key, ());
            map.remove(&key);
        }
    }
}

/// The live ids among `ref_ids`, sorted and deduplicated (out-of-range and
/// already-vacant ids are skipped). Shared by the three `remove_many`s.
fn live_ids(live: &[bool], ref_ids: &[usize]) -> Vec<u32> {
    let mut dead: Vec<u32> = ref_ids
        .iter()
        .filter(|&&id| live.get(id).copied().unwrap_or(false))
        .map(|&id| id as u32)
        .collect();
    dead.sort_unstable();
    dead.dedup();
    dead
}

/// Physically delete every posting `dead` selects, and drop the keys whose
/// lists that empties.
///
/// Dropping emptied keys is what keeps enroll/erase churn from growing the
/// map without bound (a catalog that only ever holds one live track used
/// to accumulate every key any past track had). `retain` visits every slot
/// once and never rehashes. Lists are pre-scanned read-only for a dead
/// posting first: a single-reference remove touches a small fraction of
/// lists, and the rest skip the write pass.
fn erase_postings<P: Copy>(map: &mut HashMap<u32, Postings<P>>, dead: impl Fn(&P) -> bool) {
    map.retain(|_, list| {
        if list.iter().any(&dead) {
            list.retain(|p| !dead(p));
        }
        !list.is_empty()
    });
}

/// An in-memory inverted index over several Wang fingerprints.
///
/// This is a matching **accelerator** — it combines many references into
/// one `hash → Vec<(ref_id, t_anchor)>` map so the per-query cost of
/// 1:N lookups is paid once. It is dropped with its owning scope and is
/// never serialised.
///
/// # Performance
///
/// - **Build cost:** `O(Σ hashes)` — one HashMap insertion per hash.
/// - **Query cost:** `O(Q × avg_postings + C)` where `Q` = query hash
///   count, `avg_postings` = mean posting list length, `C` = candidate
///   references that received any vote (scored individually).
/// - **Memory:** `8 bytes × total_hashes` of postings (after stop-hash
///   removal) plus HashMap overhead. Single-posting keys — the large
///   majority on real catalogs — are stored inline in the map slot with no
///   separate allocation.
///
/// For catalogs above ~10 000 tracks, raise `min_votes` / `min_score`
/// pre-filters or shard the index.
pub struct WangIndex {
    /// Inverted index: hash → list of (reference id, t_anchor).
    ///
    /// `ref_id` is stored as `u32` (8 bytes per posting instead of 16 for
    /// `(usize, u32)` on 64-bit). Reference counts above `u32::MAX` are
    /// rejected at build time.
    map: HashMap<u32, Postings<(u32, u32)>>,
    /// Frame rates are stored per-reference for offset conversion.
    fps: alloc::vec::Vec<f32>,
    /// Largest `t_anchor` per reference, stored so `query` can reproduce
    /// [`WangMatcher`](super::WangMatcher)'s dense offset-histogram span
    /// (`[-q_max, +r_max]`) when computing prominence. Without it the index
    /// would normalise over the observed vote span only and disagree with
    /// the 1:1 matcher on `is_match`.
    r_max: alloc::vec::Vec<u32>,
    /// Liveness bit parallel to `fps`: `false` after [`WangIndex::remove`].
    /// Bookkeeping for [`WangIndex::live_count`] and slot reuse (removal is
    /// physical, so query needs no guard).
    live: alloc::vec::Vec<bool>,
    /// LIFO stack of ids vacated by [`WangIndex::remove`], reused by
    /// [`WangIndex::insert`] so enroll/erase churn does not grow the id
    /// space. `u32` ids are stored directly to avoid `as` casts on reuse.
    vacant: alloc::vec::Vec<u32>,
    /// Live reference count (`live` set bits). Maintained incrementally so
    /// [`WangIndex::live_count`] is O(1).
    live_count: usize,
    /// Scratch for [`WangIndex::insert`]: keys observed over the stop-hash
    /// limit during the current insert (usually empty — the common case
    /// prunes zero keys). Reused across calls; empty outside `insert`.
    touched: alloc::vec::Vec<u32>,
    /// Stop-hash tombstones: keys pruned for exceeding the posting cap stay
    /// suppressed for the lifetime of this index (append-only fixed policy),
    /// **including after the references that pushed them over the cap are
    /// removed** — a later reference carrying that hash is not matched on
    /// it. Lowering `max_postings_per_hash` on a later insert stops only
    /// the keys that insert touches; raising it does not resurrect stopped
    /// keys (audit F05). Rebuild from the surviving references to reset.
    stopped: HashMap<u32, ()>,
}

impl WangIndex {
    /// Build an index from a slice of Wang fingerprints.
    ///
    /// Only hashes appearing in ≤ `max_postings_per_hash` references are
    /// kept (TF-IDF-style stop-hash removal applied globally).
    ///
    /// # Panics
    ///
    /// Panics if `refs.len()` exceeds `u32::MAX` (reference ids are stored
    /// as `u32`).
    pub fn build(refs: &[crate::classical::WangFingerprint], max_postings_per_hash: u32) -> Self {
        assert!(
            refs.len() <= u32::MAX as usize,
            "reference count exceeds u32::MAX"
        );
        let mut map: HashMap<u32, Postings<(u32, u32)>> = super::maps::hashmap_with_capacity(
            refs.iter().map(|r| r.hashes.len()).sum::<usize>() / 2,
        );
        let fps: Vec<f32> = refs.iter().map(|r| r.frames_per_sec).collect();
        let r_max: Vec<u32> = refs
            .iter()
            .map(|r| r.hashes.iter().map(|h| h.t_anchor).max().unwrap_or(0))
            .collect();

        for (ref_id, fp) in refs.iter().enumerate() {
            for h in &fp.hashes {
                map.entry(h.hash)
                    .or_default()
                    .push((ref_id as u32, h.t_anchor));
            }
        }

        let stopped = stop_list(&mut map, max_postings_per_hash);

        let live_count = refs.len();
        let live = alloc::vec![true; live_count];
        Self {
            map,
            fps,
            r_max,
            live,
            vacant: Vec::new(),
            live_count,
            touched: Vec::new(),
            stopped,
        }
    }

    /// Append one fingerprint to the catalog, returning its stable `ref_id`.
    ///
    /// Applies the same stop-hash policy as [`WangIndex::build`] to the
    /// posting lists this fingerprint touches: appending only grows lists, so
    /// pruning touched lists yields exactly the map state a full
    /// `build` + `retain` would produce for the same input sequence **with
    /// the same `max_postings_per_hash` on every call** (pinned by the
    /// `insert_matches_build_parity` test). Parity does not survive a
    /// changed limit — a lower one prunes only the keys this insert touches
    /// — or interleaved `remove` calls, after which stop-hash tombstones
    /// can outlive the references that caused them (see the `stopped`
    /// field); call [`WangIndex::build`] on the survivors to re-derive.
    ///
    /// Reuses the most-recently-vacated id when [`WangIndex::remove`] has
    /// freed one (documented LIFO reuse); otherwise appends. Ids of live
    /// references never change, so in-flight query results stay valid.
    ///
    /// # Performance
    ///
    /// `O(H)` amortised for `H` hashes: one map lookup per hash; the
    /// stop-hash prune pass is empty in the common case (over-limit keys
    /// are recorded inline while the list is borrowed — no second lookup).
    /// The map is pre-reserved and the over-limit scratch is reused across
    /// calls, so steady-state insertion performs no auxiliary allocation
    /// beyond posting pushes (and rare map growth).
    ///
    /// # Panics
    ///
    /// Panics if the catalog already holds `u32::MAX` live references (same
    /// bound [`WangIndex::build`] enforces).
    pub fn insert(
        &mut self,
        fp: &crate::classical::WangFingerprint,
        max_postings_per_hash: u32,
    ) -> usize {
        let (ref_id, rid) = assign_ref_id(self.fps.len(), &mut self.vacant);
        // Pre-reserve map slots: ~half the new hashes are unique (same
        // heuristic as `build`'s `/2` capacity guess). BTreeMap fallback
        // (no_std) has no `reserve` — reservation is `std`-only.
        #[cfg(feature = "std")]
        self.map.reserve(fp.hashes.len() / 2);
        insert_postings(
            &mut self.map,
            &mut self.stopped,
            &mut self.touched,
            fp.hashes.iter().map(|h| (h.hash, (rid, h.t_anchor))),
            max_postings_per_hash,
        );
        let t_max = fp.hashes.iter().map(|h| h.t_anchor).max().unwrap_or(0);
        if ref_id < self.fps.len() {
            // Vacated-slot reuse: fps/live/r_max already have entries.
            self.fps[ref_id] = fp.frames_per_sec;
            self.r_max[ref_id] = t_max;
            self.live[ref_id] = true;
        } else {
            debug_assert_eq!(ref_id, self.fps.len());
            self.fps.push(fp.frames_per_sec);
            self.r_max.push(t_max);
            self.live.push(true);
        }
        self.live_count += 1;
        ref_id
    }

    /// Erase all postings for `ref_id`. Returns `false` if the id is
    /// out of range or already vacant (no-op).
    ///
    /// The id is tombstoned — never implicitly reused except by a later
    /// [`WangIndex::insert`], which documents LIFO reuse — so ids of live
    /// references never shift under concurrent readers of query results.
    /// Removal is physical (postings are deleted from every list, not just
    /// flagged), so `query` needs no liveness guard and its throughput is
    /// bit-identical before and after removes.
    ///
    /// # Performance
    ///
    /// `O(P)` where `P` = total postings: one in-place `retain` scan per
    /// posting list (memmove-compressed, no allocation), with keys the
    /// removal empties dropped from the map (see [`WangIndex::len`]).
    ///
    /// Lists are pre-scanned read-only for the id first: a single-ref
    /// remove touches only a small fraction of lists, so the rest pay a
    /// vectorized read pass (no write traffic) instead of a read+write
    /// `retain`.
    pub fn remove(&mut self, ref_id: usize) -> bool {
        if ref_id >= self.live.len() || !self.live[ref_id] {
            return false;
        }
        let rid = ref_id as u32;
        erase_postings(&mut self.map, |&(r, _)| r == rid);
        self.live[ref_id] = false;
        self.fps[ref_id] = f32::NAN;
        self.r_max[ref_id] = 0;
        self.vacant.push(rid);
        self.live_count -= 1;
        true
    }

    /// Erase many references in a SINGLE map pass. Returns the number of ids
    /// actually erased (out-of-range / already-vacant ids are skipped).
    ///
    /// One `remove` call scans the whole map (`O(P)`); erasing `K` refs one
    /// by one costs `K × O(P)`. This batches them: each posting list is
    /// visited once and compacted once. Membership is a binary search over
    /// the sorted id list (cache-hot; sorting `K` ids is negligible against
    /// the map pass). Vacated ids are pushed in ascending order so the
    /// SMALLEST erased id is reused first by a later `insert` (documented;
    /// use single `remove` for strict LIFO).
    ///
    /// Prefer this over looped `remove` for takedown batches, retention
    /// purges, and catalog churn.
    pub fn remove_many(&mut self, ref_ids: &[usize]) -> usize {
        let dead = live_ids(&self.live, ref_ids);
        if dead.is_empty() {
            return 0;
        }
        erase_postings(&mut self.map, |&(r, _)| dead.binary_search(&r).is_ok());
        let n = dead.len();
        for &rid in &dead {
            let id = rid as usize;
            self.live[id] = false;
            self.fps[id] = f32::NAN;
            self.r_max[id] = 0;
        }
        // Ascending push order → smallest id popped first by `insert`.
        for &rid in dead.iter().rev() {
            self.vacant.push(rid);
        }
        self.live_count -= n;
        n
    }

    /// Append many fingerprints, returning their stable `ref_id`s in order.
    ///
    /// A loop of `insert` re-reserves map capacity per fingerprint and grows
    /// the table repeatedly; this reserves once for the whole batch
    /// (`Σ hashes / 2`) and then runs the same single-lookup push loop per
    /// fingerprint, so bulk ingest (the #119 parallel-extract → serial-ingest
    /// workflow) pays table growth once. Per-fingerprint semantics —
    /// stop-hash parity, slot reuse, id stability — are identical to
    /// [`WangIndex::insert`].
    pub fn insert_many(
        &mut self,
        fps: &[crate::classical::WangFingerprint],
        max_postings_per_hash: u32,
    ) -> Vec<usize> {
        #[cfg(feature = "std")]
        self.map
            .reserve(fps.iter().map(|f| f.hashes.len()).sum::<usize>() / 2);
        let mut ids = Vec::with_capacity(fps.len());
        for fp in fps {
            ids.push(self.insert(fp, max_postings_per_hash));
        }
        ids
    }

    /// Live reference count (excludes ids erased by [`WangIndex::remove`]).
    /// O(1) — maintained incrementally.
    ///
    /// Note [`WangIndex::len`] keeps its historical meaning (unique hash key
    /// count); use this for the catalog track count.
    #[must_use]
    pub fn live_count(&self) -> usize {
        self.live_count
    }

    /// Estimated P(same recording | evidence) for a [`WangIndex::query`]
    /// result: same `wang-v1` map as [`WangMatcher::calibrated_score`](super::WangMatcher::calibrated_score).
    ///
    /// The index computes prominence on a sparse per-reference tally while
    /// the matcher uses a dense consolidated histogram, so the two paths
    /// can differ marginally on `is_match` near the boundary (documented
    /// at `query`) — the calibrated value of an index-only result carries
    /// that selection bias. Raw fields untouched.
    #[must_use]
    pub fn calibrated_score(&self, r: &super::MatchResult) -> f32 {
        super::calibration::calibrated_wang(r)
    }

    /// Measured heap footprint in bytes: posting lists (capacity × 8 bytes
    /// per `(u32, u32)` posting) + map slots (capacity × key size + slot
    /// overhead) + `fps` / `live` / `vacant` / `touched` storage.
    /// O(map size) — call rarely (capacity planning, shard-split decisions),
    /// not per query.
    ///
    /// The hash tables (including the stop-hash tombstones) are sized from
    /// their real layout, so the estimate tracks the heap to within allocator
    /// rounding on `std`; the `no_std` `BTreeMap` term is coarser. Use for
    /// alerting and sharding, not billing.
    #[must_use]
    pub fn estimated_bytes(&self) -> usize {
        // The table bytes already hold one key and one value per bucket, so
        // there is no separate per-key term (audit §4.2 #10).
        let mut bytes = map_heap_bytes(&self.map) + map_heap_bytes(&self.stopped);
        for list in self.map.values() {
            bytes += list.heap_bytes();
        }
        bytes += self.fps.capacity() * size_of::<f32>();
        bytes += self.r_max.capacity() * size_of::<u32>();
        bytes += self.live.capacity() * size_of::<bool>();
        bytes += self.vacant.capacity() * size_of::<u32>();
        bytes += self.touched.capacity() * size_of::<u32>();
        bytes
    }

    /// Query the index, returning the single best-matching reference.
    ///
    /// Votes are tallied **per reference** (offset `δ = t_ref − t_query`,
    /// matching [`WangMatcher`](super::WangMatcher)), then each candidate
    /// reference is scored independently and the best match that clears
    /// the thresholds is returned as `(ref_id, result)`.
    ///
    /// This is an accelerator: it shares one inverted-index lookup across
    /// all references. Its prominence/score are computed on the sparse
    /// per-reference tally, so they can differ marginally from a direct
    /// [`WangMatcher::match_one`](super::WangMatcher::match_one) call
    /// (which uses a dense, box-consolidated histogram). For exact 1:1
    /// scores use [`match_ranked`].
    #[must_use]
    pub fn query(
        &self,
        query: &crate::classical::WangFingerprint,
        cfg: &crate::matching::WangMatchConfig,
    ) -> Option<(usize, MatchResult)> {
        use crate::matching::{TimeOffset, clamp_score};

        if query.hashes.is_empty() || self.map.is_empty() {
            return None;
        }

        // Votes as one flat `(ref_id, offset δ, query-hash index)` list,
        // sorted by `(ref_id, δ)`, instead of a `HashMap<u32, Vec<_>>`.
        //
        // The map allocated one `Vec` per candidate reference (≈ one
        // allocation per reference in the catalog per query — the dominant
        // remaining per-query allocation, audit §4.2 #8), plus its own table.
        // A flat list is one allocation, is contiguous for the scoring pass,
        // and needs no hashing. Sorting by offset too makes each reference's
        // per-offset counts a run-length pass and its contributor window a
        // binary search, so scoring needs no per-candidate map either.
        //
        // Pre-size to the common case (each query hash hits a small handful
        // of postings) so the list does not pay repeated doubling
        // reallocations. An underestimate simply grows as usual.
        let mut flat: Vec<(u32, i64, u32)> = Vec::with_capacity(
            query
                .hashes
                .len()
                .saturating_mul(2)
                .min(MAX_VOTES_PER_QUERY),
        );
        'outer: for (qi, h) in query.hashes.iter().enumerate() {
            let q_t = h.t_anchor as i64;
            if let Some(list) = self.map.get(&h.hash) {
                for &(ref_id, tr) in list {
                    if flat.len() >= MAX_VOTES_PER_QUERY {
                        break 'outer;
                    }
                    if flat.len() == flat.capacity() {
                        let capacity = flat
                            .capacity()
                            .saturating_mul(2)
                            .clamp(4, MAX_VOTES_PER_QUERY);
                        flat.reserve_exact(capacity - flat.len());
                    }
                    flat.push((ref_id, tr as i64 - q_t, qi as u32));
                }
            }
        }
        // Unstable: the outputs below (bin counts, the contributor set)
        // do not depend on order within a `(ref_id, δ)` key, and the
        // in-place sort needs no `n/2` scratch buffer (24 MB at 1M votes).
        flat.sort_unstable_by_key(|&(r, d, _)| (r, d));

        let tol = cfg.offset_tolerance_frames as i64;
        let q_len = query.hashes.len().max(1) as f32;
        let mut best: Option<(usize, MatchResult)> = None;

        // Loop-invariant: the query's largest anchor frame. The prominence
        // below reproduces `WangMatcher`'s dense `[-q_max, +r_max]` span, so
        // this depends only on the query — it used to be recomputed inside
        // the per-candidate loop (an O(Q) scan per candidate).
        let q_max = query.hashes.iter().map(|h| h.t_anchor).max().unwrap_or(0);

        // Reusable scratch for the per-candidate scoring steps (audit §4.2
        // #8): `clear()` retains capacity, so after the first few candidates
        // these stop allocating entirely.
        let mut bin_vec: Vec<(i64, u32)> = Vec::new();
        let mut contrib_indices: Vec<u32> = Vec::new();

        // Deterministic candidate order: walk the flat list in ascending
        // `ref_id` order (guaranteed by the sort above), not in HashMap
        // order, so exact (score, prominence) ties and the perfect-score
        // early-exit always resolve to the lowest reference id regardless of
        // hasher state (audit 67-1 follow-up).
        let mut group_start = 0usize;
        while group_start < flat.len() {
            let ref_id = flat[group_start].0;
            let mut group_end = group_start + 1;
            while group_end < flat.len() && flat[group_end].0 == ref_id {
                group_end += 1;
            }
            let votes = &flat[group_start..group_end];
            group_start = group_end;
            // `matching/mod.rs` documents that a frame-rate mismatch returns
            // `MatchResult::NONE` in all builds. The index stores per-reference
            // rates, so enforce the same rule per candidate — otherwise the
            // `offset.ms` below is converted with the reference rate while the
            // query claims a different one.
            let fps = self.fps.get(ref_id as usize).copied().unwrap_or(62.5);
            if !crate::matching::frames_per_sec_compatible(query.frames_per_sec, fps) {
                continue;
            }
            // Quick pre-filter: if the total raw vote count for this
            // reference is below min_votes, the consolidated peak can
            // never reach the threshold either (consolidation can only
            // redistribute votes, not create them). Skip the expensive
            // binning + consolidation entirely.
            if (votes.len() as u32) < cfg.min_votes {
                continue;
            }

            // Raw per-offset bin counts (sparse), in offset order — a
            // run-length pass over the δ-sorted votes, so every downstream
            // step (peak search, prominence, plateau selection) is
            // independent of hasher state (audit 67-1).
            bin_vec.clear();
            for &(_, d, _) in votes {
                match bin_vec.last_mut() {
                    Some((last, n)) if *last == d => *n += 1,
                    _ => bin_vec.push((d, 1)),
                }
            }

            // Box-consolidate over every integer centre in the dense span
            // `[-q_max, +r_max]`, including unoccupied offsets (audit F03).
            let r_max = self.r_max.get(ref_id as usize).copied().unwrap_or(0);
            let dmin = -(q_max as i64);
            let dmax = r_max as i64;
            let (peak_votes, peak_off, mode_off, dense_total) =
                super::consolidate::sparse_wang_box_peak(&bin_vec, tol, dmin, dmax);

            if peak_votes < cfg.min_votes {
                continue;
            }

            // Dense-range prominence within the sparse index's full span:
            // background mean divides by the full `[-q_max, +r_max]` width,
            // not just occupied-bin centres. Unlike the dense matcher, the
            // sparse index need not truncate spans beyond 10M bins. Retaining
            // those matches preserves the historical long-span index behavior.
            let dense_bins = (q_max as u64 + r_max as u64 + 1) as f32;
            let sum_rest: u64 = dense_total - peak_votes as u64;
            let mean_rest = sum_rest as f32 / (dense_bins - 1.0).max(1.0);
            let prominence = peak_votes as f32 / (mean_rest + 1.0);
            if super::below_threshold(prominence, cfg.min_prominence) {
                continue;
            }

            // Contrib count: distinct query-hash indices near peak — the
            // δ-sorted votes within `peak_off ± tol` are one contiguous run.
            let lo = votes.partition_point(|&(_, d, _)| d < peak_off - tol);
            let hi = votes.partition_point(|&(_, d, _)| d <= peak_off + tol);
            contrib_indices.clear();
            contrib_indices.extend(votes[lo..hi].iter().map(|&(_, _, qi)| qi));
            contrib_indices.sort_unstable();
            contrib_indices.dedup();
            let score = clamp_score(contrib_indices.len() as f32 / q_len);
            if super::below_threshold(score, cfg.min_score) {
                continue;
            }

            // fps was fetched and compatibility-checked above; the 62.5
            // fallback is defensive only.
            let result = MatchResult {
                is_match: true,
                score,
                votes: peak_votes,
                prominence,
                offset: TimeOffset::from_frames(mode_off, fps),
                time_scale: 1.0,
            };

            track_best(&mut best, ref_id as usize, result);
        }

        best
    }

    /// Return the number of unique hash keys in the index.
    ///
    /// This is **key-space**, not catalog size: the number of hashes with at
    /// least one live posting (keys [`WangIndex::remove`] empties are dropped).
    /// Use [`WangIndex::live_count`] for the number of references.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Return `true` if the index holds **no hash keys at all**.
    ///
    /// Key-space, like [`WangIndex::len`]. A catalog whose references all
    /// had their hashes stop-listed is non-empty yet has no keys; to test
    /// whether any reference is enrolled, use [`WangIndex::is_empty_catalog`].
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Return `true` if the catalog holds no live references.
    ///
    /// This is the "no tracks enrolled" test; unlike
    /// [`WangIndex::is_empty`] it tracks removals.
    #[must_use]
    pub fn is_empty_catalog(&self) -> bool {
        self.live_count == 0
    }
}

/// An in-memory inverted index over several Haitsma fingerprints.
///
/// Uses the same sub-fingerprint LUT strategy as
/// [`HaitsmaMatcher`](super::HaitsmaMatcher): build `u32 → Vec<(ref_id,
/// frame_pos)>` per frame, probe each query frame to discover candidate
/// alignments, then verify the best per-reference BER.
///
/// # Performance
///
/// - **Build:** `O(Σ frames)` — one LUT insertion per frame across all refs.
/// - **Query:** `O(Q + C × overlap)` where `Q` = query frames probed,
///   `C` = candidate refs with LUT hits, `overlap` = BER verification
///   window.
/// - **Memory:** `8 bytes × total_frames` (LUT) plus `4 bytes × total_frames`
///   (per-ref frame clone for BER verification).
///
/// # Scoring note
///
/// Only **exact** sub-fingerprint matches are probed (no bit-flip
/// neighbours). This is faster than the `probe_bit_flips` option on
/// [`HaitsmaMatcher`](super::HaitsmaMatcher) but may miss weaker matches.
/// Prominence uses `0.5 / BER` — not directly comparable to
/// `HaitsmaMatcher`'s `median_BER / (BER + ε)` formula.
pub struct HaitsmaIndex {
    /// Inverted index: 32-bit sub-fingerprint → list of (ref_id, frame_pos).
    ///
    /// `ref_id` is stored as `u32` (8 bytes per posting instead of 16 for
    /// `(usize, u32)` on 64-bit).
    lut: HashMap<u32, Postings<(u32, u32)>>,
    /// Per-reference frame slices for BER verification.
    frames: Vec<Vec<u32>>,
    /// Per-reference frame rates for offset conversion.
    fps: Vec<f32>,
    /// Liveness bit parallel to `fps`/`frames`. Query's LUT gather can still
    /// surface a dead id's postings only if `remove` left them — it does
    /// not (removal is physical), so this bit is bookkeeping for
    /// [`HaitsmaIndex::live_count`] and slot reuse, not a query filter.
    live: alloc::vec::Vec<bool>,
    /// LIFO stack of ids vacated by [`HaitsmaIndex::remove`].
    vacant: alloc::vec::Vec<u32>,
    /// Live reference count, maintained incrementally (O(1) read).
    live_count: usize,
    /// Reused touched-key scratch for [`HaitsmaIndex::insert`].
    touched: alloc::vec::Vec<u32>,
    /// Stop-hash tombstones (see [`WangIndex::stopped`]).
    stopped: HashMap<u32, ()>,
}

impl HaitsmaIndex {
    /// Build from a slice of Haitsma fingerprints.
    ///
    /// `max_postings_per_hash` caps the size of each sub-fingerprint's
    /// posting list. Hashes that appear in more than this many positions
    /// (silence / DC / highly repetitive content) are dropped entirely —
    /// the same TF-IDF-style stop-hash pruning used by [`WangIndex`] and
    /// [`PanakoIndex`]. This keeps query-time memory and work bounded on
    /// pathological catalogs (audit B7 / A1).
    ///
    /// # Panics
    ///
    /// Panics if `refs.len()` exceeds `u32::MAX` (reference ids are stored
    /// as `u32`).
    pub fn build(
        refs: &[crate::classical::HaitsmaFingerprint],
        max_postings_per_hash: u32,
    ) -> Self {
        assert!(
            refs.len() <= u32::MAX as usize,
            "reference count exceeds u32::MAX"
        );
        let mut lut: HashMap<u32, Postings<(u32, u32)>> = super::maps::hashmap_with_capacity(
            refs.iter().map(|r| r.frames.len()).sum::<usize>() / 2,
        );
        let frames: Vec<Vec<u32>> = refs.iter().map(|r| r.frames.clone()).collect();
        let fps: Vec<f32> = refs.iter().map(|r| r.frames_per_sec).collect();

        for (ref_id, fp) in refs.iter().enumerate() {
            for (pos, &frame) in fp.frames.iter().enumerate() {
                lut.entry(frame)
                    .or_default()
                    .push((ref_id as u32, pos as u32));
            }
        }

        // Silence/DC frames produce enormous posting lists; prune them so
        // memory and query time stay bounded.
        let stopped = stop_list(&mut lut, max_postings_per_hash);

        let live_count = refs.len();
        let live = alloc::vec![true; live_count];
        Self {
            lut,
            frames,
            fps,
            live,
            vacant: Vec::new(),
            live_count,
            touched: Vec::new(),
            stopped,
        }
    }

    /// Append one fingerprint to the catalog, returning its stable `ref_id`.
    ///
    /// Same stop-hash parity contract as [`WangIndex::insert`]: pruning the
    /// touched LUT lists yields exactly the map state a full `build` +
    /// `retain` would produce for the same append-only input sequence
    /// (pinned by the `insert_matches_build_parity` test; parity is over
    /// append-only histories — interleaved `remove` calls physically delete
    /// postings, which `build` has no equivalent for).
    ///
    /// The reference's full frame vector is cloned for BER verification
    /// (same as `build`); on vacated-slot reuse the old vector is replaced,
    /// freeing the previous allocation. Ids of live references never change.
    ///
    /// # Performance
    ///
    /// `O(F)` amortised for `F` frames: one LUT lookup per frame plus
    /// touched-list pruning. Pre-reserved map capacity and reused
    /// touched-key scratch keep steady-state insertion free of auxiliary
    /// allocation beyond posting pushes (and rare map growth).
    ///
    /// # Panics
    ///
    /// Panics if the catalog already holds `u32::MAX` live references (same
    /// bound [`HaitsmaIndex::build`] enforces).
    pub fn insert(
        &mut self,
        fp: &crate::classical::HaitsmaFingerprint,
        max_postings_per_hash: u32,
    ) -> usize {
        let (ref_id, rid) = assign_ref_id(self.fps.len(), &mut self.vacant);
        #[cfg(feature = "std")]
        self.lut.reserve(fp.frames.len() / 2);
        insert_postings(
            &mut self.lut,
            &mut self.stopped,
            &mut self.touched,
            fp.frames
                .iter()
                .enumerate()
                .map(|(pos, &frame)| (frame, (rid, pos as u32))),
            max_postings_per_hash,
        );
        if ref_id < self.frames.len() {
            self.frames[ref_id] = fp.frames.clone();
            self.fps[ref_id] = fp.frames_per_sec;
            self.live[ref_id] = true;
        } else {
            debug_assert_eq!(ref_id, self.frames.len());
            self.frames.push(fp.frames.clone());
            self.fps.push(fp.frames_per_sec);
            self.live.push(true);
        }
        self.live_count += 1;
        ref_id
    }

    /// Erase all postings for `ref_id`. Returns `false` if the id is
    /// out of range or already vacant (no-op).
    ///
    /// Removal is physical: postings are deleted from every LUT list **and**
    /// the reference's frame vector is freed immediately (replaced with an
    /// empty `Vec`), so the dominant `HaitsmaIndex` memory term — the
    /// per-ref frame clone (`4 bytes × frames`) — is returned on erase.
    /// `query` needs no liveness guard; throughput is bit-identical before
    /// and after removes. Ids of live references never shift.
    ///
    /// # Performance
    ///
    /// `O(F)` for a reference of `F` frames: its postings can only live
    /// under its own frame values (the LUT key *is* the frame), which the
    /// index already keeps for BER verification, so only those lists are
    /// visited — not the whole LUT. Keys the removal empties are dropped
    /// (see [`HaitsmaIndex::len`]).
    pub fn remove(&mut self, ref_id: usize) -> bool {
        if ref_id >= self.live.len() || !self.live[ref_id] {
            return false;
        }
        let rid = ref_id as u32;
        let frames = core::mem::take(&mut self.frames[ref_id]);
        Self::erase_keys(&mut self.lut, &frames, |&(r, _)| r == rid);
        self.live[ref_id] = false;
        self.fps[ref_id] = f32::NAN;
        self.vacant.push(rid);
        self.live_count -= 1;
        true
    }

    /// Delete the postings `dead` selects from the lists under `keys`,
    /// dropping keys that empties. A repeated key finds nothing left to
    /// delete on its second visit.
    fn erase_keys(
        lut: &mut HashMap<u32, Postings<(u32, u32)>>,
        keys: &[u32],
        dead: impl Fn(&(u32, u32)) -> bool,
    ) {
        for key in keys {
            if let Some(list) = lut.get_mut(key) {
                if list.iter().any(&dead) {
                    list.retain(|p| !dead(p));
                }
                if list.is_empty() {
                    lut.remove(key);
                }
            }
        }
    }

    /// Erase many references (plus their frame vectors). Returns the number
    /// of ids actually erased. Each reference's postings are found through
    /// its own frames, as in [`HaitsmaIndex::remove`], so the cost is the
    /// erased references' frame count, not the LUT size. Vacated ids are
    /// pushed ascending (smallest reused first — documented).
    pub fn remove_many(&mut self, ref_ids: &[usize]) -> usize {
        let dead = live_ids(&self.live, ref_ids);
        if dead.is_empty() {
            return 0;
        }
        let n = dead.len();
        for &rid in &dead {
            let id = rid as usize;
            let frames = core::mem::take(&mut self.frames[id]);
            Self::erase_keys(&mut self.lut, &frames, |&(r, _)| {
                dead.binary_search(&r).is_ok()
            });
            self.live[id] = false;
            self.fps[id] = f32::NAN;
        }
        for &rid in dead.iter().rev() {
            self.vacant.push(rid);
        }
        self.live_count -= n;
        n
    }

    /// Append many fingerprints, returning their stable `ref_id`s in order.
    /// Reserves LUT capacity once for the whole batch; per-fingerprint
    /// semantics identical to [`HaitsmaIndex::insert`].
    pub fn insert_many(
        &mut self,
        fps: &[crate::classical::HaitsmaFingerprint],
        max_postings_per_hash: u32,
    ) -> Vec<usize> {
        #[cfg(feature = "std")]
        self.lut
            .reserve(fps.iter().map(|f| f.frames.len()).sum::<usize>() / 2);
        let mut ids = Vec::with_capacity(fps.len());
        for fp in fps {
            ids.push(self.insert(fp, max_postings_per_hash));
        }
        ids
    }

    /// Live reference count (excludes ids erased by [`HaitsmaIndex::remove`]).
    /// O(1) — maintained incrementally.
    #[must_use]
    pub fn live_count(&self) -> usize {
        self.live_count
    }

    /// Estimated P(same recording | evidence) for a
    /// [`HaitsmaIndex::query`] result: same `haitsma-v1` map as
    /// [`HaitsmaMatcher::calibrated_score`](super::HaitsmaMatcher::calibrated_score).
    ///
    /// Both paths report `score = 1 − BER`; their prominence formulas
    /// differ (`0.5/BER` here vs `median_BER/(ber+ε)` in the matcher) —
    /// see `calibrated_haitsma` for the bounded-error discussion. Raw
    /// fields untouched.
    #[must_use]
    pub fn calibrated_score(&self, r: &super::MatchResult) -> f32 {
        super::calibration::calibrated_haitsma(r)
    }

    /// Measured heap footprint in bytes: LUT posting lists (capacity × 8
    /// bytes per `(u32, u32)` posting) + map slots + per-reference frame
    /// vectors (capacity × 4 bytes — the dominant term) + `fps` / `live` /
    /// `vacant` / `touched` storage and the stop-hash tombstones. O(map
    /// size) — call rarely, not per query. Tracks the heap to within
    /// allocator rounding on `std` (see [`WangIndex::estimated_bytes`]); for
    /// alerting and sharding, not billing.
    #[must_use]
    pub fn estimated_bytes(&self) -> usize {
        // The table bytes already hold one key and one value per bucket, so
        // there is no separate per-key term (audit §4.2 #10).
        let mut bytes = map_heap_bytes(&self.lut) + map_heap_bytes(&self.stopped);
        for list in self.lut.values() {
            bytes += list.heap_bytes();
        }
        for f in &self.frames {
            bytes += f.capacity() * size_of::<u32>();
        }
        bytes += self.frames.capacity() * size_of::<Vec<u32>>();
        bytes += self.fps.capacity() * size_of::<f32>();
        bytes += self.live.capacity() * size_of::<bool>();
        bytes += self.vacant.capacity() * size_of::<u32>();
        bytes += self.touched.capacity() * size_of::<u32>();
        bytes
    }

    /// Query the index, returning the best-matching `(ref_id, result)`.
    ///
    /// For each query frame, probes the LUT to gather candidate
    /// `(ref_id, delta)` pairs. Each candidate reference is then verified
    /// with the exact-BER path at up to the 8 most-hit candidate offsets
    /// (a repeated motif can concentrate hits at a wrong offset while the
    /// true alignment has the better BER).
    ///
    /// Only exact sub-fingerprint matches are probed (no bit-flips in
    /// the index path — use [`match_ranked`] with explicit
    /// `probe_bit_flips` when recall under codec distortion matters).
    #[must_use]
    pub fn query(
        &self,
        query: &crate::classical::HaitsmaFingerprint,
        cfg: &crate::matching::HaitsmaMatchConfig,
    ) -> Option<(usize, MatchResult)> {
        use super::haitsma::{BestAlignment, informative_frames, overlap_at};

        let q_frames = &query.frames;
        let q_len = q_frames.len();
        // A silent / constant / short-period query is not evidence of
        // identity (see `HaitsmaMatcher`); stop before probing the LUT.
        if q_len == 0 || self.lut.is_empty() || !informative_frames(q_frames) {
            return None;
        }

        let min_overlap = cfg.min_overlap_frames as usize;

        // 1. Gather candidate (ref_id, delta) hits from LUT probes as one
        //    flat list (capped like the other index queries), then sort:
        //    equal pairs become runs whose length is the hit count, and
        //    each reference's candidates become one contiguous group. This
        //    replaces a `(ref, delta) → hits` map plus a `ref → Vec` map
        //    (one allocation per candidate reference).
        let mut hits: Vec<(u32, i64)> = Vec::with_capacity(q_len.min(MAX_VOTES_PER_QUERY));

        'outer: for (q_pos, &q_frame) in q_frames.iter().enumerate() {
            if let Some(list) = self.lut.get(&q_frame) {
                for &(ref_id, r_pos) in list {
                    let delta = r_pos as i64 - q_pos as i64;
                    let overlap = overlap_at(q_len, self.frames[ref_id as usize].len(), delta);
                    if overlap >= min_overlap {
                        if hits.len() >= MAX_VOTES_PER_QUERY {
                            break 'outer;
                        }
                        hits.push((ref_id, delta));
                    }
                }
            }
        }

        if hits.is_empty() {
            return None;
        }
        hits.sort_unstable();
        let mut candidates: Vec<((u32, i64), u32)> = Vec::new();
        super::consolidate::count_sorted(&hits, &mut candidates);
        drop(hits);

        // 2. For each candidate reference, verify up to
        //    `MAX_DELTAS_PER_REF` best-hit alignments with exact BER and
        //    keep the best rate.
        //
        //    Verifying only the single most-hit delta biased the score:
        //    a repeated motif can concentrate LUT hits at a wrong offset
        //    while the true alignment (fewer bit-exact frames under
        //    codec noise) has a far better BER. Verification order is
        //    (hits desc, |δ| asc, δ asc) so it is independent of hash
        //    state, and references are visited in ascending id order for
        //    the same reason.
        const MAX_DELTAS_PER_REF: usize = 8;

        let mut best: Option<(usize, MatchResult)> = None;
        let mut deltas: Vec<(i64, u32)> = Vec::new();
        let mut group_start = 0usize;

        while group_start < candidates.len() {
            let ref_id = candidates[group_start].0.0;
            let mut group_end = group_start + 1;
            while group_end < candidates.len() && candidates[group_end].0.0 == ref_id {
                group_end += 1;
            }
            let group = &candidates[group_start..group_end];
            group_start = group_end;
            // The matcher contract (matching/mod.rs) requires both
            // fingerprints to share a frame rate; a mismatch returns
            // `MatchResult::NONE`. The index holds per-reference rates, so
            // enforce it per candidate rather than converting offsets with
            // the wrong rate.
            let ref_fps = self.fps[ref_id as usize];
            if !crate::matching::frames_per_sec_compatible(query.frames_per_sec, ref_fps) {
                continue;
            }
            // Same low-information screen `HaitsmaMatcher` applies to the
            // whole reference: a silent / constant / short-period reference
            // aligns with itself everywhere, so a near-zero BER against it
            // is not evidence. (Overlap-level screening below skips runs
            // shorter than 32 frames, so a short query used to "match" such
            // a reference with score 1.0.) The scan stops after a few
            // frames on real audio.
            if !informative_frames(&self.frames[ref_id as usize]) {
                continue;
            }
            deltas.clear();
            deltas.extend(group.iter().map(|&((_, delta), n)| (delta, n)));
            let order = |&(d1, h1): &(i64, u32), &(d2, h2): &(i64, u32)| {
                h2.cmp(&h1)
                    .then_with(|| d1.abs().cmp(&d2.abs()))
                    .then_with(|| d1.cmp(&d2))
            };
            if deltas.len() > MAX_DELTAS_PER_REF {
                deltas.select_nth_unstable_by(MAX_DELTAS_PER_REF, order);
                deltas.truncate(MAX_DELTAS_PER_REF);
            }
            deltas.sort_unstable_by(order);
            let r_len = self.frames[ref_id as usize].len();
            // Best-BER tracking with a rate-normalized early-abort bound
            // (same rationale as `HaitsmaMatcher`): a short-overlap
            // candidate must not suppress a longer better-rate one.
            let mut alignment = BestAlignment::new();
            for &(delta, _hits) in &deltas {
                alignment.consider(
                    q_frames,
                    &self.frames[ref_id as usize],
                    q_len,
                    r_len,
                    delta,
                    min_overlap,
                );
            }

            if !alignment.found() {
                continue;
            }

            let delta = alignment.delta;
            let overlap = alignment.overlap;
            let (ber, score, is_match) =
                super::haitsma::ber_score_match(alignment.hamming, overlap, cfg.max_ber);

            // Prominence: approximate — compare the winning BER against
            // what you'd expect from random alignment (~0.5).
            // ber ≈ 0 (perfect alignment) → sentinel 100.0; 1e-6 guards
            // against division by zero.
            let prominence = if ber > 1e-6 { 0.5 / ber } else { 100.0 };

            let offset = crate::matching::TimeOffset::from_frames(delta, self.fps[ref_id as usize]);
            let result = MatchResult {
                is_match,
                score,
                votes: overlap as u32,
                prominence,
                offset,
                time_scale: 1.0,
            };

            if !is_match {
                continue;
            }

            track_best(&mut best, ref_id as usize, result);
        }

        best
    }

    /// Return the number of unique sub-fingerprints (LUT keys) in the
    /// index.
    ///
    /// Key-space, not catalog size: sub-fingerprints with at least one live
    /// posting (keys [`HaitsmaIndex::remove`] empties are dropped). Use
    /// [`HaitsmaIndex::live_count`] for the reference count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lut.len()
    }

    /// Return `true` if the index holds **no LUT keys at all**.
    ///
    /// Key-space, like [`HaitsmaIndex::len`]; use
    /// [`HaitsmaIndex::is_empty_catalog`] for the liveness test.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lut.is_empty()
    }

    /// Return `true` if the catalog holds no live references.
    #[must_use]
    pub fn is_empty_catalog(&self) -> bool {
        self.live_count == 0
    }
}

/// An in-memory inverted index over several Panako fingerprints.
///
/// Uses the same 2-D Hough accumulator strategy as
/// [`PanakoMatcher`](super::PanakoMatcher) but across the combined
/// reference set for efficient 1:N lookups.
///
/// # Performance
///
/// - **Build:** `O(Σ hashes)` — one map insertion per triplet hash.
/// - **Query:** `O(Q × avg_postings + C)` where `C` = candidate refs
///   that received votes, each scored via Hough peak extraction.
/// - **Memory:** `16 bytes × total_hashes` (4 u32s per posting).
///
/// # Scoring note
///
/// RANSAC refinement is **not** applied (unlike `PanakoMatcher` with
/// `ransac_refine: true`). The coarse Hough peak determines the final
/// `time_scale` — less precise but much faster for large catalogs.
pub struct PanakoIndex {
    /// Inverted index: hash → list of (ref_id, t_a, t_b, t_c).
    ///
    /// `ref_id` is stored as `u32` (16 bytes per posting instead of 24
    /// for `(usize, u32, u32, u32)` on 64-bit).
    map: HashMap<u32, Postings<(u32, u32, u32, u32)>>,
    /// Per-reference frame rates for offset conversion.
    fps: Vec<f32>,
    /// Liveness bit parallel to `fps` (bookkeeping for
    /// [`PanakoIndex::live_count`] and slot reuse; removal is physical so
    /// query needs no guard).
    live: alloc::vec::Vec<bool>,
    /// LIFO stack of ids vacated by [`PanakoIndex::remove`].
    vacant: alloc::vec::Vec<u32>,
    /// Live reference count, maintained incrementally (O(1) read).
    live_count: usize,
    /// Reused touched-key scratch for [`PanakoIndex::insert`].
    touched: alloc::vec::Vec<u32>,
    /// Stop-hash tombstones (see [`WangIndex::stopped`]).
    stopped: HashMap<u32, ()>,
}

impl PanakoIndex {
    /// Build from a slice of Panako fingerprints.
    ///
    /// # Panics
    ///
    /// Panics if `refs.len()` exceeds `u32::MAX` (reference ids are stored
    /// as `u32`).
    pub fn build(refs: &[crate::classical::PanakoFingerprint], max_postings_per_hash: u32) -> Self {
        assert!(
            refs.len() <= u32::MAX as usize,
            "reference count exceeds u32::MAX"
        );
        let mut map: HashMap<u32, Postings<(u32, u32, u32, u32)>> =
            super::maps::hashmap_with_capacity(
                refs.iter().map(|r| r.hashes.len()).sum::<usize>() / 2,
            );
        let fps: Vec<f32> = refs.iter().map(|r| r.frames_per_sec).collect();

        for (ref_id, fp) in refs.iter().enumerate() {
            for h in &fp.hashes {
                map.entry(h.hash)
                    .or_default()
                    .push((ref_id as u32, h.t_anchor, h.t_b, h.t_c));
            }
        }

        let stopped = stop_list(&mut map, max_postings_per_hash);

        let live_count = refs.len();
        let live = alloc::vec![true; live_count];
        Self {
            map,
            fps,
            live,
            vacant: Vec::new(),
            live_count,
            touched: Vec::new(),
            stopped,
        }
    }

    /// Append one fingerprint to the catalog, returning its stable `ref_id`.
    ///
    /// Same stop-hash parity contract as [`WangIndex::insert`]: pruning the
    /// touched lists yields exactly the map state a full `build` + `retain`
    /// would produce for the same append-only input sequence (pinned by the
    /// `insert_matches_build_parity` test; parity is over append-only
    /// histories — interleaved `remove` calls physically delete postings,
    /// which `build` has no equivalent for). Ids of live references never
    /// change.
    ///
    /// # Performance
    ///
    /// `O(H)` amortised for `H` triplet hashes: one map lookup per hash plus
    /// touched-list pruning. Pre-reserved map capacity and reused
    /// touched-key scratch keep steady-state insertion free of auxiliary
    /// allocation beyond posting pushes (and rare map growth).
    ///
    /// # Panics
    ///
    /// Panics if the catalog already holds `u32::MAX` live references (same
    /// bound [`PanakoIndex::build`] enforces).
    pub fn insert(
        &mut self,
        fp: &crate::classical::PanakoFingerprint,
        max_postings_per_hash: u32,
    ) -> usize {
        let (ref_id, rid) = assign_ref_id(self.fps.len(), &mut self.vacant);
        #[cfg(feature = "std")]
        self.map.reserve(fp.hashes.len() / 2);
        insert_postings(
            &mut self.map,
            &mut self.stopped,
            &mut self.touched,
            fp.hashes
                .iter()
                .map(|h| (h.hash, (rid, h.t_anchor, h.t_b, h.t_c))),
            max_postings_per_hash,
        );
        if ref_id < self.fps.len() {
            self.fps[ref_id] = fp.frames_per_sec;
            self.live[ref_id] = true;
        } else {
            debug_assert_eq!(ref_id, self.fps.len());
            self.fps.push(fp.frames_per_sec);
            self.live.push(true);
        }
        self.live_count += 1;
        ref_id
    }

    /// Erase all postings for `ref_id`. Returns `false` if the id is
    /// out of range or already vacant (no-op).
    ///
    /// Removal is physical (postings are deleted from every list, not just
    /// flagged), so `query` needs no liveness guard and its throughput is
    /// bit-identical before and after removes. Ids of live references never
    /// shift.
    ///
    /// # Performance
    ///
    /// `O(P)` where `P` = total postings: one in-place `retain` scan per
    /// list over 16-byte tuples (memmove-compressed, no allocation), with
    /// keys the removal empties dropped (see [`PanakoIndex::len`]). Lists
    /// are pre-scanned read-only for the id first (see
    /// [`WangIndex::remove`]): most lists skip the write pass.
    pub fn remove(&mut self, ref_id: usize) -> bool {
        if ref_id >= self.live.len() || !self.live[ref_id] {
            return false;
        }
        let rid = ref_id as u32;
        erase_postings(&mut self.map, |&(r, _, _, _)| r == rid);
        self.live[ref_id] = false;
        self.fps[ref_id] = f32::NAN;
        self.vacant.push(rid);
        self.live_count -= 1;
        true
    }

    /// Erase many references in a SINGLE map pass. Returns the number of ids
    /// actually erased. Same batching rationale as [`WangIndex::remove_many`]:
    /// one `O(P)` pass instead of `K`. Vacated ids are pushed ascending
    /// (smallest reused first — documented).
    pub fn remove_many(&mut self, ref_ids: &[usize]) -> usize {
        let dead = live_ids(&self.live, ref_ids);
        if dead.is_empty() {
            return 0;
        }
        erase_postings(&mut self.map, |&(r, _, _, _)| {
            dead.binary_search(&r).is_ok()
        });
        let n = dead.len();
        for &rid in &dead {
            let id = rid as usize;
            self.live[id] = false;
            self.fps[id] = f32::NAN;
        }
        for &rid in dead.iter().rev() {
            self.vacant.push(rid);
        }
        self.live_count -= n;
        n
    }

    /// Append many fingerprints, returning their stable `ref_id`s in order.
    /// Reserves map capacity once for the whole batch; per-fingerprint
    /// semantics identical to [`PanakoIndex::insert`].
    pub fn insert_many(
        &mut self,
        fps: &[crate::classical::PanakoFingerprint],
        max_postings_per_hash: u32,
    ) -> Vec<usize> {
        #[cfg(feature = "std")]
        self.map
            .reserve(fps.iter().map(|f| f.hashes.len()).sum::<usize>() / 2);
        let mut ids = Vec::with_capacity(fps.len());
        for fp in fps {
            ids.push(self.insert(fp, max_postings_per_hash));
        }
        ids
    }

    /// Live reference count (excludes ids erased by [`PanakoIndex::remove`]).
    /// O(1) — maintained incrementally.
    #[must_use]
    pub fn live_count(&self) -> usize {
        self.live_count
    }

    /// Estimated P(same recording | evidence) for a [`PanakoIndex::query`]
    /// result: same `panako-v1` map as [`PanakoMatcher::calibrated_score`](super::PanakoMatcher::calibrated_score).
    /// Raw fields untouched.
    #[must_use]
    pub fn calibrated_score(&self, r: &super::MatchResult) -> f32 {
        super::calibration::calibrated_panako(r)
    }

    /// Measured heap footprint in bytes: posting lists (capacity × 16 bytes
    /// per `(u32, u32, u32, u32)` posting) + map slots + `fps` / `live` /
    /// `vacant` / `touched` storage and the stop-hash tombstones. O(map
    /// size) — call rarely, not per query. Tracks the heap to within
    /// allocator rounding on `std` (see [`WangIndex::estimated_bytes`]); for
    /// alerting and sharding, not billing.
    #[must_use]
    pub fn estimated_bytes(&self) -> usize {
        // The table bytes already hold one key and one value per bucket, so
        // there is no separate per-key term (audit §4.2 #10).
        let mut bytes = map_heap_bytes(&self.map) + map_heap_bytes(&self.stopped);
        for list in self.map.values() {
            bytes += list.heap_bytes();
        }
        bytes += self.fps.capacity() * size_of::<f32>();
        bytes += self.live.capacity() * size_of::<bool>();
        bytes += self.vacant.capacity() * size_of::<u32>();
        bytes += self.touched.capacity() * size_of::<u32>();
        bytes
    }

    /// Query the index, returning the best-matching `(ref_id, result)`.
    ///
    /// Uses a per-reference 2-D Hough accumulator (scale × offset)
    /// across the shared inverted index. Vote tallies are per-reference;
    /// the best reference that clears thresholds wins.
    #[must_use]
    pub fn query(
        &self,
        query: &crate::classical::PanakoFingerprint,
        cfg: &crate::matching::PanakoMatchConfig,
    ) -> Option<(usize, MatchResult)> {
        use super::panako::normalize_config;
        use crate::matching::{TimeOffset, clamp_score};

        // Normalize (clone) so a degenerate scale grid can't poison the
        // Hough accumulator with 0/∞/NaN bin widths — same treatment as
        // `PanakoMatcher::new`.
        let cfg = normalize_config(cfg.clone());

        if query.hashes.is_empty() || self.map.is_empty() {
            return None;
        }

        let scale_min = cfg.scale_min as f64;
        let scale_max = cfg.scale_max as f64;
        let scale_per_bin = (scale_max - scale_min) / cfg.scale_bins as f64;
        // ±half-bin slack so a scale at the grid edge still votes into the
        // boundary bin.
        let eps_scale = scale_per_bin * 0.5;
        let tol = cfg.offset_tolerance_frames as i64;
        let q_len = query.hashes.len().max(1) as f32;

        // Votes for every reference as one flat `(ref_id, scale_bin,
        // offset_bin)` list, sorted once: each reference's votes become a
        // contiguous run already ordered by `(scale_bin, offset_bin)`, which
        // is the order consolidation needs. This replaces a nested
        // `HashMap<ref, HashMap<bin, votes>>` and the per-candidate vectors
        // built from it (≈ 5 allocations per candidate reference). Capped
        // like `WangIndex::query` (16 bytes per vote → 160 MB worst case),
        // which also keeps a reference's vote total inside `u32`.
        let mut votes: Vec<(u32, u32, i64)> = Vec::with_capacity(
            query
                .hashes
                .len()
                .saturating_mul(2)
                .min(MAX_VOTES_PER_QUERY),
        );

        'outer: for h in &query.hashes {
            let hh = h.hash; // copy out of packed struct (unaligned refs are UB)
            let q_ta = h.t_anchor;
            let q_tc = h.t_c;
            if let Some(list) = self.map.get(&hh) {
                for &(ref_id, tr_a, _tr_b, tr_c) in list {
                    // Total arithmetic: `PanakoHash` fields are `pub` + `Pod`,
                    // so a malformed triplet (`t_c <= t_anchor`) can arrive
                    // from caller construction or `from_bytes`. Saturating
                    // subtraction avoids the debug panic / release wrap-around;
                    // degenerate spans carry no scale information, so skip.
                    let q_span = q_tc.saturating_sub(q_ta);
                    let r_span = tr_c.saturating_sub(tr_a);
                    if q_span == 0 || r_span == 0 {
                        continue;
                    }
                    let s = r_span as f64 / q_span as f64;

                    if s < scale_min - eps_scale || s > scale_max + eps_scale {
                        continue;
                    }

                    let b = tr_a as f64 - s * q_ta as f64;
                    let s_bin = ((s - scale_min) / scale_per_bin)
                        .clamp(0.0, (cfg.scale_bins - 1) as f64)
                        as u32;
                    // 1-frame offset granularity (see PanakoMatcher for
                    // why not `tol`-sized bins): the ±tol-bin
                    // consolidation window must mean ±tol frames.
                    let off_key = b.round() as i64;

                    if votes.len() >= MAX_VOTES_PER_QUERY {
                        break 'outer;
                    }
                    votes.push((ref_id, s_bin, off_key));
                }
            }
        }
        votes.sort_unstable();

        let mut best: Option<(usize, MatchResult)> = None;
        let mut bin_vec: Vec<((u32, i64), u32)> = Vec::new();
        let mut scratch = super::consolidate::HoughScratch::default();

        // Deterministic candidate order: ascending reference id (the sort
        // above), not HashMap order (same rationale as WangIndex::query).
        let mut group_start = 0usize;
        while group_start < votes.len() {
            let ref_id = votes[group_start].0;
            let mut group_end = group_start + 1;
            while group_end < votes.len() && votes[group_end].0 == ref_id {
                group_end += 1;
            }
            let group = &votes[group_start..group_end];
            group_start = group_end;

            // Frame-rate mismatch → `MatchResult::NONE`, matching every
            // matcher's contract (`matching/mod.rs`). Without this the
            // index would return a match the 1:1 matcher rejects.
            let ref_fps = self.fps[ref_id as usize];
            if !crate::matching::frames_per_sec_compatible(query.frames_per_sec, ref_fps) {
                continue;
            }
            // Quick pre-filter: if this reference's total vote count is
            // below min_votes, the consolidated peak cannot reach the
            // threshold — skip consolidation.
            if (group.len() as u64) < u64::from(cfg.min_votes) {
                continue;
            }

            // Per-bin counts, in `(s_bin, off_key)` order.
            bin_vec.clear();
            for &(_, s_bin, off_key) in group {
                match bin_vec.last_mut() {
                    Some((key, n)) if *key == (s_bin, off_key) => *n += 1,
                    _ => bin_vec.push(((s_bin, off_key), 1)),
                }
            }

            // Find the peak via neighbourhood consolidation; prominence is
            // computed on the same neighbourhood-summed values that
            // selected it (parity with PanakoMatcher and WangMatcher —
            // audit B5).
            let (peak_votes, peak_linear_idx) =
                super::consolidate::panako_hough_peak(&bin_vec, tol, &mut scratch);
            if peak_votes < cfg.min_votes {
                continue;
            }
            let ((peak_s_bin, _), _) = bin_vec[peak_linear_idx];
            let peak_off_key = super::consolidate::panako_peak_offset(
                &bin_vec,
                peak_linear_idx,
                tol,
                &mut scratch,
            );
            let prominence = super::consolidate::panako_prominence(
                &bin_vec,
                &scratch.consolidated,
                peak_linear_idx,
                tol,
            );
            if super::below_threshold(prominence, cfg.min_prominence) {
                continue;
            }

            let score = clamp_score(peak_votes as f32 / q_len);
            if super::below_threshold(score, cfg.min_score) {
                continue;
            }

            let coarse_s = scale_min + (peak_s_bin as f64 + 0.5) * scale_per_bin;
            let coarse_b = peak_off_key as f64;

            // Public contract: `time_scale = query_duration /
            // reference_duration` = `1 / s` where `s = ref_span /
            // query_span` is the internal Hough scale (audit B3).
            let time_scale = if coarse_s.abs() > 1e-6 {
                (1.0 / coarse_s).clamp(0.5, 2.0) as f32
            } else {
                1.0
            };

            // fps always populated by `build`/`insert` (compatibility was
            // checked above); the 62.5 fallback is defensive only.
            let fps = self.fps.get(ref_id as usize).copied().unwrap_or(62.5);
            let result = MatchResult {
                is_match: true,
                score,
                votes: peak_votes,
                prominence,
                offset: TimeOffset::from_frames(coarse_b.round() as i64, fps),
                time_scale,
            };

            track_best(&mut best, ref_id as usize, result);
        }

        best
    }

    /// Return the number of unique hash keys in the index.
    ///
    /// This is **key-space**, not catalog size: the number of hashes with at
    /// least one live posting (keys [`PanakoIndex::remove`] empties are dropped).
    /// Use [`PanakoIndex::live_count`] for the number of references.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Return `true` if the index holds **no hash keys at all**.
    ///
    /// Key-space, like [`PanakoIndex::len`]. A catalog whose references all
    /// had their hashes stop-listed is non-empty yet has no keys; to test
    /// whether any reference is enrolled, use [`PanakoIndex::is_empty_catalog`].
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Return `true` if the catalog holds no live references.
    ///
    /// This is the "no tracks enrolled" test; unlike
    /// [`PanakoIndex::is_empty`] it tracks removals.
    #[must_use]
    pub fn is_empty_catalog(&self) -> bool {
        self.live_count == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classical::{HaitsmaFingerprint, PanakoFingerprint, WangFingerprint};
    use crate::matching::{HaitsmaMatchConfig, PanakoMatchConfig, WangMatchConfig, WangMatcher};
    use alloc::vec;

    // --- match_best / match_ranked ---

    // ── Rayon parallel matching (audit C5) ──
    //
    // par_match_best / par_match_ranked must produce results identical
    // to their sequential counterparts, including deterministic tie
    // resolution (lowest ref id) and preserved ranking order.

    #[cfg(feature = "rayon")]
    #[test]
    fn par_match_ranked_equals_sequential() {
        let cfg = WangMatchConfig {
            min_votes: 0,
            min_score: -1.0,
            min_prominence: -1.0,
            ..Default::default()
        };
        let matcher = WangMatcher::new(cfg);
        let query = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 0);
        // Distinct scores per ref: each ref shares more anchors with the
        // query as the offset grows.
        let refs: Vec<WangFingerprint> = (0..64u32)
            .map(|i| {
                let mut anchors: Vec<u32> = (0..(8u32 + i % 9)).map(|k| 10 + k * 10).collect();
                anchors.extend((1000..1000 + i * 13).map(|t| t % 5000));
                mk(&anchors, 1)
            })
            .collect();

        let seq = match_ranked(&matcher, &query, &refs);
        let par = par_match_ranked(&matcher, &query, &refs);
        assert_eq!(par, seq, "parallel ranking must match sequential exactly");
        assert_eq!(par.len(), refs.len());
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn par_match_best_equals_sequential() {
        let cfg = WangMatchConfig {
            min_votes: 0,
            min_score: -1.0,
            min_prominence: -1.0,
            ..Default::default()
        };
        let matcher = WangMatcher::new(cfg);
        let query = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 0);
        let refs: Vec<WangFingerprint> = (0..64u32)
            .map(|i| {
                let mut anchors: Vec<u32> = (0..(8u32 + i % 9)).map(|k| 10 + k * 10).collect();
                anchors.extend((1000..1000 + i * 13).map(|t| t % 5000));
                mk(&anchors, 1)
            })
            .collect();

        let seq = match_best(&matcher, &query, &refs);
        let par = par_match_best(&matcher, &query, &refs);
        assert_eq!(par, seq, "parallel best must match sequential exactly");
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn par_match_best_tie_resolves_to_lowest_ref_id() {
        let cfg = WangMatchConfig {
            min_votes: 0,
            min_score: -1.0,
            min_prominence: -1.0,
            ..Default::default()
        };
        let matcher = WangMatcher::new(cfg);
        let fp = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 0);
        let refs = alloc::vec![fp.clone(), fp.clone(), fp.clone()];
        for _ in 0..8 {
            let (id, res) =
                par_match_best(&matcher, &fp, &refs).expect("identical references must match");
            assert_eq!(id, 0, "perfect-score tie must resolve to ref 0");
            assert!(res.is_match);
        }
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn par_match_empty_and_no_match() {
        let cfg = WangMatchConfig::default();
        let matcher = WangMatcher::new(cfg);
        let fp = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 0);
        assert!(par_match_best(&matcher, &fp, &[]).is_none());
        assert!(par_match_ranked(&matcher, &fp, &[]).is_empty());

        // No reference matches (unrelated hashes) → best is None and
        // ranked returns only non-matches.
        let unrelated = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 9_000);
        let refs = alloc::vec![mk(&[10, 20, 30, 40, 50, 60, 70, 80], 5_000)];
        assert!(par_match_best(&matcher, &unrelated, &refs).is_none());
        assert_eq!(par_match_ranked(&matcher, &unrelated, &refs).len(), 1);
    }

    #[test]
    fn match_ranked_empty_refs() {
        let cfg = WangMatchConfig {
            min_votes: 0,
            ..Default::default()
        };
        let m = WangMatcher::new(cfg);
        let fp = WangFingerprint {
            hashes: alloc::vec![],
            frames_per_sec: 62.5,
        };
        let results = match_ranked(&m, &fp, &[]);
        assert!(results.is_empty());
    }

    #[test]
    fn match_ranked_returns_all() {
        let cfg = WangMatchConfig {
            min_votes: 0,
            min_score: -1.0,
            min_prominence: -1.0,
            ..Default::default()
        };
        let m = WangMatcher::new(cfg);
        let fp = WangFingerprint {
            hashes: alloc::vec![],
            frames_per_sec: 62.5,
        };
        let refs = alloc::vec![fp.clone(), fp.clone(), fp.clone()];
        let results = match_ranked(&m, &fp, &refs);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn match_best_returns_is_match() {
        let cfg = WangMatchConfig {
            min_votes: 0,
            min_score: -1.0,
            min_prominence: -1.0,
            ..Default::default()
        };
        let m = WangMatcher::new(cfg);
        let fp = WangFingerprint {
            hashes: alloc::vec![],
            frames_per_sec: 62.5,
        };
        let refs = alloc::vec![fp.clone()];
        assert!(match_best(&m, &fp, &refs).is_none());
    }

    #[test]
    fn match_ranked_is_sorted_by_score() {
        let cfg = WangMatchConfig::default();
        let m = WangMatcher::new(cfg);

        // Build 50 synthetic refs + query.
        let query = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 0);
        let refs: Vec<WangFingerprint> = (0..50u32)
            .map(|i| mk(&[10, 20, 30, 40, 50, 60, 70, 80], i + 1))
            .collect();

        let results = match_ranked(&m, &query, &refs);
        assert_eq!(results.len(), 50);
        // Results must be sorted descending by score.
        for w in results.windows(2) {
            let (_, a) = &w[0];
            let (_, b) = &w[1];
            assert!(
                a.score >= b.score || (a.score - b.score).abs() < 1e-6,
                "not sorted: {} (score {}) before {} (score {})",
                w[0].0,
                a.score,
                w[1].0,
                b.score
            );
        }
    }

    /// Build a synthetic Wang fingerprint helper.
    fn mk(anchors: &[u32], hash_offset: u32) -> WangFingerprint {
        use crate::classical::WangHash;
        WangFingerprint {
            hashes: anchors
                .iter()
                .enumerate()
                .map(|(i, &t)| WangHash {
                    hash: (i as u32).wrapping_add(hash_offset),
                    t_anchor: t,
                })
                .collect(),
            frames_per_sec: 62.5,
        }
    }

    // --- WangIndex ---

    #[test]
    fn wang_index_returns_correct_ref_id() {
        let r0 = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 0);
        let r1 = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 1_000);
        let r2 = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 2_000);
        let refs = alloc::vec![r0, r1, r2.clone()];
        let index = WangIndex::build(&refs, 100);
        let cfg = WangMatchConfig::default();

        let (id, res) = index
            .query(&r2, &cfg)
            .expect("query identical to reference 2 must match");
        assert_eq!(id, 2, "must identify reference 2");
        assert!(res.is_match);
        assert_eq!(res.offset.frames, 0);
    }

    #[test]
    fn wang_index_recovers_offset_and_ref_id() {
        let r0 = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 0);
        let r1 = mk(&[60, 70, 80, 90, 100, 110, 120, 130], 1_000);
        let refs = alloc::vec![r0, r1];
        let index = WangIndex::build(&refs, 100);
        let cfg = WangMatchConfig::default();

        let query = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 1_000);
        let (id, res) = index.query(&query, &cfg).expect("must match reference 1");
        assert_eq!(id, 1);
        assert_eq!(res.offset.frames, 50);
    }

    #[test]
    fn wang_index_no_match_for_unrelated_query() {
        let r0 = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 0);
        let refs = alloc::vec![r0];
        let index = WangIndex::build(&refs, 100);
        let cfg = WangMatchConfig::default();
        let query = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 9_000);
        assert!(index.query(&query, &cfg).is_none());
    }

    #[test]
    fn wang_index_len_and_empty() {
        let refs = alloc::vec![mk(&[10, 20, 30], 0)];
        let index = WangIndex::build(&refs, 100);
        assert!(!index.is_empty());
        assert_eq!(index.len(), 3); // 3 unique hashes (by anchor idx)
    }

    // --- HaitsmaIndex ---

    fn mk_haitsma_fp(frames: &[u32], fps: f32) -> HaitsmaFingerprint {
        HaitsmaFingerprint {
            frames: frames.to_vec(),
            frames_per_sec: fps,
        }
    }

    #[test]
    fn haitsma_index_rejects_silent_and_repeated_queries() {
        // A posting limit far above the frame count keeps the stop-hash
        // pruning from hiding the problem: only the guard can reject these.
        let real: Vec<u32> = (0..600)
            .map(|i| (i as u32).wrapping_mul(0x9E37_79B1) ^ 0x1234_5678)
            .collect();
        let cases: Vec<(&str, Vec<u32>)> = vec![
            ("silence", vec![0; 400]),
            ("all ones", vec![u32::MAX; 400]),
            ("alternating", (0..400).map(|i| i as u32 % 2).collect()),
            (
                "period 3",
                (0..400)
                    .map(|i| [0xAAAA_AAAA_u32, 0x5555_5555, 0][i % 3])
                    .collect(),
            ),
        ];
        let cfg = HaitsmaMatchConfig::default();
        for (name, frames) in cases {
            let degenerate = mk_haitsma_fp(&frames, 78.125);
            // Catalogue holds the degenerate clip itself AND a real clip.
            let index =
                HaitsmaIndex::build(&[degenerate.clone(), mk_haitsma_fp(&real, 78.125)], 10_000);
            assert!(
                index.query(&degenerate, &cfg).is_none(),
                "{name}: degenerate query must not match"
            );
        }
    }

    #[test]
    fn haitsma_index_still_finds_real_content_beside_a_silent_reference() {
        let real: Vec<u32> = (0..600)
            .map(|i| (i as u32).wrapping_mul(0x9E37_79B1) ^ 0x1234_5678)
            .collect();
        let silent = mk_haitsma_fp(&vec![0; 600], 78.125);
        let index = HaitsmaIndex::build(&[silent, mk_haitsma_fp(&real, 78.125)], 10_000);
        let query = mk_haitsma_fp(&real[100..500], 78.125);
        let (id, r) = index
            .query(&query, &HaitsmaMatchConfig::default())
            .expect("real excerpt must match");
        assert_eq!(id, 1);
        assert!(r.is_match);
        assert_eq!(r.offset.frames, 100);
        assert_eq!(r.score, 1.0);
    }

    #[test]
    fn haitsma_index_self_match() {
        let ref_frames: Vec<u32> = (0..600)
            .map(|i| (i as u32).wrapping_mul(0x01010101))
            .collect();
        let fp = mk_haitsma_fp(&ref_frames, 78.125);
        let refs = alloc::vec![fp.clone()];
        let index = HaitsmaIndex::build(&refs, 100);
        let cfg = HaitsmaMatchConfig {
            min_overlap_frames: 256,
            ..Default::default()
        };

        let (id, res) = index.query(&fp, &cfg).expect("self-match must succeed");
        assert_eq!(id, 0);
        assert!(res.is_match);
        assert_eq!(res.offset.frames, 0);
        assert!((res.score - 1.0).abs() < 0.001, "score={}", res.score);
    }

    #[test]
    fn haitsma_index_identifies_correct_ref() {
        let r0: Vec<u32> = (0..600).map(|i| (i as u32).wrapping_mul(7919)).collect();
        let r1: Vec<u32> = (0..600)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761))
            .collect();
        let r2: Vec<u32> = (0..600)
            .map(|i| (i as u32).wrapping_mul(1_030_301))
            .collect();

        let refs = alloc::vec![
            mk_haitsma_fp(&r0, 78.125),
            mk_haitsma_fp(&r1, 78.125),
            mk_haitsma_fp(&r2, 78.125),
        ];
        let index = HaitsmaIndex::build(&refs, 100);
        let cfg = HaitsmaMatchConfig {
            min_overlap_frames: 256,
            ..Default::default()
        };

        let (id, _res) = index.query(&refs[1], &cfg).expect("must find reference 1");
        assert_eq!(id, 1);
    }

    #[test]
    fn haitsma_index_no_match_unrelated() {
        // Use large, well-separated constants so the two sets share
        // no coincidental 32-bit frame values.
        let r0: Vec<u32> = (0..600)
            .map(|i| (i as u32).wrapping_mul(7919).wrapping_add(0xA000_0000))
            .collect();
        let refs = alloc::vec![mk_haitsma_fp(&r0, 78.125)];
        let index = HaitsmaIndex::build(&refs, 100);
        let cfg = HaitsmaMatchConfig::default();

        let unrelated: Vec<u32> = (0..600)
            .map(|i| {
                (i as u32)
                    .wrapping_mul(0xDEAD_BEEF)
                    .wrapping_add(0x5000_0000)
            })
            .collect();
        let q = mk_haitsma_fp(&unrelated, 78.125);
        assert!(
            index.query(&q, &cfg).is_none(),
            "unrelated query must not match"
        );
    }

    #[test]
    fn haitsma_index_offset_recovery() {
        let ref_frames: Vec<u32> = (0..800).map(|i| (i as u32).wrapping_mul(7919)).collect();
        // Query = reference frames 100..500
        let query_frames: Vec<u32> = ref_frames[100..500].to_vec();
        let refs = alloc::vec![mk_haitsma_fp(&ref_frames, 78.125)];
        let index = HaitsmaIndex::build(&refs, 100);
        let cfg = HaitsmaMatchConfig {
            min_overlap_frames: 200,
            ..Default::default()
        };

        let (_id, res) = index
            .query(&mk_haitsma_fp(&query_frames, 78.125), &cfg)
            .expect("must find offset sub-sequence");
        assert_eq!(
            res.offset.frames, 100,
            "offset must be +100, got {}",
            res.offset.frames
        );
    }

    #[test]
    fn haitsma_index_empty_refs_returns_none() {
        let index = HaitsmaIndex::build(&[], 100);
        let cfg = HaitsmaMatchConfig::default();
        let q = mk_haitsma_fp(&[1, 2, 3, 4, 5], 78.125);
        assert!(index.query(&q, &cfg).is_none());
        assert!(index.is_empty());
    }

    #[test]
    fn haitsma_index_stop_hash_prune() {
        // max_postings_per_hash = 1: a hash appearing in two refs is dropped.
        let shared = 0xABCD_EF01u32;
        let r0 = mk_haitsma_fp(&[shared, 0x1111_1111, 0x2222_2222], 78.125);
        let r1 = mk_haitsma_fp(&[shared, 0x3333_3333, 0x4444_4444], 78.125);
        let index = HaitsmaIndex::build(&[r0, r1], 1);
        // Only unique frames remain; the shared hash must be pruned.
        assert!(
            index.len() < 5,
            "stop-hash prune should drop the shared posting"
        );
    }

    // --- PanakoIndex ---

    fn mk_panako_fp(triples: &[(u32, u32, u32)], hash_offset: u32) -> PanakoFingerprint {
        use crate::classical::PanakoHash;
        PanakoFingerprint {
            hashes: triples
                .iter()
                .enumerate()
                .map(|(i, &(ta, tb, tc))| PanakoHash {
                    hash: (i as u32).wrapping_add(hash_offset),
                    t_anchor: ta,
                    t_b: tb,
                    t_c: tc,
                })
                .collect(),
            frames_per_sec: 62.5,
        }
    }

    #[test]
    fn panako_index_self_match() {
        let fp = mk_panako_fp(
            &[
                (10, 20, 30),
                (50, 60, 70),
                (90, 100, 110),
                (130, 140, 150),
                (170, 180, 190),
                (210, 220, 230),
                (250, 260, 270),
                (290, 300, 310),
            ],
            0,
        );
        let refs = alloc::vec![fp.clone()];
        let index = PanakoIndex::build(&refs, 100);
        let cfg = PanakoMatchConfig::default();

        let (id, res) = index.query(&fp, &cfg).expect("self-match must succeed");
        assert_eq!(id, 0);
        assert!(res.is_match);
        assert_eq!(res.offset.frames, 0);
        assert!(
            (res.time_scale - 1.0).abs() < 0.1,
            "self-match time_scale must be ~1.0, got {}",
            res.time_scale
        );
    }

    #[test]
    fn panako_index_recovers_offset() {
        let r = mk_panako_fp(
            &[
                (150, 155, 165),
                (190, 195, 205),
                (230, 235, 245),
                (270, 275, 285),
                (310, 315, 325),
                (350, 355, 365),
                (390, 395, 405),
                (430, 435, 445),
            ],
            0,
        );
        let q = mk_panako_fp(
            &[
                (100, 105, 115),
                (140, 145, 155),
                (180, 185, 195),
                (220, 225, 235),
                (260, 265, 275),
                (300, 305, 315),
                (340, 345, 355),
                (380, 385, 395),
            ],
            0,
        );
        let refs = alloc::vec![r];
        let index = PanakoIndex::build(&refs, 100);
        let cfg = PanakoMatchConfig {
            ransac_refine: false,
            ..Default::default()
        };
        let (id, res) = index.query(&q, &cfg).expect("offset match must succeed");
        assert_eq!(id, 0);
        assert_eq!(
            res.offset.frames, 50,
            "offset must be +50, got {}",
            res.offset.frames
        );
        assert!(
            (res.time_scale - 1.0).abs() < 0.1,
            "constant-tempo time_scale must be ~1.0, got {}",
            res.time_scale
        );
    }

    #[test]
    fn panako_index_recovers_scale() {
        // Query timestamps compressed by 0.9 → internal s ≈ 1.111,
        // public time_scale ≈ 0.90.
        let normal = mk_panako_fp(
            &[
                (100, 120, 150),
                (200, 220, 250),
                (300, 320, 350),
                (400, 420, 450),
                (500, 520, 550),
                (600, 620, 650),
                (700, 720, 750),
                (800, 820, 850),
            ],
            0,
        );
        let fast = mk_panako_fp(
            &[
                (90, 108, 135),
                (180, 198, 225),
                (270, 288, 315),
                (360, 378, 405),
                (450, 468, 495),
                (540, 558, 585),
                (630, 648, 675),
                (720, 738, 765),
            ],
            0,
        );
        let refs = alloc::vec![normal];
        let index = PanakoIndex::build(&refs, 100);
        let cfg = PanakoMatchConfig {
            scale_min: 0.85,
            scale_max: 1.20,
            ransac_refine: false,
            min_votes: 3,
            min_prominence: 1.0,
            min_score: 0.05,
            ..Default::default()
        };
        let (id, res) = index
            .query(&fast, &cfg)
            .expect("tempo-stretched query must match");
        assert_eq!(id, 0);
        assert!(
            (res.time_scale - 0.9).abs() < 0.08,
            "expected time_scale ~0.90, got {}",
            res.time_scale
        );
    }

    #[test]
    fn panako_index_identifies_correct_ref() {
        // Use enough triples to satisfy min_votes=5.
        let triples: Vec<(u32, u32, u32)> = (0..10u32)
            .map(|i| (i * 40 + 10, i * 40 + 20, i * 40 + 30))
            .collect();
        let r0 = mk_panako_fp(&triples, 0);
        let r1 = mk_panako_fp(&triples, 1_000);
        let refs = alloc::vec![r0.clone(), r1.clone()];
        let index = PanakoIndex::build(&refs, 100);
        let cfg = PanakoMatchConfig {
            min_votes: 3,
            min_prominence: 1.0,
            min_score: 0.05,
            ..Default::default()
        };

        let (id, _res) = index.query(&r1, &cfg).expect("must find reference 1");
        assert_eq!(id, 1);
    }

    #[test]
    fn panako_index_no_match_unrelated() {
        let r0 = mk_panako_fp(&[(10, 20, 30), (50, 60, 70)], 0);
        let refs = alloc::vec![r0];
        let index = PanakoIndex::build(&refs, 100);
        let cfg = PanakoMatchConfig::default();
        let q = mk_panako_fp(&[(10, 20, 30), (50, 60, 70)], 9_000);
        assert!(index.query(&q, &cfg).is_none());
    }

    // ── Mutable catalog: insert / remove / live_count / estimated_bytes ──
    //
    // Parity contract: build(all) ≡ build(prefix) + insert(rest) on query
    // outputs (append-only histories). Removal is physical, slot reuse is
    // LIFO, live_count is O(1), estimated_bytes is a documented
    // approximation.

    #[test]
    fn wang_insert_matches_build_parity() {
        let refs: Vec<WangFingerprint> = (0..8u32)
            .map(|i| mk(&[10, 20, 30, 40, 50, 60, 70, 80], i * 1_000))
            .collect();
        let full = WangIndex::build(&refs, 100);
        let mut inc = WangIndex::build(&refs[..3], 100);
        for fp in &refs[3..] {
            inc.insert(fp, 100);
        }
        assert_eq!(inc.live_count(), 8);
        let cfg = WangMatchConfig::default();
        for (qi, q) in refs.iter().enumerate() {
            assert_eq!(
                full.query(q, &cfg),
                inc.query(q, &cfg),
                "query {qi} must agree between build and build+insert"
            );
        }
    }

    #[test]
    fn wang_insert_returns_stable_ids() {
        let mut index = WangIndex::build(&[], 100);
        let a = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 0);
        let b = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 1_000);
        assert_eq!(index.insert(&a, 100), 0);
        assert_eq!(index.insert(&b, 100), 1);
        assert_eq!(index.live_count(), 2);
        let cfg = WangMatchConfig::default();
        let (id, res) = index.query(&b, &cfg).expect("inserted ref must match");
        assert_eq!(id, 1);
        assert!(res.is_match);
    }

    #[test]
    fn wang_remove_erases_and_reports() {
        let r0 = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 0);
        let r1 = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 1_000);
        let mut index = WangIndex::build(&[r0.clone(), r1.clone()], 100);
        let cfg = WangMatchConfig::default();
        assert!(index.remove(0));
        assert!(!index.remove(0), "double remove is a no-op");
        assert!(!index.remove(99), "out-of-range remove is a no-op");
        assert_eq!(index.live_count(), 1);
        // Erased ref no longer matches; survivor is unaffected.
        assert!(index.query(&r0, &cfg).is_none());
        let (id, res) = index.query(&r1, &cfg).expect("survivor must match");
        assert_eq!(id, 1);
        assert!(res.is_match);
    }

    #[test]
    fn wang_insert_reuses_vacated_slot_lifo() {
        let r0 = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 0);
        let r1 = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 1_000);
        let r2 = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 2_000);
        let mut index = WangIndex::build(&[r0, r1.clone(), r2.clone()], 100);
        assert!(index.remove(2));
        assert!(index.remove(0));
        // LIFO: most-recently-vacated (0) is reused first.
        let r3 = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 3_000);
        assert_eq!(index.insert(&r3, 100), 0);
        let r4 = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 4_000);
        assert_eq!(index.insert(&r4, 100), 2);
        assert_eq!(index.live_count(), 3);
        let cfg = WangMatchConfig::default();
        let (id, _) = index.query(&r3, &cfg).expect("reused slot must match");
        assert_eq!(id, 0);
    }

    #[test]
    fn wang_estimated_bytes_grows_with_catalog() {
        let empty = WangIndex::build(&[], 100);
        assert_eq!(empty.live_count(), 0);
        let refs: Vec<WangFingerprint> = (0..4u32)
            .map(|i| mk(&[10, 20, 30, 40, 50, 60, 70, 80], i * 1_000))
            .collect();
        let full = WangIndex::build(&refs, 100);
        assert!(
            full.estimated_bytes() > empty.estimated_bytes(),
            "footprint must grow with catalog"
        );
        // Lower bound sanity: at least 8 bytes per posting + fps/live bits.
        let postings: usize = refs.iter().map(|r| r.hashes.len()).sum();
        assert!(
            full.estimated_bytes() >= postings * 8,
            "estimate {} must cover {} postings × 8 bytes",
            full.estimated_bytes(),
            postings
        );
    }

    /// Enroll/erase churn with one live track at a time must not grow the
    /// key space: `remove` used to keep every emptied key, so 200 cycles
    /// left ~400k keys (~31 MB) behind a single 2 000-frame reference.
    #[test]
    fn churn_does_not_grow_the_key_space() {
        let mk_h = |seed: u32| {
            mk_haitsma_fp(
                &(0..2_000)
                    .map(|i| (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed))
                    .collect::<Vec<_>>(),
                78.125,
            )
        };
        let mut h = HaitsmaIndex::build(&[mk_h(0)], 100);
        let mut w = WangIndex::build(&[mk(&(0..2_000).collect::<Vec<_>>(), 0)], 100);
        let mut p = PanakoIndex::build(&[mk_panako_fp(&[(0, 1, 2); 500], 0)], 100);
        for cycle in 1..=200_u32 {
            let id = h.insert(&mk_h(cycle * 7_919), 100);
            assert!(h.remove(id));
            let id = w.insert(&mk(&(0..2_000).collect::<Vec<_>>(), cycle * 10_000), 100);
            assert!(w.remove(id));
            let id = p.insert(&mk_panako_fp(&[(0, 1, 2); 500], cycle * 1_000), 100);
            assert!(p.remove(id));
        }
        assert_eq!(h.len(), 2_000);
        assert_eq!(w.len(), 2_000);
        assert_eq!(p.len(), 500);
        assert!(h.remove_many(&[0]) == 1 && h.is_empty());
        assert!(w.remove_many(&[0]) == 1 && w.is_empty());
        assert!(p.remove_many(&[0]) == 1 && p.is_empty());
    }

    /// The index must reject a reference the 1:1 matcher rejects as
    /// uninformative: 1 000 identical frames against a 20-frame identical
    /// query used to match with score 1.0 and prominence 100.
    #[test]
    fn haitsma_index_rejects_an_uninformative_reference() {
        let reference = mk_haitsma_fp(&[0xDEAD_BEEF; 1_000], 78.125);
        let query = mk_haitsma_fp(&[0xDEAD_BEEF; 20], 78.125);
        let cfg = HaitsmaMatchConfig {
            min_overlap_frames: 16,
            ..Default::default()
        };
        let matcher = crate::matching::HaitsmaMatcher::new(cfg.clone());
        assert!(!matcher.match_one(&query, &reference).is_match);
        let index = HaitsmaIndex::build(&[reference], 10_000);
        assert_eq!(index.query(&query, &cfg), None);
    }

    /// `HaitsmaIndex::remove` finds a reference's postings through its own
    /// frames rather than scanning the LUT; the result must equal a build
    /// that never contained it, including for a reference that shares
    /// frame values with a survivor.
    #[test]
    fn haitsma_remove_matches_build_without_the_reference() {
        let mk_h = |seed: u32| {
            mk_haitsma_fp(
                &(0..600)
                    .map(|i| (i as u32).wrapping_mul(seed))
                    .collect::<Vec<_>>(),
                78.125,
            )
        };
        let refs: Vec<HaitsmaFingerprint> = [7919, 2_654_435_761, 7919, 40503].map(mk_h).to_vec();
        let mut removed = HaitsmaIndex::build(&refs, 100);
        assert!(removed.remove(1));
        assert_eq!(removed.remove_many(&[2, 2, 9]), 1);
        let survivors = [refs[0].clone(), refs[3].clone()];
        let rebuilt = HaitsmaIndex::build(&survivors, 100);
        assert_eq!(removed.len(), rebuilt.len());
        let cfg = HaitsmaMatchConfig {
            min_overlap_frames: 256,
            ..Default::default()
        };
        for (q, want_id) in [(&refs[0], 0), (&refs[3], 3)] {
            let got = removed.query(q, &cfg).expect("survivor must match");
            let want = rebuilt.query(q, &cfg).expect("survivor must match");
            assert_eq!(got.0, want_id);
            assert_eq!(got.1, want.1);
        }
        assert!(removed.query(&refs[1], &cfg).is_none());
    }

    #[test]
    fn haitsma_insert_matches_build_parity() {
        let mk_h = |seed: u32| {
            mk_haitsma_fp(
                &(0..600)
                    .map(|i| (i as u32).wrapping_mul(seed))
                    .collect::<Vec<_>>(),
                78.125,
            )
        };
        let refs: Vec<HaitsmaFingerprint> =
            [7919, 2_654_435_761, 1_030_301, 40503].map(mk_h).to_vec();
        let full = HaitsmaIndex::build(&refs, 100);
        let mut inc = HaitsmaIndex::build(&refs[..1], 100);
        for fp in &refs[1..] {
            inc.insert(fp, 100);
        }
        assert_eq!(inc.live_count(), 4);
        let cfg = HaitsmaMatchConfig {
            min_overlap_frames: 256,
            ..Default::default()
        };
        for (qi, q) in refs.iter().enumerate() {
            assert_eq!(
                full.query(q, &cfg),
                inc.query(q, &cfg),
                "query {qi} must agree between build and build+insert"
            );
        }
    }

    #[test]
    fn haitsma_remove_frees_frames_and_reports() {
        let r0 = mk_haitsma_fp(
            &(0..600)
                .map(|i| (i as u32).wrapping_mul(7919))
                .collect::<Vec<_>>(),
            78.125,
        );
        let r1 = mk_haitsma_fp(
            &(0..600)
                .map(|i| (i as u32).wrapping_mul(2_654_435_761))
                .collect::<Vec<_>>(),
            78.125,
        );
        let mut index = HaitsmaIndex::build(&[r0.clone(), r1.clone()], 100);
        let before = index.estimated_bytes();
        let cfg = HaitsmaMatchConfig {
            min_overlap_frames: 256,
            ..Default::default()
        };
        assert!(index.remove(0));
        assert!(!index.remove(0));
        assert_eq!(index.live_count(), 1);
        // Frame memory for the erased ref is freed: estimate must shrink by
        // at least the 600×4-byte frame vector, modulo small bookkeeping
        // growth (the `vacant` push can add one allocation quantum).
        assert!(
            index.estimated_bytes() + 600 * 4 <= before + 256,
            "remove must free the frame vector"
        );
        let (id, res) = index.query(&r1, &cfg).expect("survivor must match");
        assert_eq!(id, 1);
        assert!(res.is_match);
    }

    #[test]
    fn haitsma_insert_reuses_vacated_slot() {
        let mk_h = |seed: u32| {
            mk_haitsma_fp(
                &(0..600)
                    .map(|i| (i as u32).wrapping_mul(seed))
                    .collect::<Vec<_>>(),
                78.125,
            )
        };
        let mut index = HaitsmaIndex::build(&[mk_h(7919)], 100);
        assert!(index.remove(0));
        let id = index.insert(&mk_h(2_654_435_761), 100);
        assert_eq!(id, 0, "vacated slot must be reused");
        assert_eq!(index.live_count(), 1);
    }

    #[test]
    fn panako_insert_matches_build_parity() {
        let triples: Vec<(u32, u32, u32)> = (0..10u32)
            .map(|i| (i * 40 + 10, i * 40 + 20, i * 40 + 30))
            .collect();
        let refs = alloc::vec![
            mk_panako_fp(&triples, 0),
            mk_panako_fp(&triples, 1_000),
            mk_panako_fp(&triples, 2_000),
        ];
        let full = PanakoIndex::build(&refs, 100);
        let mut inc = PanakoIndex::build(&refs[..1], 100);
        for fp in &refs[1..] {
            inc.insert(fp, 100);
        }
        assert_eq!(inc.live_count(), 3);
        let cfg = PanakoMatchConfig {
            min_votes: 3,
            min_prominence: 1.0,
            min_score: 0.05,
            ..Default::default()
        };
        for (qi, q) in refs.iter().enumerate() {
            assert_eq!(
                full.query(q, &cfg),
                inc.query(q, &cfg),
                "query {qi} must agree between build and build+insert"
            );
        }
    }

    #[test]
    fn panako_remove_and_slot_reuse() {
        let triples: Vec<(u32, u32, u32)> = (0..10u32)
            .map(|i| (i * 40 + 10, i * 40 + 20, i * 40 + 30))
            .collect();
        let r0 = mk_panako_fp(&triples, 0);
        let r1 = mk_panako_fp(&triples, 1_000);
        let mut index = PanakoIndex::build(&[r0, r1.clone()], 100);
        let cfg = PanakoMatchConfig {
            min_votes: 3,
            min_prominence: 1.0,
            min_score: 0.05,
            ..Default::default()
        };
        assert!(index.remove(0));
        assert_eq!(index.live_count(), 1);
        let (id, res) = index.query(&r1, &cfg).expect("survivor must match");
        assert_eq!(id, 1);
        assert!(res.is_match);
        let r2 = mk_panako_fp(&triples, 2_000);
        assert_eq!(index.insert(&r2, 100), 0, "vacated slot must be reused");
        assert_eq!(index.live_count(), 2);
        assert!(index.estimated_bytes() > 0);
    }

    // ── Batch APIs: insert_many / remove_many ──

    #[test]
    fn wang_insert_many_matches_looped_insert() {
        let refs: Vec<WangFingerprint> = (0..6u32)
            .map(|i| mk(&[10, 20, 30, 40, 50, 60, 70, 80], i * 1_000))
            .collect();
        let mut a = WangIndex::build(&refs[..2], 100);
        let mut b = WangIndex::build(&refs[..2], 100);
        let ids_a: Vec<usize> = refs[2..].iter().map(|fp| a.insert(fp, 100)).collect();
        let ids_b = b.insert_many(&refs[2..], 100);
        assert_eq!(ids_a, ids_b);
        assert_eq!(ids_b, vec![2, 3, 4, 5]);
        let cfg = WangMatchConfig::default();
        for q in &refs {
            assert_eq!(a.query(q, &cfg), b.query(q, &cfg));
        }
    }

    #[test]
    fn wang_remove_many_matches_looped_remove() {
        let refs: Vec<WangFingerprint> = (0..6u32)
            .map(|i| mk(&[10, 20, 30, 40, 50, 60, 70, 80], i * 1_000))
            .collect();
        let mut a = WangIndex::build(&refs, 100);
        let mut b = WangIndex::build(&refs, 100);
        for id in [1, 3, 4] {
            assert!(a.remove(id));
        }
        assert_eq!(b.remove_many(&[1, 3, 4]), 3);
        // Duplicates + dead + out-of-range ids are skipped and counted once.
        assert_eq!(b.remove_many(&[1, 3, 9, 99]), 0);
        assert_eq!(a.live_count(), b.live_count());
        assert_eq!(a.live_count(), 3);
        let cfg = WangMatchConfig::default();
        for q in &refs {
            assert_eq!(a.query(q, &cfg), b.query(q, &cfg));
        }
        // Survivor ids are stable across both paths.
        let (id, res) = b.query(&refs[5], &cfg).expect("survivor must match");
        assert_eq!(id, 5);
        assert!(res.is_match);
    }

    #[test]
    fn haitsma_batch_apis_agree_with_looped() {
        let mk_h = |seed: u32| {
            mk_haitsma_fp(
                &(0..600)
                    .map(|i| (i as u32).wrapping_mul(seed))
                    .collect::<Vec<_>>(),
                78.125,
            )
        };
        let refs: Vec<HaitsmaFingerprint> =
            [7919, 2_654_435_761, 1_030_301, 40503].map(mk_h).to_vec();
        let mut a = HaitsmaIndex::build(&refs[..1], 100);
        let ids = a.insert_many(&refs[1..], 100);
        assert_eq!(ids, vec![1, 2, 3]);
        assert_eq!(a.live_count(), 4);
        assert_eq!(a.remove_many(&[0, 2, 2, 99]), 2);
        assert_eq!(a.live_count(), 2);
        let cfg = HaitsmaMatchConfig {
            min_overlap_frames: 256,
            ..Default::default()
        };
        let (id, res) = a.query(&refs[1], &cfg).expect("survivor must match");
        assert_eq!(id, 1);
        assert!(res.is_match);
    }

    #[test]
    fn panako_batch_apis_agree_with_looped() {
        let triples: Vec<(u32, u32, u32)> = (0..10u32)
            .map(|i| (i * 40 + 10, i * 40 + 20, i * 40 + 30))
            .collect();
        let refs = alloc::vec![
            mk_panako_fp(&triples, 0),
            mk_panako_fp(&triples, 1_000),
            mk_panako_fp(&triples, 2_000),
        ];
        let mut a = PanakoIndex::build(&refs[..1], 100);
        let ids = a.insert_many(&refs[1..], 100);
        assert_eq!(ids, vec![1, 2]);
        assert_eq!(a.remove_many(&[0]), 1);
        assert_eq!(a.live_count(), 2);
        let cfg = PanakoMatchConfig {
            min_votes: 3,
            min_prominence: 1.0,
            min_score: 0.05,
            ..Default::default()
        };
        let (id, res) = a.query(&refs[2], &cfg).expect("survivor must match");
        assert_eq!(id, 2);
        assert!(res.is_match);
    }

    // ── Deterministic candidate ordering (ascending ref id) ──
    //
    // Two identical references both score perfectly; the query must
    // always return ref 0, not whichever reference the hasher's
    // iteration order happens to visit first.

    #[test]
    fn wang_index_tie_resolves_to_lowest_ref_id() {
        let fp = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 0);
        let refs = alloc::vec![fp.clone(), fp.clone()];
        let index = WangIndex::build(&refs, 100);
        for _ in 0..8 {
            let (id, res) = index.query(&fp, &WangMatchConfig::default()).unwrap();
            assert_eq!(id, 0);
            assert!(res.is_match);
            assert!((res.score - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn haitsma_index_tie_resolves_to_lowest_ref_id() {
        let frames: alloc::vec::Vec<u32> = (0..600)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761))
            .collect();
        let fp = mk_haitsma_fp(&frames, 78.125);
        let refs = alloc::vec![fp.clone(), fp.clone()];
        let index = HaitsmaIndex::build(&refs, 1_000);
        for _ in 0..8 {
            let (id, res) = index.query(&fp, &HaitsmaMatchConfig::default()).unwrap();
            assert_eq!(id, 0);
            assert!(res.is_match);
        }
    }

    #[test]
    fn panako_index_tie_resolves_to_lowest_ref_id() {
        let fp = mk_panako_fp(
            &[
                (10, 15, 20),
                (30, 35, 40),
                (50, 55, 60),
                (70, 75, 80),
                (90, 95, 100),
                (110, 115, 120),
            ],
            0,
        );
        let refs = alloc::vec![fp.clone(), fp.clone()];
        let index = PanakoIndex::build(&refs, 100);
        for _ in 0..8 {
            let (id, res) = index.query(&fp, &PanakoMatchConfig::default()).unwrap();
            assert_eq!(id, 0);
            assert!(res.is_match);
        }
    }

    // ── WangIndex prominence parity with the dense matcher ──
    //
    // A peak with a wide, sparse background must score the same
    // *acceptance* in the index as in `WangMatcher`. Pre-fix, the index
    // divided the background by occupied bins only, understating
    // prominence and rejecting matches the 1:1 matcher accepts.

    #[test]
    fn wang_index_prominence_parity_with_matcher_on_sparse_background() {
        use crate::classical::WangHash;

        // Reference: 8 aligned anchors at t≈1000, two decoys at t=2500
        // and t=600.
        let mut ref_hashes = alloc::vec![
            WangHash {
                hash: 200,
                t_anchor: 2_500,
            },
            WangHash {
                hash: 201,
                t_anchor: 600,
            },
        ];
        for i in 0..8_u32 {
            ref_hashes.push(WangHash {
                hash: 100 + i,
                t_anchor: 1_000 + i * 10,
            });
        }
        let reference = WangFingerprint {
            hashes: ref_hashes,
            frames_per_sec: 62.5,
        };

        // Query: the same 8 anchors at δ=+100, plus the two decoy
        // matches at δ=+2_500 and δ=−1_800 — a deliberately wide,
        // sparse offset spread.
        let mut q_hashes = alloc::vec![
            WangHash {
                hash: 200,
                t_anchor: 0,
            },
            WangHash {
                hash: 201,
                t_anchor: 2_400,
            },
        ];
        for i in 0..8_u32 {
            q_hashes.push(WangHash {
                hash: 100 + i,
                t_anchor: 900 + i * 10,
            });
        }
        let query = WangFingerprint {
            hashes: q_hashes,
            frames_per_sec: 62.5,
        };

        let matcher = WangMatcher::new(WangMatchConfig::default());
        let m = matcher.match_one(&query, &reference);
        assert!(m.is_match, "dense matcher accepts the wide-sparse case");

        let index = WangIndex::build(&[reference], 100);
        let (id, r) = index
            .query(&query, &WangMatchConfig::default())
            .expect("index must accept what the dense matcher accepts");
        assert_eq!(id, 0);
        assert!(r.is_match);
        // Both paths report the same winning offset.
        assert_eq!(r.offset.frames, m.offset.frames);
    }

    // ── HaitsmaIndex top-k delta verification ──
    //
    // A repeated motif concentrates exact LUT hits at a WRONG offset;
    // the true alignment has fewer exact frames but far better BER.
    // Pre-fix, only the most-hit delta was verified → no match.

    #[test]
    fn haitsma_index_verifies_beyond_the_most_hit_delta() {
        struct XorShift(u32);
        impl XorShift {
            fn next(&mut self) -> u32 {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 17;
                self.0 ^= self.0 << 5;
                self.0
            }
        }

        let mut rng = XorShift(0x1D3A_5EED);
        let r: alloc::vec::Vec<u32> = (0..900).map(|_| rng.next() | 1).collect();

        // Query: 50-frame exact motif at δ=+100 (high LUT hits, BER≈0.44
        // over the 450-frame overlap → rejected), plus a 400-frame
        // window at δ=+400 lightly flipped (few exact frames, low BER
        // → the true match: Q[50+j] = flip(R[450+j]) ⇒ δ = 400).
        let mut q: alloc::vec::Vec<u32> = r[100..150].to_vec();
        for (j, &src) in r[450..850].iter().enumerate() {
            // Keep exactly three bit-exact frames so the LUT discovers
            // δ=+400; flip a handful of bits everywhere else.
            let exact = j == 0 || j == 100 || j == 200;
            let mut v = src;
            if !exact {
                // One flipped bit per frame: ~3% BER in the window, so
                // the total over the 450-frame overlap is ≈ (50·0.5 +
                // 400·0.03)/450 ≈ 0.08 — well under the decoy's ≈0.44.
                v ^= 1 << (rng.next() % 32);
            }
            q.push(v);
        }
        assert_eq!(q.len(), 450);

        let query = mk_haitsma_fp(&q, 78.125);
        let reference = mk_haitsma_fp(&r, 78.125);
        let index = HaitsmaIndex::build(&[reference], 1_000);

        let (id, res) = index
            .query(&query, &HaitsmaMatchConfig::default())
            .expect("true alignment (δ=400, low BER) must be found beyond the top-hit δ=100");
        assert_eq!(id, 0);
        assert!(res.is_match);
        assert_eq!(res.offset.frames, 400);
        // Score reflects the lightly-flipped window (BER ≈ 0.08),
        // not the ≈0.44-BER decoy overlap (which would score ≈ 0.56).
        assert!(
            res.score > 0.85,
            "score {} should reflect the low-BER alignment",
            res.score
        );
    }

    // ── Vote capping under flooded input ──
    //
    // `query` gathers votes into one flat list capped at
    // `MAX_VOTES_PER_QUERY` (audit 67-6, §4.1 #6), so a hostile catalog
    // cannot make a single query allocate without bound. The cap must not
    // change which reference wins, and results must stay deterministic.

    /// A query whose hashes all collide with every reference must still
    /// return a deterministic winner and not panic.
    ///
    /// Every reference is built from the same hash list, so the vote list is
    /// maximally dense (one vote per reference per query hash) and every
    /// candidate ties on raw votes. The tie-break must land on the lowest
    /// reference id, exactly as it does without flooding.
    #[test]
    fn wang_index_query_is_deterministic_under_hash_flooding() {
        use crate::classical::WangHash;

        // 64 references that all share the same 256 hashes → every candidate
        // accumulates 256 votes, a total of 16_384 votes in the flat list.
        let shared: Vec<WangHash> = (0..256_u32)
            .map(|i| WangHash {
                hash: i,
                t_anchor: i * 4,
            })
            .collect();
        let fp = WangFingerprint {
            hashes: shared.clone(),
            frames_per_sec: 62.5,
        };
        let refs: Vec<WangFingerprint> = (0..64).map(|_| fp.clone()).collect();
        let index = WangIndex::build(&refs, 1_000_000);

        let cfg = WangMatchConfig::default();
        let first = index.query(&fp, &cfg);
        assert!(
            first.is_some(),
            "a self-match among identical refs must match"
        );
        let (ref_id, res) = first.expect("checked Some");
        assert_eq!(
            ref_id, 0,
            "all refs tie, so the lowest id must win (got {ref_id})"
        );
        assert!(res.is_match);

        // Repeated queries must agree exactly (no hasher-order dependence).
        for run in 0..8 {
            let again = index.query(&fp, &cfg).expect("repeat query matches");
            assert_eq!(again.0, ref_id, "ref_id changed on run {run}");
            assert_eq!(
                again.1.score, res.score,
                "score changed on run {run} under flooding"
            );
            assert_eq!(again.1.offset.frames, res.offset.frames);
        }
    }

    // ── Frame-rate contract on the index paths ──
    //
    // `matching::mod` documents that a mismatched `frames_per_sec`
    // returns `MatchResult::NONE` in all builds. The matchers enforce it;
    // the indexes must too, or they return a match the 1:1 path rejects
    // (and convert `offset.ms` with the wrong rate).

    #[test]
    fn index_query_rejects_frame_rate_mismatch() {
        use crate::classical::{HaitsmaFingerprint, PanakoFingerprint, PanakoHash, WangHash};

        // Wang
        let wang = WangFingerprint {
            hashes: (0..8_u32)
                .map(|i| WangHash {
                    hash: i,
                    t_anchor: 10 + i * 10,
                })
                .collect(),
            frames_per_sec: 62.5,
        };
        let wang_q = WangFingerprint {
            hashes: wang.hashes.clone(),
            frames_per_sec: 31.25,
        };
        let widx = WangIndex::build(core::slice::from_ref(&wang), 100);
        assert!(
            widx.query(&wang_q, &WangMatchConfig::default()).is_none(),
            "WangIndex must reject an fps mismatch"
        );

        // Haitsma
        let haitsma = HaitsmaFingerprint {
            frames: (0..200_u32)
                .map(|i| i.wrapping_mul(2_654_435_761))
                .collect(),
            frames_per_sec: 78.125,
        };
        let haitsma_q = HaitsmaFingerprint {
            frames: haitsma.frames.clone(),
            frames_per_sec: 39.0625,
        };
        let hidx = HaitsmaIndex::build(&[haitsma], 1_000);
        assert!(
            hidx.query(&haitsma_q, &HaitsmaMatchConfig::default())
                .is_none(),
            "HaitsmaIndex must reject an fps mismatch"
        );

        // Panako
        let panako = PanakoFingerprint {
            hashes: (0..10_u32)
                .map(|i| PanakoHash {
                    hash: 500 + i,
                    t_anchor: 100 + i * 10,
                    t_b: 105 + i * 10,
                    t_c: 110 + i * 10,
                })
                .collect(),
            frames_per_sec: 62.5,
        };
        let panako_q = PanakoFingerprint {
            hashes: panako.hashes.clone(),
            frames_per_sec: 31.25,
        };
        let pidx = PanakoIndex::build(&[panako], 100);
        assert!(
            pidx.query(&panako_q, &PanakoMatchConfig::default())
                .is_none(),
            "PanakoIndex must reject an fps mismatch"
        );
    }

    /// `match_best`, `match_ranked`, and (with `rayon`) `par_match_best`
    /// must agree on a score-1.0 tie broken by prominence. The old
    /// perfect-score early exit in `match_best` returned the first 1.0 hit
    /// while `par_match_best` returned the higher-prominence one.
    #[test]
    fn match_best_agrees_with_ranked_on_perfect_score_ties() {
        use crate::classical::WangHash;

        let query = WangFingerprint {
            hashes: (0..8_u32)
                .map(|i| WangHash {
                    hash: i,
                    t_anchor: i * 10,
                })
                .collect(),
            frames_per_sec: 62.5,
        };
        // ref0: exact copy plus 99 scattered decoys (lower prominence).
        let mut ref0_hashes: alloc::vec::Vec<WangHash> = query.hashes.clone();
        for i in 0..99_u32 {
            ref0_hashes.push(WangHash {
                hash: 1_000 + i,
                t_anchor: 5_000 + i * 37,
            });
        }
        let ref0 = WangFingerprint {
            hashes: ref0_hashes,
            frames_per_sec: 62.5,
        };
        // ref1: exact copy only (higher prominence).
        let ref1 = WangFingerprint {
            hashes: query.hashes.clone(),
            frames_per_sec: 62.5,
        };

        let matcher = WangMatcher::new(WangMatchConfig::default());
        let ranked = match_ranked(&matcher, &query, &[ref0.clone(), ref1.clone()]);
        let best = match_best(&matcher, &query, &[ref0, ref1]).expect("a reference matches");
        assert_eq!(
            best.0, ranked[0].0,
            "match_best must return the same winner as match_ranked"
        );
        assert_eq!(best.1, ranked[0].1);
    }

    /// `is_empty_catalog()` tracks removals; key-space `is_empty()` follows
    /// once removal drops the emptied keys.
    #[test]
    fn is_empty_catalog_tracks_removals() {
        use crate::classical::WangHash;

        let fp = WangFingerprint {
            hashes: (0..8_u32)
                .map(|i| WangHash {
                    hash: i,
                    t_anchor: i * 10,
                })
                .collect(),
            frames_per_sec: 62.5,
        };
        let mut idx = WangIndex::build(&[fp], 100);
        assert!(!idx.is_empty_catalog());
        assert!(idx.remove(0));
        assert_eq!(idx.live_count(), 0);
        assert!(idx.is_empty_catalog());
        // Removal drops the keys it empties, so the key space empties too.
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
    }
}
