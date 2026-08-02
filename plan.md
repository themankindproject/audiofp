# audiofp — Matching Subsystem Plan

**Status:** implemented on `feat/matching-subsystem` (Phases 0–2, 4–6);
**Phase 3 (Panako 2-D Hough) is still a stub** — `PanakoMatcher` /
`PanakoIndex` always return non-match / empty. Hot-path maps use
`HashMap` under `std` (`BTreeMap` without `std`).
**Scope:** add an in-memory **matching / identification** layer on top of the
existing fingerprinters.
**Explicit constraint:** *matching only* — **no storage, no persistence, no
serialization format, no database adapters, no on-disk index.** Everything
operates on in-memory fingerprints and returns an in-memory result.

Target crate version at time of writing: `0.3.7` · edition 2024 · MSRV `1.93.0`.

---

## Table of contents

1. [Motivation & the gap](#1-motivation--the-gap)
2. [Goals & non-goals](#2-goals--non-goals)
3. [Facts the design depends on](#3-facts-the-design-depends-on)
4. [Module architecture](#4-module-architecture)
5. [Common types & the `Matcher` trait](#5-common-types--the-matcher-trait)
6. [Wang matcher — offset-histogram voter](#6-wang-matcher--offset-histogram-voter)
7. [Panako matcher — tempo-invariant 2-D Hough](#7-panako-matcher--tempo-invariant-2-d-hough)
8. [Haitsma matcher — BER sliding + sub-fingerprint LUT](#8-haitsma-matcher--ber-sliding--sub-fingerprint-lut)
9. [Neural matcher — cosine similarity](#9-neural-matcher--cosine-similarity)
10. [1:N matching & optional transient index](#10-1n-matching--optional-transient-index)
11. [Public API surface & usage](#11-public-api-surface--usage)
12. [Phased implementation plan](#12-phased-implementation-plan)
13. [Testing & threshold calibration](#13-testing--threshold-calibration)
14. [Performance budget](#14-performance-budget)
15. [Risks & open decisions](#15-risks--open-decisions)
16. [Appendix A — secondary findings (non-matching)](#appendix-a--secondary-findings-non-matching)

---

## 1. Motivation & the gap

The crate extracts fingerprints but has **no matching logic in the library**.
The only comparison code is `examples/match_two_files.rs`, which is unsound as a
template:

```rust
// examples/match_two_files.rs — current behaviour
wang.extract(buf)?.hashes.into_iter().map(|h| h.hash).collect::<HashSet<u32>>()
```

- It **discards `t_anchor`** — throwing away all temporal alignment, which is
  the whole point of landmark fingerprinting. Two unrelated tracks that share
  common landmark hashes (very likely for similar instrumentation) score as
  "same recording."
- It **dedups by hash value** (`HashSet<u32>`), collapsing repeated landmarks
  and distorting the Jaccard ratio.
- Its own doc comment concedes the real method is missing: *"For real matching
  at scale you'd use `t_anchor` to verify same-offset collisions and apply a
  histogram-of-time-deltas voter."*

`future.md` lists a CLI `match` command (§5.1) and vector-DB adapters (§8.1) but
**no matcher library module**. This plan fills that gap with a dependency-free,
`no_std + alloc` matching module.

---

## 2. Goals & non-goals

### Goals

- Given a **query** fingerprint and a **reference** fingerprint (same
  algorithm), decide **is this the same recording**, with a normalized
  confidence score and the estimated **time offset** (and **time-scale** for
  Panako).
- One matcher per algorithm: Wang, Panako, Haitsma, Neural.
- 1:1 (`match_one`) and 1:N (`match_best`, `match_ranked`) — all in memory.
- Statistically sound: robust to the false positives that plague naive
  hash-set overlap.
- Match the crate's conventions: `no_std + alloc`, zero new dependencies,
  `Config` + `Default`, `AfpError`/`Result`, `#[must_use]`, doctests,
  determinism.

### Non-goals (explicitly out of scope)

- ❌ Any persistence / serialization / wire format for fingerprints or indexes.
- ❌ Any database adapter (FAISS, hnswlib, RocksDB, sqlite, Postgres, Redis).
- ❌ On-disk / memory-mapped index.
- ❌ A CLI (that is roadmap §5.1, separate).
- ❌ Changes to the fingerprint *extractors* (matching consumes their existing
  output types unchanged).

> The **optional transient in-memory index** in §10 is a matching *accelerator*
> that lives in RAM for the lifetime of the value and is never serialized. It is
> not "storage" in the excluded sense.

---

## 3. Facts the design depends on

These are read directly from the source and are the contract the matcher relies
on. **If any extractor changes these, the matcher must change too.**

### 3.1 Fingerprint output types

| Type | Definition | Sorted by |
|---|---|---|
| `WangFingerprint` | `{ hashes: Vec<WangHash>, frames_per_sec: f32 }` | `(t_anchor, hash)` |
| `PanakoFingerprint` | `{ hashes: Vec<PanakoHash>, frames_per_sec: f32 }` | `(t_anchor, t_b, t_c, hash)` |
| `HaitsmaFingerprint` | `{ frames: Vec<u32>, frames_per_sec: f32 }` | frame order (n=1..) |
| `NeuralFingerprint` | `{ embeddings: Vec<NeuralEmbedding>, embedding_dim, frames_per_sec }` | input order |

```rust
// classical/wang.rs
pub struct WangHash   { pub hash: u32, pub t_anchor: u32 }
// classical/panako.rs
pub struct PanakoHash { pub hash: u32, pub t_anchor: u32, pub t_b: u32, pub t_c: u32 }
// neural/embedder.rs
pub struct NeuralEmbedding { pub vector: Vec<f32>, pub t_start: TimestampMs }
```

### 3.2 Hash bit layouts

```text
WangHash::hash (32 bits, MSB first)
[31..23] f_a_q  9 bits   anchor freq bucket (bin*512/513)
[22..14] f_b_q  9 bits   target freq bucket
[13.. 0] Δt    14 bits   frames anchor→target, clamped 1..=0x3FFF

PanakoHash::hash (32 bits, MSB first)  — tempo-invariant via β
[31..30] sign       2 bits
[29..28] mag_order  2 bits
[27..23] β          5 bits   round((t_c - t_b)/(t_c - t_a) * 31), 0..=31
[22..15] Δf_ab      8 bits   signed, clamped ±127
[14.. 7] Δf_bc      8 bits   signed, clamped ±127
[ 6.. 0] reserved   7 bits   zero

Haitsma frame (32 bits, "MSB-zero")
bit 31 → band 0, bit 0 → band 31
```

### 3.3 Frame timing (needed to convert frame offsets → milliseconds)

| Algorithm | Sample rate | n_fft / hop | frames/s | ms per frame |
|---|---|---|---|---|
| Wang | 8 kHz | 1024 / 128 | 62.5 | **16.0** |
| Panako | 8 kHz | 1024 / 128 | 62.5 | **16.0** |
| Haitsma | 5 kHz | 2048 / 64 | 78.125 | **12.8** |
| Neural | model (16 kHz default) | — | `1/hop_secs` (default 1.0) | `hop_secs*1000` |

**Conversion:** `ms = frames * 1000 / frames_per_sec`. Do this with the
fingerprint's own `frames_per_sec` (never hardcode 62.5/78.125) so custom
configs stay correct.

### 3.4 Invariants

- Fingerprints are **deterministic** and **sorted** (see 3.1). The Wang/Panako
  sort by `t_anchor` first is convenient (query hashes are time-ordered).
- Silence → empty hashes / all-zero Haitsma frames. Matchers must treat empty
  inputs as `MatchResult::NONE`, never divide by zero.
- All hash structs are `bytemuck::Pod`; `Peak` is 12-byte `repr(C)`.

---

## 4. Module architecture

New module **`src/matching/`**, always compiled (like `dsp`/`classical`),
`no_std + alloc`, **no new dependencies**. Neural sub-matcher gated on the
existing `neural` feature; parallel 1:N gated on the existing `rayon` feature.

```
src/matching/
├── mod.rs        // MatchResult, TimeOffset, Matcher trait, score helpers, re-exports
├── wang.rs       // WangMatcher      — offset-histogram voter
├── panako.rs     // PanakoMatcher    — 2-D (scale, offset) Hough + RANSAC
├── haitsma.rs    // HaitsmaMatcher   — BER sliding + sub-fingerprint LUT
├── neural.rs     // NeuralMatcher    — cosine (cfg: neural)
└── index.rs      // WangIndex/PanakoIndex — optional transient in-memory 1:N accelerator
```

`src/lib.rs` additions:

```rust
pub mod matching;
pub use matching::{MatchResult, TimeOffset, Matcher};
```

No `Cargo.toml` dependency changes. (Optionally introduce a `matching` feature,
default-on, if we ever want minimal builds to drop it — not required now since
it pulls nothing in.)

---

## 5. Common types & the `Matcher` trait

`src/matching/mod.rs`:

```rust
/// Signed time offset of the query's start relative to the reference's start.
/// Negative = the query begins before the reference.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TimeOffset {
    /// Offset in reference STFT frames (exact, native precision).
    pub frames: i64,
    /// Offset in milliseconds (derived via the fingerprint's frames_per_sec).
    pub ms: i64,
}

impl TimeOffset {
    #[must_use]
    pub fn from_frames(frames: i64, frames_per_sec: f32) -> Self {
        let ms = (frames as f64 * 1000.0 / frames_per_sec as f64).round() as i64;
        Self { frames, ms }
    }
    pub const ZERO: TimeOffset = TimeOffset { frames: 0, ms: 0 };
}

/// Outcome of matching a query fingerprint against one reference.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MatchResult {
    /// True iff the score cleared the configured decision threshold.
    pub is_match: bool,
    /// Normalized confidence in [0.0, 1.0], algorithm-agnostic (higher = better).
    pub score: f32,
    /// Raw aligned-evidence count (landmark votes / aligned Haitsma frames).
    pub votes: u32,
    /// Peak prominence: peak evidence ÷ background. High = sharp spike
    /// (true match); ~1 = flat (random collisions). The false-positive guard.
    pub prominence: f32,
    /// Estimated alignment of the query within the reference.
    pub offset: TimeOffset,
    /// Estimated time-scale (query_seconds / ref_seconds). 1.0 for algorithms
    /// with no tempo model (Wang, Haitsma, neural). Panako fills this in.
    pub time_scale: f32,
}

impl MatchResult {
    pub const NONE: MatchResult = MatchResult {
        is_match: false, score: 0.0, votes: 0, prominence: 0.0,
        offset: TimeOffset::ZERO, time_scale: 1.0,
    };
}

/// One matcher per fingerprinting algorithm.
pub trait Matcher {
    type Fingerprint;                       // e.g. WangFingerprint
    type Config: Clone + Send + Sync;

    fn new(cfg: Self::Config) -> Self;
    fn config(&self) -> &Self::Config;

    /// Match a query against a single reference. Fully in-memory. Infallible:
    /// empty/degenerate inputs yield `MatchResult::NONE`.
    fn match_one(&self, query: &Self::Fingerprint, reference: &Self::Fingerprint) -> MatchResult;
}
```

**Cross-cutting rules**

- `f32` is not `Ord`; use `a.partial_cmp(&b).unwrap_or(Ordering::Equal)` for all
  score sorting (matches the crate's existing NaN-safe convention).
- `match_one` `debug_assert!`s `query.frames_per_sec == reference.frames_per_sec`;
  in release it proceeds using the reference's rate.
- No panics on any input (uphold `tests/property.rs` guarantees). No
  `partial_cmp().unwrap()`.

---

## 6. Wang matcher — offset-histogram voter

The canonical Shazam alignment. Matching landmark hashes must agree on a single
**constant time offset**; random collisions scatter across offsets while a true
match spikes at one.

### 6.1 Config

```rust
#[derive(Clone, Debug)]
pub struct WangMatchConfig {
    /// Consolidate votes within ±N frames of the peak (framing jitter). Default 1.
    pub offset_tolerance_frames: u32,
    /// Absolute floor on peak vote count. Default 5.
    pub min_votes: u32,
    /// Decision threshold on the normalized score in [0,1]. Default 0.15.
    pub min_score: f32,
    /// Peak ÷ background floor (false-positive guard). Default 5.0.
    pub min_prominence: f32,
    /// Skip hashes whose reference posting list exceeds this ("stop-hashes").
    /// Kills the pure-tone/silence collision explosion. Default 100.
    pub max_postings_per_hash: u32,
}
impl Default for WangMatchConfig {
    fn default() -> Self {
        Self { offset_tolerance_frames: 1, min_votes: 5, min_score: 0.15,
               min_prominence: 5.0, max_postings_per_hash: 100 }
    }
}
```

### 6.2 Algorithm

1. **Index the reference** — `HashMap<u32 /*hash*/, Vec<u32 /*t_anchor*/>>`.
   Drop hashes whose posting list length `> max_postings_per_hash` (TF-IDF-style
   stop-hash removal). `O(R)`.
2. **Vote** — for each query `(h, t_q)`, for each ref `t_r ∈ index[h]`, compute
   `δ = t_r as i64 - t_q as i64` and increment a **dense histogram**.
   `O(Q · avg_postings)`.
   - δ range: `[-(max_query_t_anchor), max_ref_t_anchor]`.
   - Histogram = `Vec<u32>` of length `range+1`, index `= δ - δ_min` → O(1) bump,
     no hashing on the hot path.
3. **Consolidate** — box-convolve with a `±offset_tolerance_frames` window so
   jitter-split votes coalesce; then take the max bin `(δ*, peak_votes)`. `O(range)`.
4. **Prominence** — `prominence = peak_votes / (mean_of_nonpeak_bins + 1.0)`.
   Equivalent z-score form: `(peak - μ)/σ` over bins. This is the discriminator
   between a true spike and flat random collisions.
5. **Score & decide**
   - Estimate the aligned span from `min/max t_q` that voted at `δ*`; let
     `denom = max(1, query_hashes_within_span)`.
   - `score = clamp(peak_votes as f32 / denom as f32, 0.0, 1.0)`.
   - `is_match = peak_votes >= min_votes && score >= min_score && prominence >= min_prominence`.
   - `offset = TimeOffset::from_frames(δ*, frames_per_sec)`, `time_scale = 1.0`.

### 6.3 Pseudocode

```rust
fn match_one(&self, q: &WangFingerprint, r: &WangFingerprint) -> MatchResult {
    if q.hashes.is_empty() || r.hashes.is_empty() { return MatchResult::NONE; }

    // 1. index reference (skip stop-hashes)
    let mut index: HashMap<u32, Vec<u32>> = HashMap::new();
    for h in &r.hashes { index.entry(h.hash).or_default().push(h.t_anchor); }
    index.retain(|_, v| v.len() as u32 <= self.cfg.max_postings_per_hash);

    // 2. dense offset histogram
    let q_max = q.hashes.iter().map(|h| h.t_anchor).max().unwrap() as i64;
    let r_max = r.hashes.iter().map(|h| h.t_anchor).max().unwrap() as i64;
    let (dmin, dmax) = (-q_max, r_max);
    let mut hist = vec![0u32; (dmax - dmin + 1) as usize];
    for h in &q.hashes {
        if let Some(list) = index.get(&h.hash) {
            for &tr in list {
                let d = tr as i64 - h.t_anchor as i64;
                hist[(d - dmin) as usize] += 1;
            }
        }
    }

    // 3. consolidate ±tol, find peak
    // 4. prominence
    // 5. score + decision  (see 6.2)
}
```

### 6.4 Complexity & memory

- Time `O(R + Q + range)`; sub-millisecond for song-length inputs.
- Memory: dense histogram ≈ 4 bytes/frame → 4 min @ 62.5 fps ≈ **60 KB**, transient.
- If `range` is ever huge (hour-long refs), fall back to `HashMap<i64,u32>`; a
  4-byte dense array is fine up to ~ tens of minutes.

---

## 7. Panako matcher — tempo-invariant 2-D Hough

Panako's β makes the **hash values** survive ±5 % time-stretch, but under scale
`s` the alignment is a **line** `t_ref ≈ s·t_query + b`, not a constant offset.
So vote in 2-D `(scale, offset)` space.

### 7.1 Config

```rust
#[derive(Clone, Debug)]
pub struct PanakoMatchConfig {
    pub scale_min: f32,        // 0.80
    pub scale_max: f32,        // 1.25
    pub scale_bins: u32,       // 24  (~2% resolution)
    pub offset_tolerance_frames: u32, // 1
    pub min_votes: u32,        // 5
    pub min_score: f32,        // 0.15
    pub min_prominence: f32,   // 5.0
    pub max_postings_per_hash: u32, // 100
    pub ransac_refine: bool,   // true
}
```

### 7.2 Algorithm

1. **Index reference** by hash → `Vec<(t_anchor, t_b, t_c)>`.
2. For each matched (query triple, ref triple) with equal hash value:
   - **Local scale** from triple spans:
     `s = (t_c_ref − t_a_ref) as f32 / (t_c_query − t_a_query) as f32`
     (same acoustic triple stretches proportionally). Guard the divisor `> 0`.
   - **Predicted offset** `b = t_a_ref as f32 − s * t_a_query as f32`.
   - Vote into 2-D histogram `[scale_bin(s)][offset_bin(b)]` if
     `s ∈ [scale_min, scale_max]`.
3. **Peak** of the 2-D histogram → coarse `(s*, b*, votes)`.
4. **RANSAC refine** (if `ransac_refine`): fit a line through the matched
   `(t_query, t_ref)` pairs near the peak → continuous `s`, `b`, inlier count →
   final `votes`.
5. Score/decision mirror Wang; `time_scale = s*`,
   `offset = TimeOffset::from_frames(round(b*), frames_per_sec)`.

### 7.3 Complexity

`O(R + matched_pairs + scale_bins·offset_bins)`. Heavier than Wang (2-D grid)
but still fast for song-length inputs. Validate against
`tests/property.rs::panako_tempo_robustness`.

---

## 8. Haitsma matcher — BER sliding + sub-fingerprint LUT

Haitsma is a dense per-frame 32-bit code; matching is **bit-error-rate (BER)
minimization** over alignments.

### 8.1 Config

```rust
#[derive(Clone, Debug)]
pub struct HaitsmaMatchConfig {
    pub max_ber: f32,             // 0.35 (paper's block threshold)
    pub min_overlap_frames: u32,  // 256  (~ one sub-fingerprint block)
    pub use_lut: bool,            // true (LUT acceleration for long refs)
    pub probe_bit_flips: u8,      // 0 (raise to 1–2 for noisier inputs)
}
```

### 8.2 Two tiers

**Exact BER (small inputs / verification):** for each offset δ,
`hamming = Σ (q[i] ^ r[i+δ]).count_ones()` (→ hardware POPCNT/CNT,
auto-vectorizable over `u64` pairs). **Early-abort** the inner accumulation once
`hamming` exceeds the best-so-far — most offsets die in a few frames.
`BER = hamming / (32 * overlap)`; best δ = `argmin`.

**Sub-fingerprint LUT (scale / long references):** build
`HashMap<u32, Vec<pos>>` over reference frames. When `BER < ~0.35`, at least one
query frame is bit-exact (Haitsma's key property), so probe each query frame's
exact `u32` (plus optional 1–2 bit-flip probes) → candidate offsets → run exact
BER only there. Turns `O(Q·R)` into `O(Q + candidates·overlap)`.

### 8.3 Output

- `score = 1.0 - min_ber` (higher = better, consistent with the others).
- `is_match = min_ber <= max_ber && overlap >= min_overlap_frames`.
- `offset = TimeOffset::from_frames(δ, frames_per_sec)`; `time_scale = 1.0`
  (Haitsma has no tempo tolerance).
- `votes = overlap_frames`; `prominence = median_ber / (min_ber + ε)` (spread
  between the best and typical alignment).

### 8.4 Popcount helper

```rust
#[inline]
fn hamming_u32(a: u32, b: u32) -> u32 { (a ^ b).count_ones() }
// Batch over u64 pairs for the inner loop; count_ones() lowers to POPCNT/CNT.
```

---

## 9. Neural matcher — cosine similarity

Embeddings are L2-normalized by default, so **cosine = dot product**.

### 9.1 Config

```rust
#[derive(Clone, Debug)]
pub struct NeuralMatchConfig {
    pub min_cosine: f32,        // 0.80 — MODEL DEPENDENT (documented as such)
    pub aggregation: Aggregation, // SlidingMax (default) | Global | Dtw
    pub assume_normalized: bool,  // true; if false, matcher normalizes internally
}
pub enum Aggregation { Global, SlidingMax, Dtw }
```

### 9.2 Aggregation modes

- **Global** — mean-pool query and reference to one vector each (renormalize),
  single cosine. Fast, coarse; good for whole-clip / cover similarity.
- **SlidingMax** (default) — slide the query embedding sequence over the
  reference; at each offset `j`, `mean_i cos(q_i, r_{j+i})`; take the max.
  Localizes a short query inside a long reference. `offset = argmax_j`.
- **Dtw** — dynamic time warping for tempo-flexible sequence matching (heavier;
  opt-in). May report a warp ratio as `time_scale`.

### 9.3 Output

- `score = clamp((best_cos + 1.0) * 0.5, 0.0, 1.0)` (or `max(0, best_cos)`).
- `is_match = best_cos >= min_cosine`.
- `offset` from the winning window's `t_start`; `time_scale = 1.0` (except Dtw).
- Complexity `O(Nq · Nr · D)`; windows are ~1 s so `Nq/Nr` are tiny → cheap.
- **Gated `#[cfg(feature = "neural")]`** (needs `NeuralFingerprint`).

---

## 10. 1:N matching & optional transient index

Free functions on top of `match_one`:

```rust
/// Best match across a set of references (iterates match_one).
pub fn match_best<M: Matcher>(m: &M, query: &M::Fingerprint, refs: &[M::Fingerprint])
    -> Option<(usize, MatchResult)>;

/// All references scored, sorted by score desc (ties → prominence desc).
pub fn match_ranked<M: Matcher>(m: &M, query: &M::Fingerprint, refs: &[M::Fingerprint])
    -> Vec<(usize, MatchResult)>;
```

- With **`rayon`** enabled, both parallelize across references
  (`refs.into_par_iter()`), reusing the existing feature. Parallel results must
  equal sequential (mirror `lib.rs::batch_parallel_produces_same_results_as_sequential`).

**Optional transient index** (`matching/index.rs`) — `WangIndex` / `PanakoIndex`:
ingest several references into one combined in-memory inverted index
(`hash → Vec<(ref_id, t_anchor)>`) so per-query cost is paid once across the set.

```rust
let index = WangIndex::build(&references);   // in RAM only
let hits  = index.query(&q, &cfg);           // Vec<(ref_id, MatchResult)>
// index dropped at end of scope — never serialized.
```

> **This is a matching accelerator, not storage.** It holds no file handles,
> defines no wire format, and is dropped with its owning scope. It exists purely
> so 1:N doesn't rebuild a per-reference index on every query.

---

## 11. Public API surface & usage

```rust
use audiofp::classical::Wang;
use audiofp::matching::{WangMatcher, WangMatchConfig, Matcher};
use audiofp::{AudioBuffer, Fingerprinter, SampleRate};

// extract (existing API, unchanged)
let mut wang = Wang::default();
let q = wang.extract(AudioBuffer::new(&query_pcm, SampleRate::HZ_8000))?;
let r = wang.extract(AudioBuffer::new(&ref_pcm,   SampleRate::HZ_8000))?;

// match (new API)
let matcher = WangMatcher::new(WangMatchConfig::default());
let m = matcher.match_one(&q, &r);
if m.is_match {
    println!("same recording (score {:.2}, offset {} ms, votes {})",
             m.score, m.offset.ms, m.votes);
}
```

1:N:

```rust
use audiofp::matching::match_ranked;
let refs: Vec<WangFingerprint> = /* … */;
for (id, res) in match_ranked(&matcher, &q, &refs).into_iter().take(5) {
    println!("#{id}: score {:.2} offset {} ms", res.score, res.offset.ms);
}
```

---

## 12. Phased implementation plan

Each phase is independently shippable and reviewable. Run after every phase:
`cargo fmt --all -- --check`, `cargo clippy --all-targets --all-features -- -D warnings`,
`cargo test --all-features`, and `cargo build --no-default-features` (no_std path).

| Phase | Deliverable | Files | Acceptance criteria |
|---|---|---|---|
| **0. Scaffold** | `TimeOffset`, `MatchResult`, `Matcher` trait, score/order helpers; wire `pub mod matching` | `matching/mod.rs`, `lib.rs` | builds on no_std+alloc; `cargo doc` clean; unit tests for `TimeOffset::from_frames` (incl. Haitsma 12.8 ms/frame) |
| **1. Wang** | `WangMatcher` + `WangMatchConfig` (dense histogram, ±tol consolidation, prominence, stop-hash cap) | `matching/wang.rs` | self-match `score≈1.0, offset=0`; time-shifted copy recovers offset within ±1 frame; unrelated pair `is_match=false`; reuse `robustness.rs` synth+noise+lowpass helpers |
| **2. Haitsma** | `HaitsmaMatcher` (exact BER + early-abort, then LUT + optional bit-flip probes) | `matching/haitsma.rs` | self-match `BER=0`; offset recovery; `haitsma_robust_to_lowpass/noise` corpora stay under `max_ber`; LUT path == exact path |
| **3. Panako** | `PanakoMatcher` (2-D Hough + optional RANSAC) | `matching/panako.rs` | recovers `(scale, offset)` on time-stretched copy; passes `property.rs::panako_tempo_robustness`; unrelated `is_match=false` |
| **4. Neural** | `NeuralMatcher` (cosine; Global + SlidingMax; Dtw optional) behind `neural` | `matching/neural.rs` | identical clips → cosine 1.0 via passthrough fixture in `neural/test_support.rs`; SlidingMax localizes a sub-clip |
| **5. 1:N** | `match_best`, `match_ranked`; `rayon` parallel; optional `WangIndex`/`PanakoIndex` | `matching/index.rs`, `matching/mod.rs` | ranking order correct; parallel == sequential; index query == pairwise `match_one` |
| **6. Integrate** | Rewrite `examples/match_two_files.rs` on `WangMatcher`; `benches/matching.rs`; `USAGE.md` + README section; mark `future.md` | `examples/`, `benches/`, docs | example prints score/offset; bench runs; clippy/fmt/test green |

Dependency order: **0 → 1 → {2,3,4} → 5 → 6**. Phases 2/3/4 are parallelizable
once 0 and 1 land (1 establishes the histogram/scoring patterns 2/3 reuse).

---

## 13. Testing & threshold calibration

### 13.1 Synthetic invariants (no corpus needed — do these first)

Reuse the existing helpers: `synth`, `add_noise`, `lowpass`, `inject_spikes`
(in `tests/robustness.rs`, `tests/property.rs`).

- **Self-match:** `match_one(fp, fp)` → `is_match`, `score` near max, `offset=0`,
  `time_scale=1.0`.
- **Offset recovery:** prepend `k` frames of silence/noise to the query; matcher
  must report `offset ≈ ±k` within tolerance.
- **Unrelated:** two independent synth signals → `is_match=false`, low
  prominence.
- **Monotonicity:** increasing added noise / lowpass severity → non-increasing
  score.
- **Determinism:** same inputs → identical `MatchResult` (bit-for-bit).
- **No panic:** NaN/∞/empty/one-frame inputs → `MatchResult::NONE`, never panic.
- **Panako tempo:** ±5 % stretch still matches and recovers `time_scale`.
- **Haitsma:** `BER=0` on self-match; LUT path equals exact path.

### 13.2 Property tests

- Wang/Panako: for any injected integer offset `k`, the peak bin equals `k`.
- Prominence of a true (shifted-copy) match strictly exceeds that of a random
  pair, across randomized signals.

### 13.3 Golden tests

Extend `tests/regression.rs` with one golden `(query, reference) → MatchResult`
per algorithm to lock scoring stability across releases (store the expected
`score`/`votes`/`offset`).

### 13.4 Threshold calibration (needs real audio — flag as follow-up)

Default `min_score` / `max_ber` / `min_cosine` / `min_prominence` are **starting
points**. Tune them against roadmap §3.1's real CC0 codec corpus (MP3@128k,
AAC@128k, Opus@32k round-trips) once it exists. **Do not over-fit thresholds to
synthetic signals.** Document this dependency in the matcher rustdoc.

---

## 14. Performance budget

| Matcher | Time | Memory | Notes |
|---|---|---|---|
| Wang | `O(R + Q + range)` | ~4 B/frame histogram (~60 KB/4 min) | sub-ms for songs |
| Panako | `O(R + pairs + scale_bins·offset_bins)` | 2-D grid (`scale_bins`×offset range) | heavier grid, still fast |
| Haitsma exact | `O(Q·R)` popcounts w/ early-abort | O(1) beyond inputs | LUT → `O(Q + cand·overlap)` |
| Neural SlidingMax | `O(Nq·Nr·D)` | O(1) beyond inputs | `Nq/Nr` tiny (≈ seconds) |

- No allocation in inner loops beyond the histogram/index built once per match.
- `rayon` 1:N scales across references; each `match_one` is independent.

---

## 15. Risks & open decisions

1. **`match_one` fallible vs infallible.** Chosen: infallible, returns
   `MatchResult::NONE` on empty/degenerate input; `debug_assert` on
   `frames_per_sec` mismatch. Alternative: return `Result` and error on mismatch.
   *Decision: infallible* (matching a query is not an error even when it fails).
2. **Dense vs sparse Wang histogram.** Dense `Vec<u32>` chosen for speed; add a
   sparse `HashMap` fallback only if hour-long references appear.
3. **Panako refinement.** RANSAC is optional (`ransac_refine`, default on). If it
   proves flaky, coarse 2-D Hough peak alone is the fallback.
4. **Haitsma LUT recall.** The "≥1 exact sub-fingerprint" assumption weakens near
   the BER threshold; `probe_bit_flips` mitigates. Keep exact-BER as the correctness
   oracle in tests.
5. **Neural thresholds are model-specific.** `min_cosine` default 0.8 is nominal;
   must be documented as model-dependent and left to the caller.
6. **`matching` feature flag?** Not needed now (zero deps). Revisit only if a
   downstream minimal build wants to exclude it.

---

## Appendix A — secondary findings (non-matching)

Out of scope for this plan, captured for follow-up:

**Correctness / template**
- `examples/match_two_files.rs` is statistically unsound (see §1). Phase 6
  replaces it with `WangMatcher`.

**Documentation drift**
- The `rayon` feature is real (`Cargo.toml` + `fingerprint_batch_parallel`) but
  **missing from both the README feature table and the `lib.rs` doc table**.
- `future.md` is stale: says version `0.2.1` / MSRV `1.85.0`; actual is `0.3.7`
  / `1.93.0` (README badge agrees at 1.93+). It also marks shipped items
  (neural §1.1, rayon batch §2.5) as future.

**Performance (extraction already strong; residual wins)**
- Neural batched inference (roadmap 1.1.1, P1) — 5–20× on small models.
- SIMD mel matvec (roadmap 1.1.2) — 2–4× on the neural front-end.
- `PeakPicker::pick` returns via `self.candidates.clone()` — one alloc+copy per
  `extract`; add a `pick_into(&mut Vec<Peak>)` for a fully alloc-free offline path.
- Classical `StreamingFingerprinter::push_with` uses the allocating default; the
  Wang/Panako/Haitsma streamers could override it to feed the callback directly
  from their internal buffers for a true zero-alloc streaming path.

**API note**
- `TimestampMs` is unsigned; offsets can be negative, hence the signed
  `TimeOffset` type in §5 rather than overloading `TimestampMs`.
