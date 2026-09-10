//! Allocation and heap-footprint regression tests for the 1:N index types.
//!
//! Two audit findings are pinned here:
//!
//! * **§4.2 #8 — `WangIndex::query` per-candidate allocations.** The query
//!   loop used to build five independent scratch structures *per candidate
//!   reference* (`bins`, `bin_vec`, `consolidated`, the plateau `Vec`, and
//!   `contrib_indices`). They are now hoisted out of the loop and `clear()`ed
//!   between candidates, so a warm query's allocation count is independent of
//!   how many candidates it scores.
//! * **§4.2 #10 — `estimated_bytes` double-counted map keys.** The key term
//!   was added once as `map.len() * size_of::<K>()` and again inside the
//!   per-slot term, over-reporting the footprint. The second test compares the
//!   reported figure against the heap actually held, so a regression in either
//!   direction fails.
//!
//! Counting allocations instead of timing makes both gates deterministic and
//! machine-independent — unlike a criterion bench, these cannot pass on a
//! quiet box and fail on a busy one.
//!
//! Requires `std` (custom global allocator). Incompatible with `mimalloc`,
//! which installs its own `#[global_allocator]`.

#![cfg(all(feature = "std", not(feature = "mimalloc")))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use audiofp::Fingerprinter;
use audiofp::SampleRate;
use audiofp::classical::HaitsmaFingerprint;
use audiofp::classical::{Haitsma, Wang, WangFingerprint};
use audiofp::matching::{HaitsmaIndex, WangIndex, WangMatchConfig};

/// Counting wrapper around [`System`]. Thread-local so parallel tests in this
/// binary never contaminate each other's deltas.
struct CountingAlloc;

thread_local! {
    static COUNT: Cell<usize> = const { Cell::new(0) };
    static LIVE_BYTES: Cell<i64> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: delegates to `System` with the caller's layout.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            // `try_with` skips counting during TLS init/teardown, where
            // counting would recurse or panic.
            let _ = COUNT.try_with(|c| c.set(c.get() + 1));
            let _ = LIVE_BYTES.try_with(|c| c.set(c.get() + layout.size() as i64));
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let _ = LIVE_BYTES.try_with(|c| c.set(c.get() - layout.size() as i64));
        // SAFETY: paired with the allocation this layout describes.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: delegates to `System` with the caller's arguments.
        let out = unsafe { System.realloc(ptr, layout, new_size) };
        if !out.is_null() {
            let _ = COUNT.try_with(|c| c.set(c.get() + 1));
            let _ =
                LIVE_BYTES.try_with(|c| c.set(c.get() + new_size as i64 - layout.size() as i64));
        }
        out
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn allocs() -> usize {
    COUNT.with(Cell::get)
}

fn live_bytes() -> i64 {
    LIVE_BYTES.with(Cell::get)
}

fn synth(seed: u32, sr: u32, secs: usize) -> Vec<f32> {
    let n = sr as usize * secs;
    let mut out = Vec::with_capacity(n);
    let mut x = seed.max(1);
    for i in 0..n {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        let noise = (x as i32 as f32) / (i32::MAX as f32) * 0.05;
        let t = i as f32 / sr as f32;
        out.push(
            0.5 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
                + 0.3 * (2.0 * std::f32::consts::PI * 1320.0 * t).sin()
                + noise,
        );
    }
    out
}

fn wang_fp(seed: u32, secs: usize) -> WangFingerprint {
    let samples = synth(seed, 8_000, secs);
    Wang::default()
        .extract(&samples, SampleRate::HZ_8000)
        .expect("wang extract")
}

/// A warm `WangIndex::query` must not allocate per candidate reference.
///
/// Measured on this workload (100 refs, 3 s query, warm):
///
/// | Revision | Allocations per query |
/// |---|---|
/// | before the §4.2 #8 fix | **369** |
/// | after (flat vote list + hoisted scratch) | **16** |
///
/// The bound is 50: roughly 3× the current behaviour for headroom, but far
/// below the 369 (and the 193 of the first, partial fix) that any regression
/// to per-candidate allocation produces.
#[test]
fn wang_index_query_does_not_allocate_per_candidate() {
    let query = wang_fp(7, 3);
    let mut catalog: Vec<WangFingerprint> = (0..100u32).map(|i| wang_fp(10 + i, 3)).collect();
    // Plant the true match in the middle so the candidate set is non-trivial.
    catalog[50] = query.clone();
    let index = WangIndex::build(&catalog, 100);
    let cfg = WangMatchConfig::default();

    // Warm up: the first queries grow the reused scratch to its high-water
    // mark, which is amortised (not per-query) cost.
    for _ in 0..4 {
        std::hint::black_box(index.query(&query, &cfg));
    }

    const REPS: usize = 50;
    let before = allocs();
    for _ in 0..REPS {
        std::hint::black_box(index.query(&query, &cfg));
    }
    let per_query = (allocs() - before) / REPS;

    // The flat vote list is pre-sized and the scoring scratch is reused, so a
    // warm query should allocate only a small constant (the flat list's sort
    // buffer and any growth).
    assert!(
        per_query <= 50,
        "WangIndex::query allocated {per_query} times per query (warm, 100 refs); \
         the per-candidate scratch reuse has regressed (was 369 before the fix, 16 after)"
    );
}

/// `estimated_bytes` must track the heap the index actually holds.
///
/// §4.2 #10: the map-key term was counted twice, so the report overshot. This
/// measures real live bytes across the build and compares.
#[test]
fn haitsma_index_estimated_bytes_tracks_real_heap() {
    let refs: Vec<HaitsmaFingerprint> = (0..32u32)
        .map(|i| {
            let frames: Vec<u32> = (0..600u32).map(|j| (i * 7 + j) ^ (j << 3)).collect();
            HaitsmaFingerprint {
                frames,
                frames_per_sec: 78.125,
            }
        })
        .collect();

    let base = live_bytes();
    let index = HaitsmaIndex::build(&refs, 100_000);
    let real = live_bytes() - base;
    let reported = index.estimated_bytes() as i64;

    assert!(real > 0, "index should hold heap: {real}");

    // `estimated_bytes` is documented as an approximation (allocator rounding,
    // table overhead), so this is a sanity band, not an equality: it catches a
    // gross over/under-report such as the #10 double count.
    let ratio = reported as f64 / real as f64;
    assert!(
        (0.5..=2.0).contains(&ratio),
        "estimated_bytes {reported} vs real {real} (ratio {ratio:.2}) — accounting drift"
    );
}

/// `WangIndex::estimated_bytes` must track the heap the index actually holds.
///
/// `estimated_bytes` is documented as an approximation (allocator rounding,
/// table overhead), so the bound is a sanity band rather than an equality — it
/// catches a gross over/under-report such as the §4.2 #10 double count.
#[test]
fn wang_index_estimated_bytes_tracks_real_heap() {
    let refs: Vec<WangFingerprint> = (0..64u32).map(|i| wang_fp(200 + i, 3)).collect();

    let base = live_bytes();
    let index = WangIndex::build(&refs, 100);
    let real = live_bytes() - base;
    let reported = index.estimated_bytes() as i64;

    assert!(real > 0, "index should hold heap: {real}");
    let ratio = reported as f64 / real as f64;
    assert!(
        (0.5..=2.0).contains(&ratio),
        "WangIndex::estimated_bytes {reported} vs real {real} (ratio {ratio:.2})"
    );
}

/// §4.2 #9 decision input: how much does `Vec<Vec<u32>>` cost over a flat arena?
///
/// `HaitsmaIndex` stores each reference's frames as its own `Vec<u32>`. The
/// audit proposed replacing that with a view/arena. This measures the actual
/// overhead of the nested representation against a single flat `Vec<u32>` of
/// identical contents, so the refactor can be justified by a number instead
/// of an assumption.
///
/// Overhead per reference is the `Vec` header (24 bytes: ptr/len/cap) plus
/// allocator rounding minus the arena's zero per-reference cost.
#[test]
fn haitsma_frames_arena_overhead_is_measured() {
    const REFS: usize = 100;
    const FRAMES_PER_REF: usize = 234; // ≈ 3 s at 78.125 frames/s

    // Nested: one `Vec<u32>` allocation per reference.
    let base_nested = live_bytes();
    let nested: Vec<Vec<u32>> = (0..REFS)
        .map(|r| (0..FRAMES_PER_REF).map(|i| (r * 31 + i) as u32).collect())
        .collect();
    let nested_bytes = live_bytes() - base_nested;

    // Flat: one allocation for all frame data.
    let base_flat = live_bytes();
    let flat: Vec<u32> = (0..REFS * FRAMES_PER_REF)
        .map(|i| (i * 31) as u32)
        .collect();
    let flat_bytes = live_bytes() - base_flat;

    // Keep both alive across the measurement.
    std::hint::black_box(&nested);
    std::hint::black_box(&flat);

    let data = (REFS * FRAMES_PER_REF * 4) as i64;
    let overhead = nested_bytes - data;

    // Report the composition; `--nocapture` shows it.
    println!(
        "haitsma frames: data={data} B, nested={nested_bytes} B (+{overhead} B), \
         flat={flat_bytes} B, nested/flat={:.3}",
        nested_bytes as f64 / flat_bytes as f64
    );

    // The nested form must not be wildly worse than payload + one Vec header
    // per reference: 24 B/ref is the irreducible per-reference cost.
    let per_ref_overhead = overhead as f64 / REFS as f64;
    assert!(
        per_ref_overhead < 64.0,
        "per-reference overhead {per_ref_overhead:.1} B exceeds a Vec header + rounding \
         (nested={nested_bytes}, data={data})"
    );
}

/// Pin the Haitsma frame count formula used by the #9 sizing arithmetic.
///
/// `frames_per_sec` is `sr / hop` = 5000/64 = 78.125, but the frame *count*
/// is not `secs * frames_per_sec`: the extractor drops the final partial
/// analysis window, so it is `floor((n_samples - n_fft) / hop)` with
/// `n_fft = 2048` and `hop = 64`. Both data points below are checked so the
/// formula (not just one coincidental value) is pinned.
#[test]
fn haitsma_frame_count_follows_the_floor_formula() {
    let (n_fft, hop) = (2048usize, 64usize);
    for secs in [3usize, 5] {
        let n = 5_000 * secs;
        let fp = Haitsma::default()
            .extract(&synth(3, 5_000, secs), SampleRate::HZ_5000)
            .expect("haitsma extract");
        assert!(
            (fp.frames_per_sec - 78.125).abs() < 1e-3,
            "frames_per_sec should be sr/hop = 78.125, got {}",
            fp.frames_per_sec
        );
        let expect = (n - n_fft) / hop;
        assert_eq!(
            fp.frames.len(),
            expect,
            "{secs}s ({n} samples) should yield floor(({n} - {n_fft})/{hop}) = {expect} frames"
        );
    }
}
