//! Allocation-freedom regression tests for [`ZeroAllocStreaming`].
//!
//! Each test warms up a streamer with representative pushes (allowing
//! amortised `Vec` growth), snapshots the global allocation count, then
//! runs steady-state `push_with` / `flush_with` and asserts ZERO further
//! allocations. Any future edit that introduces an allocation in the hot
//! path (a stray `Vec::new`, a growing buffer, a collecting iterator)
//! fails these tests.
//!
//! Requires `std` (custom global allocator + threads are out of scope for
//! the `no_std` build). Incompatible with the `mimalloc` feature (both
//! install a `#[global_allocator]`), so the whole file is compiled out
//! when `mimalloc` is enabled — CI's `--all-features` runs skip it.

#![cfg(all(feature = "std", not(feature = "mimalloc")))]

use std::alloc::{GlobalAlloc, Layout, System};

use audiofp::classical::{StreamingHaitsma, StreamingPanako, StreamingWang};
use audiofp::{TimestampMs, ZeroAllocStreaming};

/// Counting wrapper around [`System`]: every successful `alloc` bumps the
/// *calling thread's* counter. Thread-local (not global) so parallel tests
/// in this binary never contaminate each other's deltas. `try_with` skips
/// the count during TLS initialization itself (where counting would
/// recurse).
struct CountingAlloc;

std::thread_local! {
    static LOCAL_ALLOCS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: delegates to `System`.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            // `try_with` (not `with`): skips the count during TLS
            // initialization/teardown itself, where counting would recurse
            // or panic. Skipped counts only affect interpreter startup, far
            // outside any measured phase.
            let _ = LOCAL_ALLOCS.try_with(|c| c.set(c.get() + 1));
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: paired with `alloc` above.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

/// Snapshot the current thread's allocation count.
fn allocs() -> usize {
    LOCAL_ALLOCS.with(|c| c.get())
}

/// Deterministic two-tone + noise signal, rich enough to emit real frames.
fn synth(seed: u32, sr: u32, len: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(len);
    let mut x = seed.max(1);
    for n in 0..len {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        let noise = (x as i32 as f32) / (i32::MAX as f32) * 0.05;
        let t = n as f32 / sr as f32;
        out.push(
            0.5 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
                + 0.3 * (2.0 * std::f32::consts::PI * 1320.0 * t).sin()
                + noise,
        );
    }
    out
}

/// Assert `S: ZeroAllocStreaming` at the type level (the bound generic
/// helpers rely on), then drive warmup + measured phases on ONE stream:
/// `warmup_chunks` pushes stabilize every amortised buffer, then the next
/// `measured_chunks` pushes plus `flush` must allocate zero times.
///
/// Warmup has three stages: exotic calls (`push`/`flush`, which reset
/// capacity), steady pushes, a FULL-BACKLOG `flush_with` (which pays the
/// end-of-stream high-water growths), and re-stabilization pushes (which
/// recover the `take()` resets). See inline comments.
fn assert_zero_alloc<S>(
    make: impl Fn() -> S,
    sr: u32,
    chunk: usize,
    warmup_chunks: usize,
    measured_chunks: usize,
) where
    S: ZeroAllocStreaming,
{
    fn assert_bound<S: ZeroAllocStreaming>() {}
    assert_bound::<S>();

    /// Re-stabilization pushes after the full-backlog warmup flush.
    const RESTABILIZE_CHUNKS: usize = 40;

    let mut s = make();
    let audio = synth(
        0xC0FFEE,
        sr,
        chunk * (warmup_chunks + RESTABILIZE_CHUNKS + measured_chunks + 4),
    );
    let mut chunks = audio.chunks(chunk);

    // Warmup order matters (see below): the allocating `push` / `flush`
    // (which reset buffer capacity via `mem::take`) run FIRST, then steady
    // `push_with` re-stabilizes every buffer to its steady-state capacity.
    // (Putting a `flush()` right before measurement would falsely fail —
    // the take() leaves capacity zero and the next push must regrow.
    // `flush()` may allocate by contract; only `flush_with` is measured.)
    let _ = s.push(&audio[..chunk.min(audio.len())]);
    let _ = s.push_with(&audio[..(2 * chunk).min(audio.len())], |_, _| {});
    let _ = s.flush();
    for c in chunks.by_ref().take(warmup_chunks) {
        let _ = s.push_with(c, |_, _| {});
    }
    // Full-backlog flush: the mid-warmup flush above sees a FILLING pipeline
    // (small backlog), but the measured flush sees the FULL steady backlog
    // (zone-depth worth of pending anchors). Without a full-backlog flush in
    // warmup, the measured flush would establish new high-waters (emitted
    // buffer, anchor deque) and falsely fail. This flush pays those one-time
    // growths here; the re-stabilization pushes below recover the
    // `mem::take` capacity resets it causes.
    let _ = s.flush_with(|_, _| {});
    // Re-stabilize after the take() resets above: refill every buffer to its
    // steady-state capacity BEFORE the measured phase.
    for c in chunks.by_ref().take(RESTABILIZE_CHUNKS) {
        let _ = s.push_with(c, |_, _| {});
    }

    // Measured phase on the SAME stream: steady-state pushes + flush must
    // not allocate. (A fresh stream would restart amortised growth and
    // falsely fail — warmup capacity belongs to this instance.)
    // Push and flush budgets are tracked separately so failures pinpoint
    // which half regressed.
    let mut emitted = 0usize;
    let before_push = allocs();
    for c in chunks.take(measured_chunks) {
        emitted += s.push_with(c, |_: TimestampMs, _| {}).unwrap();
    }
    let push_grown = allocs() - before_push;
    let before_flush = allocs();
    emitted += s.flush_with(|_: TimestampMs, _| {}).unwrap();
    let flush_grown = allocs() - before_flush;
    assert_eq!(
        push_grown, 0,
        "push_with allocated {push_grown} times after warmup (emitted {emitted})"
    );
    assert_eq!(
        flush_grown, 0,
        "flush_with allocated {flush_grown} times after warmup (emitted {emitted})"
    );
    assert!(
        emitted > 0,
        "fixture must emit frames or the test is vacuous"
    );
}

#[test]
fn streaming_wang_is_zero_alloc_after_warmup() {
    assert_zero_alloc(StreamingWang::default, 8_000, 1_024, 120, 40);
}

#[test]
fn streaming_panako_is_zero_alloc_after_warmup() {
    assert_zero_alloc(StreamingPanako::default, 8_000, 1_024, 120, 40);
}

#[test]
fn streaming_haitsma_is_zero_alloc_after_warmup() {
    assert_zero_alloc(StreamingHaitsma::default, 5_000, 1_024, 120, 40);
}
