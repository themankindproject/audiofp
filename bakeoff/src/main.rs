//! Chromaprint bakeoff harness (issue #88).
//!
//! Measures audiofp (Wang / Panako / Haitsma) vs chromaprint on the shared
//! robustness corpus (#87): overlap, 1:N identification, and latency.
//!
//! Usage:
//!   cargo run --release -- --report   # print the markdown report to stdout
//!   cargo run --release -- --check    # run the invariant suite, exit 1 on violation
//!   cargo run --release               # both
//!
//! No changes to audiofp sources — this consumes the public API only.

mod cp;

use std::collections::HashMap;
use std::time::Instant;

use audiofp::dsp::resample::SincResampler;
use audiofp::io::decode_to_mono;
use audiofp::matching::{
    HaitsmaMatchConfig, HaitsmaMatcher, Matcher, PanakoMatchConfig, PanakoMatcher, WangIndex,
    WangMatchConfig,
};
use audiofp::{
    Fingerprinter, Haitsma, HaitsmaFingerprint, Panako, PanakoFingerprint, SampleRate, Wang,
    WangFingerprint,
};

const ASSETS: &str = "../tests/assets";
const REPETITIONS: usize = 3;
const POSTINGS: u32 = 1000;

/// One corpus track: a display name, its lossless reference file, and all of
/// its codec/rate variants (paths relative to `ASSETS`).
struct Track {
    name: &'static str,
    reference: &'static str,
    variants: &'static [&'static str],
}

/// The #87 corpus, verbatim. References are the lossless sources.
fn corpus() -> Vec<Track> {
    vec![
        Track {
            name: "galway",
            reference: "galway.flac",
            variants: &[
                "galway.flac",
                "galway.wav",
                "galway.mp3",
                "galway.ogg",
                "galway.m4a",
                "galway.aac",
                "galway.aiff",
                "galway_stereo.mp3",
            ],
        },
        Track {
            name: "freak",
            reference: "freak.flac",
            variants: &[
                "freak.flac",
                "freak.wav",
                "freak.mp3",
                "freak.ogg",
                "freak.m4a",
                "freak_8000hz.mp3",
                "freak_11025hz.mp3",
                "freak_16000hz.mp3",
                "freak_22050hz.mp3",
                "freak_32000hz.mp3",
                "freak_44100hz.mp3",
            ],
        },
        Track {
            name: "bach_goldberg_aria",
            reference: "catalog/bach_goldberg_aria.ogg",
            variants: &["catalog/bach_goldberg_aria.ogg"],
        },
        Track {
            name: "bach_goldberg_var4",
            reference: "catalog/bach_goldberg_var4.ogg",
            variants: &["catalog/bach_goldberg_var4.ogg"],
        },
        Track {
            name: "beethoven_coriolan",
            reference: "catalog/beethoven_coriolan.ogg",
            variants: &["catalog/beethoven_coriolan.ogg"],
        },
        Track {
            name: "beethoven_egmont",
            reference: "catalog/beethoven_egmont.ogg",
            variants: &["catalog/beethoven_egmont.ogg"],
        },
        Track {
            name: "beethoven_eroica_mvt1",
            reference: "catalog/beethoven_eroica_mvt1.ogg",
            variants: &["catalog/beethoven_eroica_mvt1.ogg"],
        },
        Track {
            name: "dvorak_american_mvt1",
            reference: "catalog/dvorak_american_mvt1.ogg",
            variants: &["catalog/dvorak_american_mvt1.ogg"],
        },
        Track {
            name: "grieg_morning",
            reference: "catalog/grieg_morning.ogg",
            variants: &["catalog/grieg_morning.ogg"],
        },
    ]
}

/// Decoded PCM for one file (mono, native rate) + decode timing.
struct Decoded {
    file: String,
    samples: Vec<f32>,
    sr: u32,
    duration: f32,
    decode_ms: f64,
}

/// Per-stage kernel timing (median of `REPETITIONS` after one warmup).
#[derive(Debug, Clone, Copy)]
struct KernelTiming {
    /// Resample to the target rate, or copy PCM when native == target.
    resample_ms: f64,
    /// Core fingerprint extraction on target-rate PCM.
    extract_ms: f64,
    /// resample + extract — what a caller pays after decode.
    kernel_ms: f64,
}

/// Fingerprint data + staged kernel timing for one file.
struct Fps {
    wang: WangFingerprint,
    panako: PanakoFingerprint,
    haitsma: HaitsmaFingerprint,
    cp: Vec<u32>,
    wang_timing: KernelTiming,
    panako_timing: KernelTiming,
    haitsma_timing: KernelTiming,
    cp_ms: f64,
}

type Fmap = HashMap<String, Fps>;
type Dmap = HashMap<String, Decoded>;

/// Median of a non-empty slice.
fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    s[s.len() / 2]
}

/// Resample (if needed) then extract, `REPETITIONS` times after one warmup.
///
/// Setup (resampler construction, one untimed resample+extract) stays outside
/// the timed loop. Each timed repetition measures resample and extract
/// separately; `kernel_ms` is their sum so comparisons include the resample
/// work the methodology claims.
fn extract_timed<F: Fingerprinter + Default>(
    samples: &[f32],
    native: u32,
    target: u32,
) -> (F::Output, KernelTiming) {
    let mut f = F::default();
    let rate = SampleRate::new(target).expect("non-zero target rate");
    let resampler = (native != target).then(|| SincResampler::new(native, target));

    let resample_input = |raw: &[f32]| -> Vec<f32> {
        match &resampler {
            Some(r) => r.process(raw),
            None => raw.to_vec(),
        }
    };

    // Warmup: one full resample + extract (not timed).
    let warmup = resample_input(samples);
    let mut fp = f.extract(&warmup, rate).expect("extract warmup");

    let mut resample_times = Vec::with_capacity(REPETITIONS);
    let mut extract_times = Vec::with_capacity(REPETITIONS);
    let mut kernel_times = Vec::with_capacity(REPETITIONS);
    for _ in 0..REPETITIONS {
        let t_kernel = Instant::now();
        let t_resample = Instant::now();
        let input = resample_input(samples);
        let resample_ms = t_resample.elapsed().as_secs_f64() * 1e3;

        let t_extract = Instant::now();
        fp = f.extract(&input, rate).expect("extract");
        let extract_ms = t_extract.elapsed().as_secs_f64() * 1e3;
        let kernel_ms = t_kernel.elapsed().as_secs_f64() * 1e3;

        resample_times.push(resample_ms);
        extract_times.push(extract_ms);
        kernel_times.push(kernel_ms);
    }

    let timing = KernelTiming {
        resample_ms: median(&resample_times),
        extract_ms: median(&extract_times),
        kernel_ms: median(&kernel_times),
    };
    (fp, timing)
}

/// Extract every fingerprint + staged kernel timing for one decoded file.
fn extract_all(d: &Decoded) -> Fps {
    let (wang, wang_timing) = extract_timed::<Wang>(&d.samples, d.sr, 8_000);
    let (panako, panako_timing) = extract_timed::<Panako>(&d.samples, d.sr, 8_000);
    let (haitsma, haitsma_timing) = extract_timed::<Haitsma>(&d.samples, d.sr, 5_000);
    let mut cp_times = Vec::with_capacity(REPETITIONS);
    // Warmup once, then time (chromaprint resamples internally).
    let mut cp_raw = cp::extract(&d.samples, d.sr);
    for _ in 0..REPETITIONS {
        let t = Instant::now();
        cp_raw = cp::extract(&d.samples, d.sr);
        cp_times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    Fps {
        wang,
        panako,
        haitsma,
        cp: cp_raw,
        wang_timing,
        panako_timing,
        haitsma_timing,
        cp_ms: median(&cp_times),
    }
}

/// Jaccard over a slice of hashable ids (ROBUSTNESS.md definition).
fn jaccard<T: std::hash::Hash + Eq>(a: &[T], b: &[T]) -> f32 {
    let sa: std::collections::HashSet<&T> = a.iter().collect();
    let sb: std::collections::HashSet<&T> = b.iter().collect();
    let inter = sa.intersection(&sb).count();
    let union = sa.union(&sb).count().max(1);
    inter as f32 / union as f32
}

/// Aligned 32-bit frame similarity (Haitsma; ROBUSTNESS.md definition).
fn bit_sim(a: &[u32], b: &[u32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    (0..n)
        .map(|i| 32.0 - (a[i] ^ b[i]).count_ones() as f32)
        .sum::<f32>()
        / (n as f32 * 32.0)
}

/// Chromaprint offset-tolerant 32-bit hamming, best over shift k in [-2, 2].
/// Returns (best_normalized_similarity, best_shift). Ties resolve to the
/// smallest |shift|, then the lower shift.
fn cp_shift_sim(a: &[u32], b: &[u32]) -> (f32, i32) {
    let (la, lb) = (a.len(), b.len());
    if la == 0 || lb == 0 {
        return (0.0, 0);
    }
    let mut best = (0.0f32, 0i32);
    for k in -2i32..=2 {
        let (start_a, start_b) = if k >= 0 {
            (0usize, k as usize)
        } else {
            ((-k) as usize, 0usize)
        };
        let n = la.saturating_sub(start_a).min(lb.saturating_sub(start_b));
        if n == 0 {
            continue;
        }
        let sim = (0..n)
            .map(|i| 32.0 - (a[start_a + i] ^ b[start_b + i]).count_ones() as f32)
            .sum::<f32>()
            / (n as f32 * 32.0);
        let better = if sim > best.0 + 1e-9 {
            true
        } else if (sim - best.0).abs() <= 1e-9 {
            k.abs() < best.1.abs() || (k.abs() == best.1.abs() && k < best.1)
        } else {
            false
        };
        if better {
            best = (sim, k);
        }
    }
    best
}

fn fmt3(x: f32) -> String {
    format!("{x:.3}")
}

/// The 1:N identification catalog: one reference fingerprint per track.
struct Catalog {
    /// catalog slot -> track index
    ids: Vec<usize>,
    wang: Vec<WangFingerprint>,
    panako: Vec<PanakoFingerprint>,
    haitsma: Vec<HaitsmaFingerprint>,
    cp: Vec<Vec<u32>>,
}

fn build_catalog(tracks: &[Track], fmap: &Fmap) -> Catalog {
    let ids: Vec<usize> = (0..tracks.len()).collect();
    let mut wang = Vec::new();
    let mut panako = Vec::new();
    let mut haitsma = Vec::new();
    let mut cp = Vec::new();
    for &tid in &ids {
        let f = &fmap[tracks[tid].reference];
        wang.push(f.wang.clone());
        panako.push(f.panako.clone());
        haitsma.push(f.haitsma.clone());
        cp.push(f.cp.clone());
    }
    Catalog {
        ids,
        wang,
        panako,
        haitsma,
        cp,
    }
}

/// Top-1 (slot, score, margin) by a 1:1 `score` closure over catalog slots.
/// Ties resolve to the lowest slot.
fn best_of<Q, R>(query: &Q, refs: &[R], mut score: impl FnMut(&Q, &R) -> f32) -> (usize, f32, f32) {
    let mut scored: Vec<(usize, f32)> = refs
        .iter()
        .enumerate()
        .map(|(i, r)| (i, score(query, r)))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let (best_id, best) = scored[0];
    let margin = if scored.len() > 1 {
        best - scored[1].1
    } else {
        best
    };
    (best_id, best, margin)
}

/// Top-1 (slot, sim, margin) by cp_shift_sim over catalog slots.
fn cp_top1(query: &[u32], refs: &[Vec<u32>]) -> (usize, f32, f32) {
    let mut scored: Vec<(usize, f32)> = refs
        .iter()
        .enumerate()
        .map(|(i, r)| (i, cp_shift_sim(query, r).0))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let (best_id, best) = scored[0];
    let margin = if scored.len() > 1 {
        best - scored[1].1
    } else {
        best
    };
    (best_id, best, margin)
}

/// Top-1 identification slot with score and margin.
///
/// `slot == None` means the matcher/index returned no candidate (e.g.
/// WangIndex below threshold). Never use a sentinel index.
#[derive(Clone, Copy, Debug, PartialEq)]
struct MatchSlot {
    slot: Option<usize>,
    score: f32,
    margin: f32,
}

/// Resolve a catalog slot to a track display name for reporting.
fn slot_track_name(tracks: &[Track], cat: &Catalog, slot: Option<usize>) -> &'static str {
    match slot {
        Some(i) if i < cat.ids.len() => tracks[cat.ids[i]].name,
        Some(_) => "(out of range)",
        None => "— (no match)",
    }
}

/// Whether a top-1 slot identifies the query's source track.
fn id_correct(cat: &Catalog, slot: Option<usize>, source_track: usize) -> bool {
    matches!(slot, Some(i) if i < cat.ids.len() && cat.ids[i] == source_track)
}

/// One identified query row.
struct IdRow {
    file: String,
    source_track: usize,
    wang: MatchSlot,
    panako: MatchSlot,
    haitsma: MatchSlot,
    cp: MatchSlot,
    wang_query_ms: f64,
    cp_query_ms: f64,
}

/// Run 1:N identification for all query files (every non-reference variant).
fn run_identification(tracks: &[Track], fmap: &Fmap, cat: &Catalog) -> Vec<IdRow> {
    let wang_index = WangIndex::build(&cat.wang, POSTINGS);
    let panako_m = PanakoMatcher::new(PanakoMatchConfig::default());
    let haitsma_m = HaitsmaMatcher::new(HaitsmaMatchConfig::default());
    let mut rows = Vec::new();
    for (tid, track) in tracks.iter().enumerate() {
        for &file in track.variants {
            if file == track.reference {
                continue; // the reference itself is not a query
            }
            let f = &fmap[file];
            // Wang: index query (primary audiofp path).
            let t = Instant::now();
            let wang = match wang_index.query(&f.wang, &WangMatchConfig::default()) {
                Some((slot, r)) => MatchSlot {
                    slot: Some(slot),
                    score: r.score,
                    margin: 0.0,
                },
                None => MatchSlot {
                    slot: None,
                    score: 0.0,
                    margin: 0.0,
                },
            };
            let wang_query_ms = t.elapsed().as_secs_f64() * 1e3;
            // Panako / Haitsma: 1:1-vote max over the 9 refs (secondary).
            let (panako_id, panako_score, panako_margin) =
                best_of(&f.panako, &cat.panako, |q, r| {
                    panako_m.match_one(q, r).score
                });
            let panako = MatchSlot {
                slot: Some(panako_id),
                score: panako_score,
                margin: panako_margin,
            };
            let (haitsma_id, haitsma_score, haitsma_margin) =
                best_of(&f.haitsma, &cat.haitsma, |q, r| {
                    haitsma_m.match_one(q, r).score
                });
            let haitsma = MatchSlot {
                slot: Some(haitsma_id),
                score: haitsma_score,
                margin: haitsma_margin,
            };
            // Chromaprint: best-shift-hamming argmin.
            let t = Instant::now();
            let (cp_id, cp_score, cp_margin) = cp_top1(&f.cp, &cat.cp);
            let cp = MatchSlot {
                slot: Some(cp_id),
                score: cp_score,
                margin: cp_margin,
            };
            let cp_query_ms = t.elapsed().as_secs_f64() * 1e3;
            rows.push(IdRow {
                file: file.to_string(),
                source_track: tid,
                wang,
                panako,
                haitsma,
                cp,
                wang_query_ms,
                cp_query_ms,
            });
        }
    }
    rows
}

fn git_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

fn audiofp_version() -> String {
    // The harness package is 0.0.0 (unpublished); the audiofp under test is
    // the path dependency at the repo root. Read its version straight from
    // the root Cargo.toml's [package] section (deterministic, no JSON dep).
    std::fs::read_to_string("../Cargo.toml")
        .ok()
        .and_then(|t| {
            let pkg = t.find("[package]")?;
            let rest = &t[pkg..];
            let ver = rest.find("\nversion")?;
            let line = &rest[ver..];
            let eq = line.find('=')?;
            let quote = line[eq..].find('"')?;
            let rest = &line[eq + quote + 1..];
            let end = rest.find('"')?;
            Some(rest[..end].to_string())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn today() -> String {
    std::process::Command::new("date")
        .arg("+%Y-%m-%d")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

fn environment() -> String {
    let cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|c| {
            c.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split_once(':'))
                .map(|(_, v)| v.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into());
    let nproc = std::fs::read_to_string("/proc/cpuinfo")
        .map(|c| c.lines().filter(|l| l.starts_with("processor")).count())
        .unwrap_or(0);
    let os = format!("{} {}", std::env::consts::OS, std::env::consts::ARCH);
    let rustc = std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "rustc (unknown)".into());
    format!(
        "- CPU: {cpu} ({nproc} logical)\n\
         - OS: {os}\n\
         - rustc: {rustc}\n\
         - audiofp: v{ver} (commit {commit})\n\
         - chromaprint: {cp_ver} (libchromaprint-dev, system package)\n\
         - build: release, lto = \"fat\", codegen-units = 1 (harness + audiofp)\n\
         - chromaprint flags: algorithm = DEFAULT (TEST2), mono, native rate (internal resample)",
        ver = audiofp_version(),
        commit = git_commit(),
        cp_ver = cp::version(),
    )
}

const CAVEATS: &str = "\
1. **Different algorithms.** Chromaprint is a single-band chroma fingerprint
   (one u32 item per hop); audiofp's three are mel-peak based. The overlap
   columns are *not* cross-comparable — each system's overlap is against its
   own reference.
2. **Rate/resample.** audiofp is fed 8 kHz (Wang/Panako) and 5 kHz (Haitsma)
   from the shared symphonia f32 mono buffer (resampled by SincResampler);
   chromaprint is fed at the file's native rate and resamples internally
   (its own resampler). Decode is shared, resample is not.
3. **1:N scale.** The 9-track catalog is a smoke test of identification
   *logic*, not scale (scale is #103/#109's territory).
4. **chromaprint 1.6.0** (Debian/Ubuntu `libchromaprint-dev` package) — exact
   version + package source pinned above; upstream HEAD may differ.
5. **Single-machine latency.** Both systems measured single-threaded
   (rayon off) for fairness; parallel-extraction speedups are a separate
   axis (#119) and are not conflated into these numbers.
6. **Corpus bias.** 2 × 16 s contemporary-music tracks + 7 × 30 s classical.
   Classical's low crest factor exercises peak-picking differently; the
   catalog identification mixes genres on purpose (real catalogs are mixed).";

const METHODOLOGY: &str = "\
- **Overlap** (M1): Jaccard of u32 hash sets (Wang/Panako), aligned 32-bit
  frame similarity (Haitsma), offset-tolerant ±2-frame 32-bit hamming
  (chromaprint). Definitions match `ROBUSTNESS.md`; chromaprint's is the
  AcoustID-style aligned lookup. Each column is against the track's lossless
  reference.
- **Identification** (M2): 9-track catalog (2 music + 7 classical), 17 queries
  (all non-reference music variants, incl. lossless WAV/AIFF). audiofp
  primary = `WangIndex` offset-histogram voter (default config); secondary =
  Panako/Haitsma 1:1-vote max. chromaprint = best-shift-hamming argmin.
  Margin = best − runner-up (audiofp secondary + chromaprint only; the index
  path returns a single winner).
- **Latency** (M3): median of 3 timed repetitions after one untimed warmup,
  release (lto=fat), single-threaded both systems. Decode is timed once at
  ingest. audiofp kernel stages are reported separately: **resample** (when
  native rate ≠ target), **extract** (fingerprint core on target-rate PCM),
  and **kernel** (= resample + extract). chromaprint kernel is one timed step
  (includes its internal resample). e2e = decode + kernel (audiofp e2e averages
  the three kernels).
";

fn hash_ids(w: &WangFingerprint) -> Vec<u32> {
    w.hashes.iter().map(|h| h.hash).collect()
}

fn panako_ids(p: &PanakoFingerprint) -> Vec<u32> {
    p.hashes.iter().map(|h| h.hash).collect()
}

fn corpus_lines(tracks: &[Track]) -> String {
    let mut s = String::new();
    for (i, t) in tracks.iter().enumerate() {
        s.push_str(&format!(
            "- track {i}: `{}` ({} variants)\n",
            t.name,
            t.variants.len()
        ));
    }
    s.push_str(
        "  Licenses: MacLeod tracks CC-BY 3.0; catalog CC0 1.0 (tests/assets/CREDITS.md).\n",
    );
    s
}

/// M1 overlap table: every variant vs its track's lossless reference.
fn m1_table(tracks: &[Track], fmap: &Fmap, dmap: &Dmap) -> String {
    let mut s = String::from("### Overlap (vs lossless reference)\n\n");
    s.push_str("Format-specific. Columns are each system's own metric against the reference.\n\n");
    s.push_str(
        "| track | variant | dur(s) | Wang Jaccard | Panako Jaccard | Haitsma bit-sim | cp shift-sim | cp shift |\n",
    );
    s.push_str("|---|---|---|---|---|---|---|---|\n");
    for track in tracks {
        let ref_fps = &fmap[track.reference];
        for &file in track.variants {
            let f = &fmap[file];
            let d = &dmap[file];
            let (cp_sim, cp_shift) = cp_shift_sim(&ref_fps.cp, &f.cp);
            s.push_str(&format!(
                "| {} | {} | {:.1} | {} | {} | {} | {} | {:+} |\n",
                track.name,
                file,
                d.duration,
                fmt3(jaccard(&hash_ids(&ref_fps.wang), &hash_ids(&f.wang))),
                fmt3(jaccard(
                    &panako_ids(&ref_fps.panako),
                    &panako_ids(&f.panako)
                )),
                fmt3(bit_sim(&ref_fps.haitsma.frames, &f.haitsma.frames)),
                fmt3(cp_sim),
                cp_shift,
            ));
        }
    }
    // Cross-track discrimination (do different songs stay far apart?).
    let g = &fmap["galway.flac"];
    let f = &fmap["freak.flac"];
    let (cp_cross, _) = cp_shift_sim(&g.cp, &f.cp);
    s.push_str(&format!(
        "\nCross-track (galway.flac vs freak.flac): Wang Jaccard {}, Haitsma bit-sim {}, \
         cp shift-sim {}. Different songs must stay far apart: Wang Jaccard < 0.05 \
         (hash-set collision floor); Haitsma bit-sim and cp shift-sim floor at \
         ~0.5 for unrelated audio (random-bit agreement — chromaprint's dense \
         11.025 kHz fingerprints agree on more bits by chance) and rise toward \
         1.0 only on a real match (≥ 0.95).\n",
        fmt3(jaccard(&hash_ids(&g.wang), &hash_ids(&f.wang))),
        fmt3(bit_sim(&g.haitsma.frames, &f.haitsma.frames)),
        fmt3(cp_cross),
    ));
    s
}

fn tick(b: bool) -> &'static str {
    if b { "✓" } else { "✗" }
}

/// M2 identification table.
fn m2_table(tracks: &[Track], cat: &Catalog, rows: &[IdRow]) -> String {
    let mut s = String::new();
    s.push_str(
        "| query | source | Wang (ok) | Panako (ok) | Haitsma (ok) | cp id (ok) | cp margin |\n",
    );
    s.push_str("|---|---|---|---|---|---|---|\n");
    for r in rows {
        let src = tracks[r.source_track].name;
        let w_ok = id_correct(cat, r.wang.slot, r.source_track);
        let p_ok = id_correct(cat, r.panako.slot, r.source_track);
        let h_ok = id_correct(cat, r.haitsma.slot, r.source_track);
        let c_ok = id_correct(cat, r.cp.slot, r.source_track);
        s.push_str(&format!(
            "| {} | {} | {} {} | {} {} | {} {} | {} {} | {} |\n",
            r.file,
            src,
            slot_track_name(tracks, cat, r.wang.slot),
            tick(w_ok),
            slot_track_name(tracks, cat, r.panako.slot),
            tick(p_ok),
            slot_track_name(tracks, cat, r.haitsma.slot),
            tick(h_ok),
            slot_track_name(tracks, cat, r.cp.slot),
            tick(c_ok),
            fmt3(r.cp.margin),
        ));
    }
    let wang_correct = rows
        .iter()
        .filter(|r| id_correct(cat, r.wang.slot, r.source_track))
        .count();
    let panako_correct = rows
        .iter()
        .filter(|r| id_correct(cat, r.panako.slot, r.source_track))
        .count();
    let haitsma_correct = rows
        .iter()
        .filter(|r| id_correct(cat, r.haitsma.slot, r.source_track))
        .count();
    let cp_correct = rows
        .iter()
        .filter(|r| id_correct(cat, r.cp.slot, r.source_track))
        .count();
    let wang_no_match = rows.iter().filter(|r| r.wang.slot.is_none()).count();
    s.push_str(&format!(
        "\nTop-1 / {}: audiofp Wang {} (primary, {} no-match) · Panako {} · Haitsma {} · chromaprint {}\n",
        rows.len(),
        wang_correct,
        wang_no_match,
        panako_correct,
        haitsma_correct,
        cp_correct,
    ));
    s
}

/// M3 latency tables.
fn m3_table(tracks: &[Track], dmap: &Dmap, fmap: &Fmap, rows: &[IdRow]) -> String {
    let mut s = String::new();
    s.push_str(
        "Per file (median of 3 after warmup; decode measured once). e2e = decode + kernel. \
         audiofp stage medians are averaged across its three algorithms and may not add \
         exactly to the separately measured kernel median. Resample includes a PCM copy \
         when rates already agree.\n\n",
    );
    s.push_str(
        "| file | dur(s) | decode(ms) | e2e-audiofp(ms) | e2e-cp(ms) | \
         resample(ms) | extract(ms) | kernel(ms) | cp(ms) |\n",
    );
    s.push_str("|---|---|---|---|---|---|---|---|---|\n");
    for track in tracks {
        for &file in track.variants {
            let d = &dmap[file];
            let f = &fmap[file];
            let avg_kernel =
                (f.wang_timing.kernel_ms + f.panako_timing.kernel_ms + f.haitsma_timing.kernel_ms)
                    / 3.0;
            let avg_resample = (f.wang_timing.resample_ms
                + f.panako_timing.resample_ms
                + f.haitsma_timing.resample_ms)
                / 3.0;
            let avg_extract = (f.wang_timing.extract_ms
                + f.panako_timing.extract_ms
                + f.haitsma_timing.extract_ms)
                / 3.0;
            s.push_str(&format!(
                "| {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} |\n",
                file,
                d.duration,
                d.decode_ms,
                d.decode_ms + avg_kernel,
                d.decode_ms + f.cp_ms,
                avg_resample,
                avg_extract,
                avg_kernel,
                f.cp_ms,
            ));
        }
    }
    let n = rows.len().max(1) as f64;
    let wang_q = rows.iter().map(|r| r.wang_query_ms).sum::<f64>() / n;
    let cp_q = rows.iter().map(|r| r.cp_query_ms).sum::<f64>() / n;
    s.push_str(&format!(
        "\nQuery (mean over {} queries): Wang index {:.3} ms · cp best-shift {:.3} ms\n",
        rows.len(),
        wang_q,
        cp_q,
    ));
    s
}

/// Invariant suite. Returns violation messages (empty = all held); prints a
/// check count to stderr so stdout stays clean.
fn run_invariants(tracks: &[Track], fmap: &Fmap, rows: &[IdRow], cat: &Catalog) -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    let mut n = 0usize;

    // 0. FFI self-check: encode/decode roundtrip reproduces the raw
    //    fingerprint exactly (replaces the CLI cross-check — no CLI
    //    installed). If this fails, the FFI bookkeeping is broken.
    for track in tracks {
        for &file in track.variants {
            let f = &fmap[file];
            n += 1;
            if !cp::encode_decode_roundtrip(&f.cp) {
                v.push(format!("encode/decode roundtrip failed for {file}"));
            }
        }
    }

    // 1. Lossless identity: each reference vs itself is ~1.0 on every metric.
    for track in tracks {
        let f = &fmap[track.reference];
        n += 1;
        let self_j = jaccard(&hash_ids(&f.wang), &hash_ids(&f.wang));
        let self_b = bit_sim(&f.haitsma.frames, &f.haitsma.frames);
        let (self_c, _) = cp_shift_sim(&f.cp, &f.cp);
        if !(self_j >= 0.999 && self_b >= 0.999 && self_c >= 0.999) {
            v.push(format!(
                "identity {}: Wang Jaccard {self_j}/Haitsma bit-sim {self_b}/cp sim {self_c} < 0.999",
                track.reference
            ));
        }
    }

    // 2. Cross-track floor: different songs must not collide.
    //    - Wang Jaccard: ≤ 0.05 (hash-set collision floor from ROBUSTNESS.md).
    //    - Haitsma bit-sim: ≤ 0.60. Raw 32-bit frames of unrelated audio
    //      agree on ~half the bits by chance (0.50 is the random floor);
    //      0.60 allows codec-induced alignment drift while still catching
    //      a real collision (same-song overlap is ≥ 0.75).
    //    - cp shift-sim: ≤ 0.75. Chromaprint fingerprints are denser
    //      (11.025 kHz internal) than audiofp's 8 kHz hashes, so
    //      bit-chance agreement of unrelated audio floors higher —
    //      measured ~0.50 on this corpus — while true matches are
    //      ≥ 0.95. A collision pushes sim toward 1.0.
    {
        let g = &fmap["galway.flac"];
        let f = &fmap["freak.flac"];
        let j = jaccard(&hash_ids(&g.wang), &hash_ids(&f.wang));
        let b = bit_sim(&g.haitsma.frames, &f.haitsma.frames);
        let (c, _) = cp_shift_sim(&g.cp, &f.cp);
        n += 3;
        if j > 0.05 {
            v.push(format!("cross-track Wang Jaccard {j} > 0.05"));
        }
        if b > 0.60 {
            v.push(format!("cross-track Haitsma bit-sim {b} > 0.60"));
        }
        if c > 0.75 {
            v.push(format!("cross-track cp shift-sim {c} > 0.75"));
        }
    }

    // 3. Identification floors: Wang ≥ 15/17 AND chromaprint ≥ 15/17.
    {
        let w = rows
            .iter()
            .filter(|r| id_correct(cat, r.wang.slot, r.source_track))
            .count();
        let c = rows
            .iter()
            .filter(|r| id_correct(cat, r.cp.slot, r.source_track))
            .count();
        n += 2;
        if w < 15 {
            v.push(format!("Wang top-1 {w}/{} < 15/{}", rows.len(), rows.len()));
        }
        if c < 15 {
            v.push(format!("cp top-1 {c}/{} < 15/{}", rows.len(), rows.len()));
        }
    }

    // 4. Kernel-latency sanity: no audiofp kernel > 500 ms per file
    //    (16 s files → 500 ms is ~32× faster than real-time).
    for track in tracks {
        for &file in track.variants {
            let f = &fmap[file];
            n += 1;
            if f.wang_timing.kernel_ms > 500.0
                || f.panako_timing.kernel_ms > 500.0
                || f.haitsma_timing.kernel_ms > 500.0
            {
                v.push(format!(
                    "kernel latency {file}: Wang {:.0}/Panako {:.0}/Haitsma {:.0} ms ≥ 500 ms",
                    f.wang_timing.kernel_ms, f.panako_timing.kernel_ms, f.haitsma_timing.kernel_ms
                ));
            }
        }
    }

    // 5. Frame-count sanity: chromaprint u32 items in [60, 260]
    //    (16 s → ~106; 30 s → ~200; band is generous for drift).
    for track in tracks {
        for &file in track.variants {
            let f = &fmap[file];
            n += 1;
            if !(60..=260).contains(&f.cp.len()) {
                v.push(format!(
                    "cp item count {file}: {} outside [60, 260]",
                    f.cp.len()
                ));
            }
        }
    }

    eprintln!("[bakeoff] invariant suite: {n} checks");
    v
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let do_report = args.iter().any(|a| a == "--report") || args.len() == 1;
    let do_check = args.iter().any(|a| a == "--check") || args.len() == 1;

    let tracks = corpus();

    // Decode every variant once (the shared stage for both systems).
    let mut dmap: Dmap = HashMap::new();
    for track in &tracks {
        for &file in track.variants {
            let path = format!("{ASSETS}/{file}");
            let t = Instant::now();
            let (samples, sr) =
                decode_to_mono(&path).unwrap_or_else(|e| panic!("decode {path}: {e}"));
            let decode_ms = t.elapsed().as_secs_f64() * 1e3;
            let duration = samples.len() as f32 / sr as f32;
            dmap.insert(
                file.to_string(),
                Decoded {
                    file: file.to_string(),
                    samples,
                    sr,
                    duration,
                    decode_ms,
                },
            );
        }
    }

    // Fingerprint every file (audiofp ×3 + chromaprint).
    let mut fmap: Fmap = HashMap::new();
    for d in dmap.values() {
        fmap.insert(d.file.clone(), extract_all(d));
    }

    let cat = build_catalog(&tracks, &fmap);
    let id_rows = run_identification(&tracks, &fmap, &cat);

    if do_report {
        let mut out = String::new();
        out.push_str("# Chromaprint Bakeoff\n\n");
        out.push_str(
            "audiofp (Wang / Panako / Haitsma) vs chromaprint on the shared robustness\n\
             corpus (#87). Numbers below were generated by `bakeoff/`, not hand-entered.\n\n",
        );
        out.push_str("## Environment\n");
        out.push_str(&environment());
        out.push('\n');
        out.push_str("\n## Corpus\n");
        out.push_str(&corpus_lines(&tracks));
        out.push_str(&format!(
            "\nChromaprint reference fingerprint size (b64 chars): galway.flac {g} · freak.flac {f}\n",
            g = cp::to_base64(&fmap["galway.flac"].cp).len(),
            f = cp::to_base64(&fmap["freak.flac"].cp).len(),
        ));
        out.push_str("\n## Methodology\n");
        out.push_str(METHODOLOGY);
        out.push_str("\n## Results\n\n");
        out.push_str(&m1_table(&tracks, &fmap, &dmap));
        out.push_str(&format!(
            "\n### Identification (top-1, {} queries / 9-track catalog)\n",
            id_rows.len()
        ));
        out.push_str(&m2_table(&tracks, &cat, &id_rows));
        out.push_str("\n### Latency\n");
        out.push_str(&m3_table(&tracks, &dmap, &fmap, &id_rows));
        out.push_str("\n## Caveats\n");
        out.push_str(CAVEATS);
        out.push('\n');
        out.push_str("\n## Reproduce\n\n");
        out.push_str(
            "    sudo apt-get install -y libchromaprint-dev   # or: brew install chromaprint\n",
        );
        out.push_str("    cd bakeoff && cargo run --release -- --report\n");
        out.push_str("    cd bakeoff && cargo run --release -- --check\n");
        out.push_str(&format!(
            "\nGenerated: {date}, audiofp v{ver} (commit {commit}), chromaprint {cpv}\n",
            date = today(),
            ver = audiofp_version(),
            commit = git_commit(),
            cpv = cp::version(),
        ));
        print!("{out}");
    }

    if do_check {
        let violations = run_invariants(&tracks, &fmap, &id_rows, &cat);
        if violations.is_empty() {
            println!("all invariants held");
            std::process::exit(0);
        }
        println!("INVARIANT VIOLATIONS:");
        for v in &violations {
            println!("  - {v}");
        }
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jaccard_identical_is_one() {
        let a = vec![1u32, 2, 3];
        assert!((jaccard(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn jaccard_disjoint_is_zero() {
        let a = vec![1u32, 2];
        let b = vec![3u32, 4];
        assert!(jaccard(&a, &b) < 1e-6);
    }

    #[test]
    fn bit_sim_identical_is_one() {
        let a = vec![0xDEADBEEFu32, 0x1234_5678];
        assert!((bit_sim(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cp_shift_sim_identical_is_one_shift_zero() {
        let a: Vec<u32> = (0..16u32).map(|i| i.wrapping_mul(0x9E37_79B9)).collect();
        let (sim, k) = cp_shift_sim(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6);
        assert_eq!(k, 0);
    }

    #[test]
    fn cp_shift_sim_rotated_detects_shift() {
        let a: Vec<u32> = (0..16u32).map(|i| i.wrapping_mul(0x9E37_79B9)).collect();
        // b is `a` shifted right by 1 (a zero prepended): b[i+1] == a[i].
        // In `cp_shift_sim`'s convention a[i] ~ b[i+k], so k = +1.
        let b: Vec<u32> = std::iter::once(0u32).chain(a.iter().copied()).collect();
        let (sim, k) = cp_shift_sim(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
        assert_eq!(k, 1);
    }

    #[test]
    fn corpus_references_exist() {
        for t in corpus() {
            let p = std::path::Path::new(ASSETS).join(t.reference);
            assert!(p.exists(), "missing corpus reference {}", p.display());
            for v in t.variants {
                let p = std::path::Path::new(ASSETS).join(v);
                assert!(p.exists(), "missing corpus variant {}", p.display());
            }
        }
    }

    /// F24: WangIndex `None` must not use a sentinel slot index — `m2_table`
    /// and invariants must render/count no-match rows without indexing panic.
    #[test]
    fn m2_table_renders_wang_no_match_without_panic() {
        let tracks = corpus();
        let cat = Catalog {
            ids: vec![0, 1],
            wang: Vec::new(),
            panako: Vec::new(),
            haitsma: Vec::new(),
            cp: Vec::new(),
        };
        let rows = vec![IdRow {
            file: "synthetic.query".to_string(),
            source_track: 0,
            wang: MatchSlot {
                slot: None,
                score: 0.0,
                margin: 0.0,
            },
            panako: MatchSlot {
                slot: Some(0),
                score: 1.0,
                margin: 0.5,
            },
            haitsma: MatchSlot {
                slot: Some(0),
                score: 1.0,
                margin: 0.5,
            },
            cp: MatchSlot {
                slot: Some(1),
                score: 0.1,
                margin: 0.0,
            },
            wang_query_ms: 0.0,
            cp_query_ms: 0.0,
        }];
        let table = m2_table(&tracks, &cat, &rows);
        assert!(
            table.contains("no match"),
            "expected explicit no-match label, got:\n{table}"
        );
        assert!(
            table.contains("1 no-match"),
            "expected no-match count in summary, got:\n{table}"
        );
    }

    #[test]
    fn id_correct_treats_none_as_incorrect() {
        let cat = Catalog {
            ids: vec![0],
            wang: Vec::new(),
            panako: Vec::new(),
            haitsma: Vec::new(),
            cp: Vec::new(),
        };
        assert!(!id_correct(&cat, None, 0));
        assert!(id_correct(&cat, Some(0), 0));
    }

    #[test]
    fn extract_timed_kernel_includes_resample_work() {
        // Native 44.1 kHz buffer resampled down to 8 kHz: resample_ms must be
        // non-zero and kernel_ms must be >= resample_ms + extract_ms (within fp
        // noise — they are measured sequentially so kernel ≈ sum).
        let samples: Vec<f32> = (0..88_200).map(|i| (i as f32 * 0.001).sin()).collect();
        let (_, timing) = extract_timed::<Wang>(&samples, 44_100, 8_000);
        assert!(
            timing.resample_ms > 0.0,
            "expected resample stage to be timed, got {timing:?}"
        );
        assert!(
            timing.kernel_ms + 1e-6 >= timing.resample_ms + timing.extract_ms,
            "kernel must cover resample+extract, got {timing:?}"
        );
    }
}
