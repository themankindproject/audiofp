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

/// Fingerprint data + kernel timing (median of `REPETITIONS`) for one file.
struct Fps {
    wang: WangFingerprint,
    panako: PanakoFingerprint,
    haitsma: HaitsmaFingerprint,
    cp: Vec<u32>,
    wang_ms: f64,
    panako_ms: f64,
    haitsma_ms: f64,
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

/// Resample (if needed) then extract, `REPETITIONS` times.
///
/// Returns the fingerprint (last run) and the median wall time (ms) of the
/// resample + extract step. The resampler is built once and reused; the
/// timing includes the resample because that is part of what a caller
/// would pay when starting from native-rate PCM.
fn extract_timed<F: Fingerprinter + Default>(
    samples: &[f32],
    native: u32,
    target: u32,
) -> (F::Output, f64) {
    let mut f = F::default();
    let rate = SampleRate::new(target).expect("non-zero target rate");
    let resampler = (native != target).then(|| SincResampler::new(native, target));
    let input: Vec<f32> = match &resampler {
        Some(r) => r.process(samples),
        None => samples.to_vec(),
    };
    let mut times = Vec::with_capacity(REPETITIONS);
    let mut fp = f.extract(&input, rate).expect("extract");
    for _ in 0..REPETITIONS {
        let t = Instant::now();
        fp = f.extract(&input, rate).expect("extract");
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    (fp, median(&times))
}

/// Extract every fingerprint + kernel timing for one decoded file.
fn extract_all(d: &Decoded) -> Fps {
    let (wang, wang_ms) = extract_timed::<Wang>(&d.samples, d.sr, 8_000);
    let (panako, panako_ms) = extract_timed::<Panako>(&d.samples, d.sr, 8_000);
    let (haitsma, haitsma_ms) = extract_timed::<Haitsma>(&d.samples, d.sr, 5_000);
    let mut cp_times = Vec::with_capacity(REPETITIONS);
    let mut cp_raw = Vec::new();
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
        wang_ms,
        panako_ms,
        haitsma_ms,
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

/// One identified query row.
struct IdRow {
    file: String,
    source_track: usize,
    /// (catalog slot, score, margin) — margin is 0 for the index path
    /// (it returns a single winner, not a ranked list).
    wang: (usize, f32, f32),
    panako: (usize, f32, f32),
    haitsma: (usize, f32, f32),
    cp: (usize, f32, f32),
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
                Some((slot, r)) => (slot, r.score, 0.0),
                None => (usize::MAX, 0.0, 0.0),
            };
            let wang_query_ms = t.elapsed().as_secs_f64() * 1e3;
            // Panako / Haitsma: 1:1-vote max over the 9 refs (secondary).
            let panako = best_of(&f.panako, &cat.panako, |q, r| {
                panako_m.match_one(q, r).score
            });
            let haitsma = best_of(&f.haitsma, &cat.haitsma, |q, r| {
                haitsma_m.match_one(q, r).score
            });
            // Chromaprint: best-shift-hamming argmin.
            let t = Instant::now();
            let cp = cp_top1(&f.cp, &cat.cp);
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
- **Latency** (M3): median of 3, release (lto=fat), single-threaded both
  systems. Kernel = fingerprint on already-decoded PCM (resample included);
  e2e = decode + kernel.
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
        let w_ok = cat.ids[r.wang.0] == r.source_track;
        let p_ok = cat.ids[r.panako.0] == r.source_track;
        let h_ok = cat.ids[r.haitsma.0] == r.source_track;
        let c_ok = cat.ids[r.cp.0] == r.source_track;
        s.push_str(&format!(
            "| {} | {} | {} {} | {} {} | {} {} | {} {} | {} |\n",
            r.file,
            src,
            tracks[cat.ids[r.wang.0]].name,
            tick(w_ok),
            tracks[cat.ids[r.panako.0]].name,
            tick(p_ok),
            tracks[cat.ids[r.haitsma.0]].name,
            tick(h_ok),
            tracks[cat.ids[r.cp.0]].name,
            tick(c_ok),
            fmt3(r.cp.2),
        ));
    }
    let wang_correct = rows
        .iter()
        .filter(|r| cat.ids[r.wang.0] == r.source_track)
        .count();
    let panako_correct = rows
        .iter()
        .filter(|r| cat.ids[r.panako.0] == r.source_track)
        .count();
    let haitsma_correct = rows
        .iter()
        .filter(|r| cat.ids[r.haitsma.0] == r.source_track)
        .count();
    let cp_correct = rows
        .iter()
        .filter(|r| cat.ids[r.cp.0] == r.source_track)
        .count();
    s.push_str(&format!(
        "\nTop-1 / {}: audiofp Wang {} (primary) · Panako {} · Haitsma {} · chromaprint {}\n",
        rows.len(),
        wang_correct,
        panako_correct,
        haitsma_correct,
        cp_correct,
    ));
    s
}

/// M3 latency tables.
fn m3_table(tracks: &[Track], dmap: &Dmap, fmap: &Fmap, rows: &[IdRow]) -> String {
    let mut s = String::new();
    s.push_str("Per file (median of 3). e2e = decode + kernel (3 audiofp kernels averaged).\n\n");
    s.push_str(
        "| file | dur(s) | decode(ms) | e2e-audiofp(ms) | e2e-cp(ms) | Wang(ms) | Panako(ms) | Haitsma(ms) | cp(ms) |\n",
    );
    s.push_str("|---|---|---|---|---|---|---|---|---|\n");
    for track in tracks {
        for &file in track.variants {
            let d = &dmap[file];
            let f = &fmap[file];
            s.push_str(&format!(
                "| {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} |\n",
                file,
                d.duration,
                d.decode_ms,
                d.decode_ms + (f.wang_ms + f.panako_ms + f.haitsma_ms) / 3.0,
                d.decode_ms + f.cp_ms,
                f.wang_ms,
                f.panako_ms,
                f.haitsma_ms,
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

    // 3. Identification floors: Wang ≥ 15/18 AND chromaprint ≥ 15/18.
    {
        let w = rows
            .iter()
            .filter(|r| cat.ids[r.wang.0] == r.source_track)
            .count();
        let c = rows
            .iter()
            .filter(|r| cat.ids[r.cp.0] == r.source_track)
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
            if f.wang_ms > 500.0 || f.panako_ms > 500.0 || f.haitsma_ms > 500.0 {
                v.push(format!(
                    "kernel latency {file}: Wang {:.0}/Panako {:.0}/Haitsma {:.0} ms ≥ 500 ms",
                    f.wang_ms, f.panako_ms, f.haitsma_ms
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
}
