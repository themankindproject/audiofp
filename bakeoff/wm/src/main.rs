//! Classical-watermark algorithmic bakeoff driver.
//!
//! For each candidate algorithm: embed a fixed payload into each corpus
//! track, run the full attack matrix, blind-detect, and report
//! BER / recovery / SNR / LSD / false-positive behaviour.
//!
//! Usage:
//!   cargo run --release                 # full matrix
//!   cargo run --release -- --quick      # subset (2 codecs, 1 track)

use audiofp_wm_bakeoff::{algo, attack, dsp, io};

use algo::{Candidate, LogDsss, MagDsss, Qim};
use attack::{Work, ber, log_spectral_distance, run_codec, run_filter, run_resample, snr_db};
use dsp::ChaCha;
use std::path::{Path, PathBuf};

const SR: u32 = 16_000;
const N_BITS: usize = 64;

/// One corpus track: (name, source file). ffmpeg decodes to mono 16 kHz.
struct Track {
    name: &'static str,
    path: &'static str,
}

fn corpus() -> Vec<Track> {
    vec![
        Track {
            name: "galway",
            path: "assets/galway.flac",
        },
        Track {
            name: "freak",
            path: "assets/freak.flac",
        },
        Track {
            name: "acidjazz",
            path: "assets/acidjazz.flac",
        },
        Track {
            name: "digya",
            path: "assets/digya.flac",
        },
        Track {
            name: "grieg",
            path: "assets/grieg.ogg",
        },
        Track {
            name: "mozart",
            path: "assets/mozart.ogg",
        },
    ]
}

/// One attack: name + a closure that maps src -> dst.
struct AttackSpec {
    name: &'static str,
    kind: AttackKind,
}

enum AttackKind {
    Codec(&'static [&'static str]), // ffmpeg output args
    Filter(&'static str),
    Resample(u32, u32), // from, to
    None,
}

fn codecs() -> Vec<AttackSpec> {
    vec![
        AttackSpec {
            name: "mp3-64",
            kind: AttackKind::Codec(&["-c:a", "libmp3lame", "-b:a", "64k"]),
        },
        AttackSpec {
            name: "mp3-96",
            kind: AttackKind::Codec(&["-c:a", "libmp3lame", "-b:a", "96k"]),
        },
        AttackSpec {
            name: "mp3-128",
            kind: AttackKind::Codec(&["-c:a", "libmp3lame", "-b:a", "128k"]),
        },
        AttackSpec {
            name: "aac-64",
            kind: AttackKind::Codec(&["-c:a", "aac", "-b:a", "64k"]),
        },
        AttackSpec {
            name: "aac-96",
            kind: AttackKind::Codec(&["-c:a", "aac", "-b:a", "96k"]),
        },
        AttackSpec {
            name: "aac-128",
            kind: AttackKind::Codec(&["-c:a", "aac", "-b:a", "128k"]),
        },
        AttackSpec {
            name: "opus-32",
            kind: AttackKind::Codec(&["-c:a", "libopus", "-b:a", "32k"]),
        },
        AttackSpec {
            name: "opus-64",
            kind: AttackKind::Codec(&["-c:a", "libopus", "-b:a", "64k"]),
        },
        AttackSpec {
            name: "opus-96",
            kind: AttackKind::Codec(&["-c:a", "libopus", "-b:a", "96k"]),
        },
        AttackSpec {
            name: "opus-128",
            kind: AttackKind::Codec(&["-c:a", "libopus", "-b:a", "128k"]),
        },
        AttackSpec {
            name: "vorbis-q2",
            kind: AttackKind::Codec(&["-c:a", "libvorbis", "-q:a", "2"]),
        },
        AttackSpec {
            name: "vorbis-q4",
            kind: AttackKind::Codec(&["-c:a", "libvorbis", "-q:a", "4"]),
        },
    ]
}

fn signal_attacks() -> Vec<AttackSpec> {
    vec![
        AttackSpec {
            name: "gain-0.5",
            kind: AttackKind::Filter("volume=0.5"),
        },
        AttackSpec {
            name: "gain-2.0",
            kind: AttackKind::Filter("volume=2.0"),
        },
        AttackSpec {
            name: "peak-norm",
            kind: AttackKind::Filter("dynaudnorm=p=0.9"),
        },
        AttackSpec {
            name: "lowpass-4k",
            kind: AttackKind::Filter("lowpass=f=4000"),
        },
        AttackSpec {
            name: "lowpass-6k",
            kind: AttackKind::Filter("lowpass=f=6000"),
        },
        AttackSpec {
            name: "highpass-300",
            kind: AttackKind::Filter("highpass=f=300"),
        },
        AttackSpec {
            name: "bandpass-300-5k",
            kind: AttackKind::Filter("highpass=f=300,lowpass=f=5000"),
        },
        AttackSpec {
            name: "eq-boost-1k",
            kind: AttackKind::Filter("equalizer=f=1000:width_type=o:width=1:g=6"),
        },
        AttackSpec {
            name: "noise-30db",
            kind: AttackKind::Filter("anoisesrc=d=60:c=pink:a=0.01,amix=inputs=2:duration=first"),
        },
        AttackSpec {
            name: "resample-8k",
            kind: AttackKind::Resample(SR, 8_000),
        },
        AttackSpec {
            name: "resample-22k",
            kind: AttackKind::Resample(SR, 22_050),
        },
        AttackSpec {
            name: "resample-44k",
            kind: AttackKind::Resample(SR, 44_100),
        },
    ]
}

fn temporal_attacks() -> Vec<AttackSpec> {
    vec![
        AttackSpec {
            name: "lead-trim-0.3s",
            kind: AttackKind::Filter("atrim=start=0.3"),
        },
        AttackSpec {
            name: "mid-extract",
            kind: AttackKind::Filter("atrim=start=3.0:end=13.0"),
        },
        AttackSpec {
            name: "tail-trim-3s",
            kind: AttackKind::Filter("atrim=end=12.0"),
        },
        AttackSpec {
            name: "tms-1.02",
            kind: AttackKind::Filter("atempo=1.02"),
        },
        AttackSpec {
            name: "tms-0.98",
            kind: AttackKind::Filter("atempo=0.98"),
        },
        AttackSpec {
            name: "requant-8bit",
            kind: AttackKind::Filter("aformat=sample_fmts=u8"),
        },
        AttackSpec {
            name: "channel-gain-0.7",
            kind: AttackKind::Filter("volume=0.7"),
        },
        AttackSpec {
            name: "clip-drive",
            kind: AttackKind::Filter("volume=4.0,alimiter=limit=0.9"),
        },
    ]
}

fn decode_to_wav(src: &Path, dst: &Path, sr: u32) {
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-i"])
        .arg(src)
        .args(["-ac", "1", "-ar", &sr.to_string(), "-c:a", "pcm_s16le"])
        .arg(dst)
        .status()
        .expect("ffmpeg spawn failed");
    assert!(status.success(), "decode failed: {src:?}");
}

struct Row {
    attack: String,
    ber: f64,
    recovered: bool,
    score: f64,
    snr: f64,
    lsd: f64,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let quick = args.iter().any(|a| a == "--quick");

    let work = Work::new(SR);
    let key = b"bakeoff-secret-key-0123456789".to_vec();
    let wrong_key = b"DIFFERENT-secret-key-98765432".to_vec();

    // Deterministic payload.
    let mut prng = ChaCha::new(b"payload", 0);
    let payload: Vec<u8> = (0..N_BITS).map(|_| (prng.u32() & 1) as u8).collect();

    let tracks = if quick { &corpus()[..1] } else { &corpus()[..] };

    let mut candidates: Vec<Box<dyn Candidate>> = vec![
        Box::new(MagDsss::new()),
        Box::new(LogDsss::new()),
        Box::new(Qim::new()),
    ];

    let mut all_attacks = codecs();
    all_attacks.extend(signal_attacks());
    all_attacks.extend(temporal_attacks());
    if quick {
        all_attacks.truncate(4);
    }

    let mut results: Vec<(&str, Vec<Row>, Vec<f64>, f64, f64)> = Vec::new();

    for cand in candidates.iter_mut() {
        eprintln!("\n=========== candidate: {} ===========", cand.name());
        let name = cand.name();
        let mut rows = Vec::new();

        // Per-track clean embedding is done once, then attacked.
        let mut embed_snr_acc = 0.0;
        let mut clean_lsd_acc = 0.0;
        let mut n_tracks = 0.0;

        // False-positive: scores on unwatermarked audio across tracks & attacks.
        let mut fp_scores: Vec<f64> = Vec::new();
        // Wrong-key false negatives (should FAIL to produce payload).
        let mut wrongkey_fail = 0usize;
        let mut wrongkey_total = 0usize;

        let mut attack_names: Vec<String> = Vec::new();

        for track in tracks {
            let src_wav = work.dir.join(format!("{}_src.wav", track.name));
            decode_to_wav(Path::new(track.path), &src_wav, SR);
            let samples = io::read_wav_mono_f32(&src_wav).unwrap();
            let (wm, _esnr) = cand.embed(&samples, &payload, &key);
            let n = samples.len().min(wm.len());
            let esnr = snr_db(&samples[..n], &wm[..n]);
            embed_snr_acc += esnr;
            let lsd = log_spectral_distance(&samples, &wm, 1024, 256);
            clean_lsd_acc += lsd;
            n_tracks += 1.0;

            let wm_wav = work.dir.join(format!("{}_wm.wav", track.name));
            io::write_wav_mono_f32(&wm_wav, &wm, SR).unwrap();

            // Clean self-detection baseline.
            let (bits, score) = cand.detect(&wm, N_BITS, &key);
            eprintln!(
                "  track {} clean: BER={:.3} score={:.4} embedSNR={:.1}dB LSD={:.1}dB",
                track.name,
                ber(&bits, &payload),
                score,
                esnr,
                lsd
            );

            // False-positive on the unwatermarked source.
            let (_b, fpscore) = cand.detect(&samples, N_BITS, &key);
            fp_scores.push(fpscore);

            // Wrong key on the watermarked audio.
            let (wbits, _ws) = cand.detect(&wm, N_BITS, &wrong_key);
            wrongkey_total += 1;
            if ber(&wbits, &payload) > 0.35 {
                wrongkey_fail += 1;
            }

            for atk in &all_attacks {
                if !attack_names.contains(&atk.name.to_string()) {
                    attack_names.push(atk.name.to_string());
                }
                let out_wav = work.dir.join(format!(
                    "{}__{}__{}.wav",
                    name,
                    track.name,
                    atk.name
                ));
                match &atk.kind {
                    AttackKind::Codec(cargs) => {
                        let v: Vec<String> = cargs.iter().map(|s| s.to_string()).collect();
                        run_codec(&wm_wav, &out_wav, &v);
                    }
                    AttackKind::Filter(f) => run_filter(&wm_wav, &out_wav, f, SR),
                    AttackKind::Resample(a, b) => run_resample(&wm_wav, &out_wav, *a, *b),
                    AttackKind::None => {
                        std::fs::copy(&wm_wav, &out_wav).unwrap();
                    }
                }
                let attacked = io::read_wav_mono_f32(&out_wav).unwrap();
                let (dbits, dscore) = cand.detect(&attacked, N_BITS, &key);
                let b = ber(&dbits, &payload);
                let rec = b < 0.10;
                let n = samples.len().min(attacked.len());
                let snr = snr_db(&samples[..n], &attacked[..n]);
                let l = log_spectral_distance(&samples, &attacked, 1024, 256);
                rows.push(Row {
                    attack: atk.name.to_string(),
                    ber: b,
                    recovered: rec,
                    score: dscore,
                    snr,
                    lsd: l,
                });
            }
        }

        let embed_snr = embed_snr_acc / n_tracks;
        let clean_lsd = clean_lsd_acc / n_tracks;
        results.push((name, rows, fp_scores, embed_snr, clean_lsd));

        eprintln!(
            "  wrong-key rejected {wrongkey_fail}/{wrongkey_total} tracks"
        );
    }

    // ---- Report ----
    println!("# Classical Watermark Bakeoff — Raw Results\n");
    println!("Payload: {N_BITS} bits (deterministic). Sample rate: {SR} Hz.");
    println!("Recovery threshold: BER < 0.10 (blind detection, correct key).\n");

    for (name, rows, fp, esnr, lsd) in &results {
        println!("## `{name}`\n");
        println!("- Embed distortion: nominal SNR {esnr:.1} dB, clean-audio LSD {lsd:.1} dB");
        let fp_mean = fp.iter().sum::<f64>() / fp.len().max(1) as f64;
        let fp_max = fp.iter().cloned().fold(f64::MIN, f64::max);
        println!(
            "- False-positive score on unwatermarked audio: mean {fp_mean:.4}, max {fp_max:.4}"
        );
        let n = rows.len();
        let rec = rows.iter().filter(|r| r.recovered).count();
        let mean_ber = rows.iter().map(|r| r.ber).sum::<f64>() / n.max(1) as f64;
        println!(
            "- Attacks: {rec}/{n} recovered ({:.0}%), mean BER {mean_ber:.3}\n",
            100.0 * rec as f64 / n.max(1) as f64
        );

        // Per-attack aggregation: need rows grouped by attack name.
        let mut by_attack: Vec<(String, f64, usize, usize)> = Vec::new();
        for r in rows {
            if let Some(e) = by_attack.iter_mut().find(|e| e.0 == r.attack) {
                e.1 += r.ber;
                e.2 += 1;
                if r.recovered {
                    e.3 += 1;
                }
            } else {
                by_attack.push((
                    r.attack.clone(),
                    r.ber,
                    1,
                    if r.recovered { 1 } else { 0 },
                ));
            }
        }
        println!("| attack | mean BER | recovered |");
        println!("|---|---|---|");
        for (a, bersum, cnt, r) in &by_attack {
            let mb = bersum / *cnt as f64;
            println!("| {a} | {mb:.3} | {r}/{cnt} |");
        }
        println!();
    }

    // Cross-candidate summary.
    println!("## Summary\n");
    println!("| candidate | attacks recovered | mean BER | embed SNR (dB) | clean LSD (dB) |");
    println!("|---|---|---|---|---|");
    for (name, rows, _, esnr, lsd) in &results {
        let n = rows.len();
        let rec = rows.iter().filter(|r| r.recovered).count();
        let mb = rows.iter().map(|r| r.ber).sum::<f64>() / n.max(1) as f64;
        println!(
            "| {name} | {rec}/{n} | {mb:.3} | {esnr:.1} | {lsd:.1} |"
        );
    }
}
