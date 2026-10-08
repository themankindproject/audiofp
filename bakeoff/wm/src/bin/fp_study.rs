//! False-positive / wrong-key calibration study for the winning design.
//!
//! The sync phase search maximises the statistic over `n_bits` trials, which
//! inflates the null distribution. This measures the null vs signal
//! distributions so a detection threshold can be calibrated honestly.

use audiofp_wm_bakeoff::algo::{Candidate, MagDsss};
use audiofp_wm_bakeoff::{dsp, io};
use std::path::Path;

fn dec(src: &Path, sr: u32) -> Vec<f32> {
    let t = std::env::temp_dir().join("fp_study.wav");
    std::process::Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-i"])
        .arg(src)
        .args(["-ac", "1", "-ar", &sr.to_string(), "-c:a", "pcm_s16le"])
        .arg(&t)
        .status()
        .unwrap();
    io::read_wav_mono_f32(&t).unwrap()
}

fn ber(a: &[u8], b: &[u8]) -> f64 {
    (0..a.len()).filter(|&i| a[i] != b[i]).count() as f64 / a.len() as f64
}

fn stats(v: &mut Vec<f64>) -> (f64, f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    let mean = v.iter().sum::<f64>() / n as f64;
    let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
    (mean, sd, v[(n * 5 / 100).min(n - 1)], v[n - 1])
}

fn main() {
    let sr = 16_000;
    let key = b"bakeoff-secret-key-0123456789";
    let mut p = dsp::ChaCha::new(b"payload", 0);
    let payload: Vec<u8> = (0..64).map(|_| (p.u32() & 1) as u8).collect();
    let tracks = [
        "assets/galway.flac",
        "assets/freak.flac",
        "assets/acidjazz.flac",
        "assets/digya.flac",
        "assets/grieg.ogg",
        "assets/mozart.ogg",
    ];
    let mut c = MagDsss::new();

    let mut h1 = Vec::new(); // watermarked, correct key   (signal)
    let mut h0_unwm = Vec::new(); // unwatermarked, correct key  (null A)
    let mut h0_wrongkey = Vec::new(); // watermarked, wrong key      (null B)
    let mut h0_1char = Vec::new(); // watermarked, 1-char key diff (null C)
    let mut ber_correct = Vec::new();
    let mut ber_wrongkey = Vec::new();

    for t in tracks {
        let x = dec(Path::new(t), sr);
        let (wm, _) = c.embed(&x, &payload, key);
        let (b1, s1) = c.detect(&wm, 64, key);
        h1.push(s1);
        ber_correct.push(ber(&b1, &payload));
        let (_b0, s0) = c.detect(&x, 64, key);
        h0_unwm.push(s0);
        let (bw, sw) = c.detect(&wm, 64, b"totally-different-key-000000000");
        h0_wrongkey.push(sw);
        ber_wrongkey.push(ber(&bw, &payload));
        let (_, sp) = c.detect(&wm, 64, b"bakeoff-secret-key-0123456788");
        h0_1char.push(sp);
        println!(
            "{t}: signal={s1:.2} unwm={s0:.2} wrongkey={sw:.2} 1char={sp:.2} BERok={:.3} BERwk={:.3}",
            ber(&b1, &payload),
            ber(&bw, &payload)
        );
    }

    let (m1, sd1, _, mx1) = stats(&mut h1);
    let (m0, sd0, p95, mx0) = stats(&mut h0_unwm);
    let (mwk, sdwk, _, mxwk) = stats(&mut h0_wrongkey);
    let (mp, sdp, _, mxp) = stats(&mut h0_1char);

    println!("\n-- distributions --");
    println!("signal  (wm, correct key): mean={m1:.2} sd={sd1:.2} max={mx1:.2}");
    println!("null A (unwatermarked):    mean={m0:.2} sd={sd0:.2} p95={p95:.2} max={mx0:.2}");
    println!("null B (wrong key):        mean={mwk:.2} sd={sdwk:.2} max={mxwk:.2}");
    println!("null C (1-char key diff):  mean={mp:.2} sd={sdp:.2} max={mxp:.2}");
    let sep = (m1 - m0) / sd0;
    println!("\nseparation (signal mean - null mean)/null sd = {sep:.1} sigma");
    println!(
        "BER correct key: mean={:.4} max={:.4}",
        ber_correct.iter().sum::<f64>() / 6.0,
        ber_correct.iter().cloned().fold(0.0, f64::max)
    );
    println!(
        "BER wrong key:   mean={:.4}",
        ber_wrongkey.iter().sum::<f64>() / 6.0
    );

    println!("\n-- payload discrimination (same audio, different payload) --");
    let x = dec(Path::new(tracks[0]), sr);
    let pl2: Vec<u8> = payload.iter().map(|b| 1 - b).collect();
    let (wm1, _) = c.embed(&x, &payload, key);
    let (wm2, _) = c.embed(&x, &pl2, key);
    let (b1, _) = c.detect(&wm1, 64, key);
    let (b2, _) = c.detect(&wm2, 64, key);
    println!("wm(payload) vs payload:      BER={:.3}", ber(&b1, &payload));
    println!("wm(payload) vs inverted:     BER={:.3}", ber(&b1, &pl2));
    println!("wm(pl2)     vs pl2:          BER={:.3}", ber(&b2, &pl2));
}
