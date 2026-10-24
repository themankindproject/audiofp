//! Regressions for the landmark extractors' streaming engine
//! (`src/classical/stream.rs`) and shared config clamping: push-size
//! independence under a pending-anchor cap, push-after-flush, offline dB
//! parity at the Nyquist bin, and the peak floor on digital silence.

mod common;

use audiofp::classical::{
    PanakoConfig, StreamingPanako, StreamingWang, Wang, WangConfig, WangHash,
};
use audiofp::{Fingerprinter, SampleRate, StreamingFingerprinter};

/// The shared two-tone signal at 8 kHz.
fn synth(seed: u32, len: usize) -> Vec<f32> {
    common::audio_gen::two_tone(seed, 8_000, len)
}

fn sorted(mut v: Vec<WangHash>) -> Vec<(u32, u32)> {
    let mut out: Vec<_> = v.drain(..).map(|h| (h.t_anchor, h.hash)).collect();
    out.sort_unstable();
    out
}

fn stream_wang(cfg: WangConfig, audio: &[f32], chunk: usize) -> Vec<WangHash> {
    let mut s = StreamingWang::new(cfg);
    let mut out = Vec::new();
    for c in audio.chunks(chunk) {
        out.extend(s.push(c).unwrap().into_iter().map(|(_, h)| h));
    }
    out.extend(s.flush_complete().unwrap().into_iter().map(|(_, h)| h));
    out
}

/// With `max_pending_anchors` set, one large push used to finalise every
/// ripe second before emitting anything, so the queue grew to seconds of
/// anchors and the cap evicted them (741 vs 3844 hashes on 20 s at
/// `Some(100)`). The cap must bind the same way for any push size.
#[test]
fn pending_anchor_cap_does_not_depend_on_push_size() {
    let audio = synth(7, 8_000 * 20);
    let cfg = WangConfig {
        max_pending_anchors: Some(100),
        ..WangConfig::default()
    };
    let chunked = sorted(stream_wang(cfg.clone(), &audio, 1_024));
    let single = sorted(stream_wang(cfg, &audio, audio.len()));
    assert!(!chunked.is_empty());
    assert_eq!(
        single.len(),
        chunked.len(),
        "single push vs 1024-sample pushes"
    );
    assert_eq!(single, chunked);

    let mut s = StreamingPanako::new(PanakoConfig {
        max_pending_anchors: Some(100),
        ..PanakoConfig::default()
    });
    let mut one = s.push(&audio).unwrap().len();
    one += s.flush_complete().unwrap().len();
    let mut s = StreamingPanako::new(PanakoConfig {
        max_pending_anchors: Some(100),
        ..PanakoConfig::default()
    });
    let mut many = 0;
    for c in audio.chunks(1_024) {
        many += s.push(c).unwrap().len();
    }
    many += s.flush_complete().unwrap().len();
    assert_eq!(one, many, "Panako single push vs 1024-sample pushes");
}

/// Pushing after a `flush` that ended mid-second is valid and appends to
/// the stream. The rest of that second's peaks used to be stranded until
/// the next flush: frames 149–187 got zero hashes during the pushes.
#[test]
fn push_after_mid_second_flush_keeps_emitting_that_second() {
    let audio = synth(11, 8_000 * 8);
    // 20 000 samples = 2.5 s: the flush lands inside second 2 (frames
    // 125–187 at 62.5 fps).
    let (head, tail) = audio.split_at(20_000);
    let mut s = StreamingWang::default();
    s.push(head).unwrap();
    s.flush().unwrap();
    let mut during_pushes = Vec::new();
    for c in tail.chunks(1_024) {
        during_pushes.extend(s.push(c).unwrap().into_iter().map(|(_, h)| h));
    }
    let rest_of_second_2 = during_pushes
        .iter()
        .filter(|h| (149..=187).contains(&h.t_anchor))
        .count();
    assert!(
        rest_of_second_2 > 0,
        "anchors in the rest of the flushed second must emit on later pushes"
    );
}

/// `x[n] = 2a` on even samples, `0` on odd: DC and Nyquist bins carry the
/// same power. Streaming used to convert bin 512 of every row on the
/// scalar `log2` path while offline converts it on the vector path; the two
/// differ by 1 ULP for this `a`, which flips peak ties at bin 512. Length is
/// chosen so the offline frame count is a multiple of 8 (no scalar tail in
/// offline either), so the two must agree exactly. (Wang only: the dB path
/// is shared `StreamCore` code, and Panako's 2·fan_out target cap breaks
/// exact ties differently from offline on a signal this degenerate.)
#[test]
fn streaming_matches_offline_at_the_nyquist_bin() {
    let a = 0.117_737_12_f32;
    let n_frames = 320; // multiple of 8
    let len = 1_024 + 128 * (n_frames - 1);
    let audio: Vec<f32> = (0..len)
        .map(|n| if n % 2 == 0 { 2.0 * a } else { 0.0 })
        .collect();

    let offline = Wang::default()
        .extract(&audio, SampleRate::HZ_8000)
        .unwrap()
        .hashes;
    for chunk in [1_024, 4_096, len] {
        let streamed = stream_wang(WangConfig::default(), &audio, chunk);
        assert_eq!(
            sorted(streamed),
            sorted(offline.clone()),
            "chunk {chunk}: streaming must match offline"
        );
    }
}

/// A peak floor below the −120 dB log floor used to turn digital silence
/// into a plateau of peaks (900 Wang / 450 Panako hashes for 4 s of zeros
/// at `min_anchor_mag_db = -121`), so silence matched silence.
#[test]
fn silence_has_no_hashes_at_any_peak_floor() {
    let silence = vec![0.0_f32; 8_000 * 4];
    for floor in [-121.0, -150.0, -200.0, f32::NEG_INFINITY] {
        let wang = Wang::new(WangConfig {
            min_anchor_mag_db: floor,
            ..WangConfig::default()
        })
        .extract(&silence, SampleRate::HZ_8000)
        .unwrap();
        assert!(wang.hashes.is_empty(), "Wang floor {floor}");
        let panako = audiofp::classical::Panako::new(PanakoConfig {
            min_anchor_mag_db: floor,
            ..PanakoConfig::default()
        })
        .extract(&silence, SampleRate::HZ_8000)
        .unwrap();
        assert!(panako.hashes.is_empty(), "Panako floor {floor}");
    }
}
