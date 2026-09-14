#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use audiofp::classical::{Panako, PanakoConfig, StreamingPanako};
use audiofp::{Fingerprinter, SampleRate, StreamingFingerprinter};
use libfuzzer_sys::fuzz_target;

mod pcm_seed {
    include!("../common/pcm_seed.rs");
}

#[derive(Arbitrary, Debug)]
struct Input {
    fan_out: u16,
    target_zone_t: u16,
    peaks_per_sec: u16,
}

fuzz_target!(|data: &[u8]| {
    let Ok(input) = Unstructured::new(data).arbitrary::<Input>() else {
        return;
    };

    let samples = pcm_seed::synth_pcm(data, 8_000, pcm_seed::WANG_MIN, pcm_seed::WANG_MIN + 8_000);

    let cfg = {
        let mut c = PanakoConfig::default();
        c.fan_out = input.fan_out.max(1).min(10);
        c.target_zone_t = input.target_zone_t.max(1).min(128);
        c.peaks_per_sec = input.peaks_per_sec.max(1).min(60);
        c
    };

    let mut offline = Panako::new(cfg.clone());
    let Ok(off) = offline.extract(&samples, SampleRate::HZ_8000) else {
        return;
    };

    let chunk = pcm_seed::chunk_size(data, 8_000);
    let mut stream = StreamingPanako::new(cfg);
    let mut online = Vec::new();
    for c in samples.chunks(chunk) {
        online.extend(stream.push(c).unwrap().into_iter().map(|(_, h)| h));
    }
    online.extend(stream.flush_complete().unwrap().into_iter().map(|(_, h)| h));

    let mut a = off.hashes;
    let mut b = online;
    a.sort_unstable_by_key(|h| (h.t_anchor, h.t_b, h.t_c, h.hash));
    b.sort_unstable_by_key(|h| (h.t_anchor, h.t_b, h.t_c, h.hash));
    assert_eq!(a, b);
});
