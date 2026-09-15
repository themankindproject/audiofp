#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use audiofp::classical::{Haitsma, StreamingHaitsma};
use audiofp::{Fingerprinter, SampleRate, StreamingFingerprinter};
use libfuzzer_sys::fuzz_target;

mod haitsma_cfg {
    include!("../common/haitsma_cfg.rs");
}
mod pcm_seed {
    include!("../common/pcm_seed.rs");
}

#[derive(Arbitrary, Debug)]
struct Input {
    _pad: u8,
}

fuzz_target!(|data: &[u8]| {
    let Ok(_input) = Unstructured::new(data).arbitrary::<Input>() else {
        return;
    };

    let samples = pcm_seed::synth_pcm(
        data,
        5_000,
        pcm_seed::HAITSMA_MIN,
        pcm_seed::HAITSMA_MIN + 5_000,
    );
    let Some(cfg) = haitsma_cfg::haitsma_cfg_from_bytes(data) else {
        return;
    };

    let mut offline = match Haitsma::try_new(cfg.clone()) {
        Ok(h) => h,
        Err(_) => return,
    };
    let Ok(off) = offline.extract(&samples, SampleRate::HZ_5000) else {
        return;
    };

    let chunk = pcm_seed::chunk_size(data, 5_000);
    let mut stream = match StreamingHaitsma::try_new(cfg) {
        Ok(s) => s,
        Err(_) => return,
    };
    let mut online: Vec<u32> = Vec::new();
    for c in samples.chunks(chunk) {
        online.extend(stream.push(c).unwrap().into_iter().map(|(_, h)| h));
    }
    online.extend(stream.flush().unwrap().into_iter().map(|(_, h)| h));

    assert_eq!(off.frames, online);
});
