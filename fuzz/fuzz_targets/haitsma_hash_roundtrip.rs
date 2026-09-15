#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use audiofp::classical::Haitsma;
use audiofp::{Fingerprinter, SampleRate};
use bytemuck;
use libfuzzer_sys::fuzz_target;

mod haitsma_cfg {
    include!("../common/haitsma_cfg.rs");
}
mod pcm_seed {
    include!("../common/pcm_seed.rs");
}

#[derive(Arbitrary, Debug)]
struct Input {
    /// Reserved for future structured fields; seed bytes drive PCM.
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
        pcm_seed::HAITSMA_MIN + 2_500,
    );
    let Some(cfg) = haitsma_cfg::haitsma_cfg_from_bytes(data) else {
        return;
    };

    let mut fp = match Haitsma::try_new(cfg) {
        Ok(fp) => fp,
        Err(_) => return,
    };
    let Ok(fpr) = fp.extract(&samples, SampleRate::HZ_5000) else {
        return;
    };

    for &frame in &fpr.frames {
        let bytes: &[u8] = bytemuck::bytes_of(&frame);
        let roundtripped: u32 = bytemuck::pod_read_unaligned(bytes);
        assert_eq!(frame, roundtripped);
    }

    let mut fp2 = Haitsma::try_new(haitsma_cfg::haitsma_cfg_from_bytes(data).unwrap()).unwrap();
    let fpr2 = fp2.extract(&samples, SampleRate::HZ_5000).unwrap();
    assert_eq!(fpr.frames, fpr2.frames, "determinism violation");
});
