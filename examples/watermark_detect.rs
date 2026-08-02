//! Detect an AudioSeal-compatible watermark with a BYO ONNX model.
//!
//! ```bash
//! cargo run --example watermark_detect --features watermark -- path/to/model.onnx [audio.wav]
//! ```
//!
//! Download / export detector weights from
//! <https://github.com/facebookresearch/audioseal> — audiofp does not
//! bundle them. Without a second path argument, runs on 1 s of silence
//! at 16 kHz (useful as a smoke test that the model loads).

use audiofp::watermark::{WatermarkConfig, WatermarkDetector};
use audiofp::{SampleRate};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model = args.next().ok_or_else(|| {
        "usage: watermark_detect <model.onnx> [audio.wav]\n\
         \n\
         Get an ONNX detector from https://github.com/facebookresearch/audioseal\n\
         (audiofp does not ship the weights)."
            .to_string()
    })?;
    let audio_path = args.next();

    let cfg = WatermarkConfig::new(&model);
    println!(
        "Loading {model} (message_bits={}, threshold={}, sr={})…",
        cfg.message_bits, cfg.threshold, cfg.sample_rate
    );
    let mut det = WatermarkDetector::new(cfg)?;

    let (samples, rate) = if let Some(path) = audio_path {
        let target = det.config().sample_rate;
        println!("Decoding {path} → mono @ {target} Hz…");
        let samples = audiofp::io::decode_to_mono_at(&path, target)?;
        let rate = SampleRate::new(target).ok_or("sample_rate must be non-zero")?;
        (samples, rate)
    } else {
        let sr = det.config().sample_rate;
        let samples = vec![0.0_f32; sr as usize]; // 1 s silence
        let rate = SampleRate::new(sr).ok_or("sample_rate must be non-zero")?;
        println!("No audio path given — using 1 s of silence @ {sr} Hz");
        (samples, rate)
    };

    let r = det.detect(&samples, rate)?;
    println!(
        "detected={} confidence={:.4} message={:#018b}",
        r.detected, r.confidence, r.message
    );
    println!(
        "localization: {} scores (flattened ONNX output[0]; time base is model-specific)",
        r.localization.len()
    );

    Ok(())
}
