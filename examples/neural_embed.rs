//! Extract embeddings with a BYO ONNX log-mel model.
//!
//! ```bash
//! cargo run --example neural_embed --features neural -- path/to/model.onnx
//! ```
//!
//! `audiofp` does not ship embedder weights. Point `--` at any ONNX file
//! whose first input is `[1, n_mels, n_frames]` log-mel and whose first
//! output is a flat embedding vector (see `USAGE.md` → Neural Embedder).

use audiofp::neural::{NeuralEmbedder, NeuralEmbedderConfig};
use audiofp::{Fingerprinter, SampleRate};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args().nth(1).ok_or_else(|| {
        "usage: neural_embed <model.onnx>\n\
         \n\
         Bring your own ONNX log-mel embedder — audiofp does not bundle weights.\n\
         See USAGE.md → Neural Embedder → Model contract."
            .to_string()
    })?;

    let cfg = NeuralEmbedderConfig::new(&model);
    println!(
        "Loading {model} (sr={}, n_mels={}, window={}s)…",
        cfg.sample_rate, cfg.n_mels, cfg.window_secs
    );
    let mut emb = NeuralEmbedder::new(cfg)?;

    // ≥ one analysis window of silence at the model's rate.
    let n = emb.window_samples().max(emb.hop_samples() * 2);
    let samples = vec![0.0_f32; n];
    let rate =
        SampleRate::new(emb.config().sample_rate).ok_or("model sample_rate must be non-zero")?;
    let fp = emb.extract(&samples, rate)?;

    println!(
        "{} embedding(s), dim={}, frames_per_sec={:.3}",
        fp.embeddings.len(),
        fp.embedding_dim,
        fp.frames_per_sec
    );
    if let Some(e) = fp.embeddings.first() {
        let preview: Vec<String> = e.vector.iter().take(8).map(|v| format!("{v:.4}")).collect();
        println!(
            "  first @ {} ms: [{}{}]",
            e.t_start.0,
            preview.join(", "),
            if e.vector.len() > 8 { ", …" } else { "" }
        );
    }

    Ok(())
}
