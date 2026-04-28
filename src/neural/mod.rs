//! Generic ONNX log-mel audio embedder.
//!
//! [`NeuralEmbedder`] wraps a user-supplied ONNX model that takes a
//! `[1, n_mels, n_frames] f32` log-mel spectrogram and returns a
//! `[1, embedding_dim]` (or `[1, 1, embedding_dim]`) `f32` embedding
//! vector. The crate ships only the front-end (STFT → log-mel) and the
//! ONNX runtime glue — bring your own model.
//!
//! The model contract is:
//!
//! - **Input 0:** `[1, n_mels, n_frames] f32`. `n_mels` is set by
//!   [`NeuralEmbedderConfig::n_mels`] and `n_frames` is fully determined
//!   by `(window_samples − n_fft) / hop + 1` (non-centred STFT).
//! - **Output 0:** the embedding vector. Any shape that flattens to a
//!   non-empty `f32` vector is accepted; the flat length becomes the
//!   embedding dimension.
//!
//! The shape is concretised **once at construction** and the model is
//! optimised + made runnable then; per-call work is just the front-end
//! (windowed FFT + log-mel) plus the inference itself. There is no
//! per-call graph rebuild.
//!
//! Available only when the `neural` feature is enabled:
//!
//! ```toml
//! [dependencies]
//! audiofp = { version = "0.2", features = ["neural"] }
//! ```
//!
//! # Example
//!
//! ```no_run
//! use audiofp::neural::{NeuralEmbedder, NeuralEmbedderConfig};
//! use audiofp::{AudioBuffer, Fingerprinter, SampleRate};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut emb = NeuralEmbedder::new(NeuralEmbedderConfig::new("my_model.onnx"))?;
//! let samples: Vec<f32> = vec![/* … 16 kHz mono PCM … */];
//! let buf = AudioBuffer { samples: &samples, rate: SampleRate::HZ_16000 };
//! let fp = emb.extract(buf)?;
//! println!("{} embeddings of dim {}", fp.embeddings.len(), fp.embedding_dim);
//! # Ok(()) }
//! ```

pub mod embedder;
pub(crate) mod frontend;
pub mod streaming;

#[cfg(test)]
pub(crate) mod test_support;

pub use embedder::{NeuralEmbedder, NeuralEmbedderConfig, NeuralEmbedding, NeuralFingerprint};
pub use streaming::StreamingNeuralEmbedder;
