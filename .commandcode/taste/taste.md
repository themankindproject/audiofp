# Taste (Continuously Learned by [CommandCode][cmd])

[cmd]: https://commandcode.ai/

# Performance
- Use `[profile.release]` with `lto = "fat"` and `codegen-units = 1` for DSP-heavy Rust crates. Confidence: 0.80
- Always run `cargo test` and `cargo clippy` before benchmarking to verify optimizations preserve correctness. Confidence: 0.85
- Pre-compute and cache all scratch buffers at construction time for streaming hot paths (zero allocation after warmup). Confidence: 0.90
- Use `cargo bench` before/after to quantify each optimization with clear throughput metrics. Confidence: 0.80
- Use `partition_point()` (binary search) on sorted arrays to narrow inner loop ranges instead of linear scans with break. Confidence: 0.75

# Code Style
- Replace `Option<u8>` with a `u8::MAX` sentinel value in tight loops to enable auto-vectorization. Confidence: 0.70
- Use sparse (CSR) representation for matrix-vector products when each row has few non-zero elements vs total columns. Confidence: 0.75
- Skip the per-bin `sqrt()` in STFT when the next pipeline stage applies `log10` (use `10·log10(power)` instead of `20·log10(magnitude)`). Confidence: 0.85

# Git
- Do not reference the AI assistant in git commit messages. Confidence: 0.70

