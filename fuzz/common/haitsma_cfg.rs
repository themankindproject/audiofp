// Valid HaitsmaConfig construction for fuzz targets.
//
// Random fmin/fmax pairs can leave bands uncovered (rejected by
// try_new) or use new() which panics. This helper only returns
// configs that pass validation.

use audiofp::classical::{Haitsma, HaitsmaConfig};

// Build a Haitsma config from fuzz bytes. Narrow or uncovered band ranges
// are rejected via `try_new` and the caller should skip the input.
pub fn haitsma_cfg_from_bytes(data: &[u8]) -> Option<HaitsmaConfig> {
    let b0 = data.first().copied().unwrap_or(0);
    let b1 = data.get(1).copied().unwrap_or(1);
    // Keep fmin/fmax inside the documented 300..2000 Hz envelope and
    // leave ≥ 100 Hz separation so every band receives at least one bin.
    let fmin = 300.0 + (b0 as f32) * 5.0; // 300..1550
    let fmax = (fmin + 100.0).max(400.0 + (b1 as f32) * 8.0).min(2000.0);
    let mut cfg = HaitsmaConfig::default();
    cfg.fmin = fmin;
    cfg.fmax = fmax;
    cfg.max_input_samples = None;
    Haitsma::try_new(cfg.clone()).ok().map(|_| cfg)
}
