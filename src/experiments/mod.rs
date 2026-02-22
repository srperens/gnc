/// Experiment configurations.
/// Each experiment is a named set of codec parameters.
use crate::CodecConfig;

pub struct Experiment {
    pub name: String,
    pub description: String,
    pub config: CodecConfig,
}

/// Phase 1 baseline experiments: sweep quantization steps
pub fn phase1_experiments() -> Vec<Experiment> {
    let steps = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0];

    steps
        .iter()
        .map(|&step| Experiment {
            name: format!("baseline_q{}", step as u32),
            description: format!(
                "YCoCg-R + LeGall 5/3 (3-level) + rANS + uniform quantization (step={})",
                step
            ),
            config: CodecConfig {
                tile_size: 256,
                quantization_step: step,
                dead_zone: 0.0,
                wavelet_levels: 3,
            },
        })
        .collect()
}

/// Compare wavelet decomposition levels at a fixed quantization step
pub fn wavelet_level_experiments() -> Vec<Experiment> {
    (1..=4)
        .map(|levels| Experiment {
            name: format!("levels_{}", levels),
            description: format!(
                "LeGall 5/3 with {} decomposition level(s), qstep=4",
                levels
            ),
            config: CodecConfig {
                tile_size: 256,
                quantization_step: 4.0,
                dead_zone: 0.0,
                wavelet_levels: levels,
            },
        })
        .collect()
}
