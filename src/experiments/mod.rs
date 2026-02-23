/// Experiment configurations.
/// Each experiment is a named set of codec parameters.
use crate::{CodecConfig, SubbandWeights, WaveletType};

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
                subband_weights: SubbandWeights::uniform(3),
                cfl_enabled: false,
                ..Default::default()
            },
        })
        .collect()
}

/// Sweep dead-zone width at several quantization steps.
/// Dead zone is expressed as a fraction of step_size.
/// Values below dead_zone * step_size are mapped to zero.
pub fn dead_zone_experiments() -> Vec<Experiment> {
    let steps = [4.0, 8.0, 16.0];
    let zones = [0.0, 0.25, 0.5, 0.75, 1.0];

    let mut exps = Vec::new();
    for &step in &steps {
        for &dz in &zones {
            exps.push(Experiment {
                name: format!("dz{:.0}_q{}", dz * 100.0, step as u32),
                description: format!(
                    "dead_zone={:.2} * step, qstep={}, 3-level wavelet",
                    dz, step
                ),
                config: CodecConfig {
                    tile_size: 256,
                    quantization_step: step,
                    dead_zone: dz,
                    wavelet_levels: 3,
                    subband_weights: SubbandWeights::uniform(3),
                    cfl_enabled: false,
                    ..Default::default()
                },
            });
        }
    }
    exps
}

/// Compare wavelet decomposition levels at a fixed quantization step
pub fn wavelet_level_experiments() -> Vec<Experiment> {
    (1..=4)
        .map(|levels| Experiment {
            name: format!("levels_{}", levels),
            description: format!("LeGall 5/3 with {} decomposition level(s), qstep=4", levels),
            config: CodecConfig {
                tile_size: 256,
                quantization_step: 4.0,
                dead_zone: 0.0,
                wavelet_levels: levels,
                subband_weights: SubbandWeights::uniform(levels),
                cfl_enabled: false,
                ..Default::default()
            },
        })
        .collect()
}

/// Subband weight experiments: compare uniform vs perceptual vs aggressive
/// across multiple quantization steps.
pub fn subband_weight_experiments() -> Vec<Experiment> {
    let steps = [4.0, 8.0, 16.0];
    let levels = 3u32;

    let presets: Vec<(&str, &str, SubbandWeights)> = vec![
        (
            "uniform",
            "all weights = 1.0 (baseline behavior)",
            SubbandWeights::uniform(levels),
        ),
        (
            "perceptual",
            "perceptual weights (HH harder, inner harder, chroma 1.5x)",
            SubbandWeights::perceptual(levels),
        ),
        (
            "aggressive_hh",
            "aggressive HH quantization (4x at innermost)",
            SubbandWeights {
                ll: 1.0,
                detail: vec![[1.0, 1.0, 2.0], [1.0, 1.0, 3.0], [1.0, 1.0, 4.0]],
                chroma_weight: 1.0,
            },
        ),
        (
            "chroma_save",
            "uniform luma, chroma 2x multiplier",
            SubbandWeights {
                ll: 1.0,
                detail: vec![[1.0, 1.0, 1.0]; levels as usize],
                chroma_weight: 2.0,
            },
        ),
        (
            "full_perceptual",
            "perceptual subband + aggressive chroma (2x)",
            SubbandWeights {
                ll: 1.0,
                detail: vec![[1.0, 1.0, 1.5], [1.5, 1.5, 2.0], [2.0, 2.0, 3.0]],
                chroma_weight: 2.0,
            },
        ),
    ];

    let mut exps = Vec::new();
    for &step in &steps {
        for (name, desc, ref weights) in &presets {
            exps.push(Experiment {
                name: format!("sb_{}_q{}", name, step as u32),
                description: format!("qstep={}, {}", step, desc),
                config: CodecConfig {
                    tile_size: 256,
                    quantization_step: step,
                    dead_zone: 0.0,
                    wavelet_levels: levels,
                    subband_weights: weights.clone(),
                    cfl_enabled: false,
                    ..Default::default()
                },
            });
        }
    }
    exps
}

/// CfL (Chroma-from-Luma) experiments: compare CfL on/off across quantization steps.
pub fn cfl_experiments() -> Vec<Experiment> {
    let steps = [4.0, 8.0, 16.0];
    let levels = 3u32;

    let mut exps = Vec::new();
    for &step in &steps {
        // Without CfL
        exps.push(Experiment {
            name: format!("cfl_off_q{}", step as u32),
            description: format!("qstep={}, CfL disabled (baseline)", step),
            config: CodecConfig {
                tile_size: 256,
                quantization_step: step,
                dead_zone: 0.0,
                wavelet_levels: levels,
                subband_weights: SubbandWeights::uniform(levels),
                cfl_enabled: false,
                ..Default::default()
            },
        });
        // With CfL
        exps.push(Experiment {
            name: format!("cfl_on_q{}", step as u32),
            description: format!("qstep={}, CfL enabled", step),
            config: CodecConfig {
                tile_size: 256,
                quantization_step: step,
                dead_zone: 0.0,
                wavelet_levels: levels,
                subband_weights: SubbandWeights::uniform(levels),
                cfl_enabled: true,
                ..Default::default()
            },
        });
    }
    exps
}

/// Wavelet type experiments: compare LeGall 5/3 vs CDF 9/7 at several quantization steps.
pub fn wavelet_experiments() -> Vec<Experiment> {
    let steps = [4.0, 8.0, 16.0];
    let levels = 3u32;

    let mut exps = Vec::new();
    for &step in &steps {
        exps.push(Experiment {
            name: format!("wavelet_53_q{}", step as u32),
            description: format!("qstep={}, LeGall 5/3 wavelet", step),
            config: CodecConfig {
                tile_size: 256,
                quantization_step: step,
                dead_zone: 0.0,
                wavelet_levels: levels,
                subband_weights: SubbandWeights::uniform(levels),
                cfl_enabled: false,
                wavelet_type: WaveletType::LeGall53,
                ..Default::default()
            },
        });
        exps.push(Experiment {
            name: format!("wavelet_97_q{}", step as u32),
            description: format!("qstep={}, CDF 9/7 wavelet", step),
            config: CodecConfig {
                tile_size: 256,
                quantization_step: step,
                dead_zone: 0.0,
                wavelet_levels: levels,
                subband_weights: SubbandWeights::uniform(levels),
                cfl_enabled: false,
                wavelet_type: WaveletType::CDF97,
                ..Default::default()
            },
        });
    }
    exps
}

/// Combined dead-zone + subband weight experiments.
/// Tests whether dead-zone and perceptual weights stack for additive compression gains.
pub fn combined_dz_subband_experiments() -> Vec<Experiment> {
    let steps = [4.0, 8.0, 16.0];
    let zones = [0.0, 0.5, 0.75];
    let levels = 3u32;

    let weight_presets: Vec<(&str, SubbandWeights)> = vec![
        ("uniform", SubbandWeights::uniform(levels)),
        ("perceptual", SubbandWeights::perceptual(levels)),
        (
            "full_perceptual",
            SubbandWeights {
                ll: 1.0,
                detail: vec![[1.0, 1.0, 1.5], [1.5, 1.5, 2.0], [2.0, 2.0, 3.0]],
                chroma_weight: 2.0,
            },
        ),
    ];

    let mut exps = Vec::new();
    for &step in &steps {
        for &dz in &zones {
            for (wname, ref weights) in &weight_presets {
                exps.push(Experiment {
                    name: format!("combo_{}_dz{}_q{}", wname, (dz * 100.0) as u32, step as u32),
                    description: format!("qstep={}, dz={:.2}, weights={}", step, dz, wname),
                    config: CodecConfig {
                        tile_size: 256,
                        quantization_step: step,
                        dead_zone: dz,
                        wavelet_levels: levels,
                        subband_weights: weights.clone(),
                        cfl_enabled: false,
                        ..Default::default()
                    },
                });
            }
        }
    }
    exps
}
