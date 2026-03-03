use crate::TemporalTransform;

pub(crate) fn group_size(mode: TemporalTransform) -> usize {
    match mode {
        TemporalTransform::None => 1,
        TemporalTransform::Haar => 2,
        TemporalTransform::LeGall53 => 4,
    }
}

pub fn is_power_of_two(n: usize) -> bool {
    n >= 2 && (n & (n - 1)) == 0
}

/// Forward Haar wavelet across 2 frames (lifting form, no normalization).
/// high = b - a
/// low = a + high / 2
pub(crate) fn haar_forward(a: &[f32], b: &[f32]) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(a.len(), b.len(), "Haar forward: frame size mismatch");
    let mut low = vec![0.0f32; a.len()];
    let mut high = vec![0.0f32; a.len()];
    for i in 0..a.len() {
        let av = a[i];
        let bv = b[i];
        let h = bv - av;
        high[i] = h;
        low[i] = av + 0.5 * h;
    }
    (low, high)
}

/// Inverse Haar wavelet across 2 frames (lifting form, no normalization).
/// a = low - high / 2, b = high + a
pub(crate) fn haar_inverse(low: &[f32], high: &[f32]) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(low.len(), high.len(), "Haar inverse: frame size mismatch");
    let mut a = vec![0.0f32; low.len()];
    let mut b = vec![0.0f32; low.len()];
    for i in 0..low.len() {
        let l = low[i];
        let h = high[i];
        let av = l - 0.5 * h;
        a[i] = av;
        b[i] = h + av;
    }
    (a, b)
}

/// Multi-level Haar forward over a group of 2^N frames.
/// Returns (final low, highs_per_level) where level 0 is the finest split.
pub fn haar_multilevel_forward(
    frames: &[&[f32]],
) -> (Vec<f32>, Vec<Vec<Vec<f32>>>) {
    let n = frames.len();
    assert!(is_power_of_two(n), "Haar multilevel requires 2^N frames");
    let mut current: Vec<Vec<f32>> = frames.iter().map(|f| f.to_vec()).collect();
    let mut highs_per_level: Vec<Vec<Vec<f32>>> = Vec::new();

    while current.len() > 1 {
        let mut next_lows: Vec<Vec<f32>> = Vec::with_capacity(current.len() / 2);
        let mut highs: Vec<Vec<f32>> = Vec::with_capacity(current.len() / 2);
        for i in (0..current.len()).step_by(2) {
            let (low, high) = haar_forward(&current[i], &current[i + 1]);
            next_lows.push(low);
            highs.push(high);
        }
        highs_per_level.push(highs);
        current = next_lows;
    }

    (current.remove(0), highs_per_level)
}

/// Multi-level Haar inverse for a group of 2^N frames.
pub fn haar_multilevel_inverse(
    low: &[f32],
    highs_per_level: &[Vec<Vec<f32>>],
) -> Vec<Vec<f32>> {
    let mut current: Vec<Vec<f32>> = vec![low.to_vec()];
    for level in (0..highs_per_level.len()).rev() {
        let highs = &highs_per_level[level];
        assert_eq!(
            current.len(),
            highs.len(),
            "Haar inverse: mismatched low/high counts at level {level}"
        );
        let mut next: Vec<Vec<f32>> = Vec::with_capacity(current.len() * 2);
        for i in 0..current.len() {
            let (a, b) = haar_inverse(&current[i], &highs[i]);
            next.push(a);
            next.push(b);
        }
        current = next;
    }
    current
}

/// Forward LeGall 5/3 wavelet across 4 frames.
/// Produces two low frames (s0, s1) and two high frames (d0, d1).
pub(crate) fn legall53_forward_4(
    f0: &[f32],
    f1: &[f32],
    f2: &[f32],
    f3: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = f0.len();
    assert_eq!(n, f1.len(), "5/3 forward: frame size mismatch");
    assert_eq!(n, f2.len(), "5/3 forward: frame size mismatch");
    assert_eq!(n, f3.len(), "5/3 forward: frame size mismatch");
    let mut s0 = vec![0.0f32; n];
    let mut s1 = vec![0.0f32; n];
    let mut d0 = vec![0.0f32; n];
    let mut d1 = vec![0.0f32; n];
    for i in 0..n {
        // Predict odd samples
        let x0 = f0[i];
        let x1 = f1[i];
        let x2 = f2[i];
        let x3 = f3[i];
        let d0i = x1 - 0.5 * (x0 + x2);
        // Symmetric extension at end: x4 = x2
        let d1i = x3 - x2;
        // Update even samples (s0 uses symmetric d_{-1} = d0)
        let s0i = x0 + 0.5 * d0i;
        let s1i = x2 + 0.25 * (d0i + d1i);
        s0[i] = s0i;
        s1[i] = s1i;
        d0[i] = d0i;
        d1[i] = d1i;
    }
    (s0, s1, d0, d1)
}

/// Inverse LeGall 5/3 wavelet across 4 frames.
pub(crate) fn legall53_inverse_4(
    s0: &[f32],
    s1: &[f32],
    d0: &[f32],
    d1: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = s0.len();
    assert_eq!(n, s1.len(), "5/3 inverse: frame size mismatch");
    assert_eq!(n, d0.len(), "5/3 inverse: frame size mismatch");
    assert_eq!(n, d1.len(), "5/3 inverse: frame size mismatch");
    let mut f0 = vec![0.0f32; n];
    let mut f1 = vec![0.0f32; n];
    let mut f2 = vec![0.0f32; n];
    let mut f3 = vec![0.0f32; n];
    for i in 0..n {
        let d0i = d0[i];
        let d1i = d1[i];
        let s0i = s0[i];
        let s1i = s1[i];
        // Inverse update
        let x0 = s0i - 0.5 * d0i;
        let x2 = s1i - 0.25 * (d0i + d1i);
        // Inverse predict
        let x1 = d0i + 0.5 * (x0 + x2);
        let x3 = d1i + x2;
        f0[i] = x0;
        f1[i] = x1;
        f2[i] = x2;
        f3[i] = x3;
    }
    (f0, f1, f2, f3)
}
