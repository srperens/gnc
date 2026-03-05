// Temporal LeGall 5/3 Wavelet Lifting (per-element, two-pass)
//
// Applies 5/3 lifting across 4 frames in the wavelet domain.
// Two separate compute passes required:
//   Pass 1 (predict): compute highpass d0, d1 from input frames
//   Pass 2 (update):  compute lowpass  s0, s1 from inputs + d0, d1
//
// Forward (pass=0, predict):
//   d0 = f1 - 0.5 * (f0 + f2)
//   d1 = f3 - f2               (reflect boundary: f4 = f2)
//
// Forward (pass=1, update):
//   s0 = f0 + 0.5 * d0         (reflect boundary: d_{-1} = d0)
//   s1 = f2 + 0.25 * (d0 + d1)
//
// Inverse (pass=0, update-undo):
//   x0 = s0 - 0.5 * d0
//   x2 = s1 - 0.25 * (d0 + d1)
//
// Inverse (pass=1, predict-undo):
//   x1 = d0 + 0.5 * (x0 + x2)
//   x3 = d1 + x2
//
// Dispatch: (ceil(count / 256), 1, 1)  — per tile, not per frame

struct Params {
    count: u32,      // total number of f32 elements (padded_w * padded_h)
    direction: u32,  // 0 = forward, 1 = inverse
    pass_idx: u32,   // 0 = first pass (predict/update-undo), 1 = second pass (update/predict-undo)
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;

// For forward predict: in0=f0, in1=f1, in2=f2, in3=f3, out0=d0, out1=d1
// For forward update:  in0=f0, in1=d0, in2=f2, in3=d1, out0=s0, out1=s1
// For inverse update-undo: in0=s0, in1=d0, in2=s1, in3=d1, out0=x0, out1=x2
// For inverse predict-undo: in0=x0, in1=d0, in2=x2, in3=d1, out0=x1, out1=x3
@group(0) @binding(1) var<storage, read> in0: array<f32>;
@group(0) @binding(2) var<storage, read> in1: array<f32>;
@group(0) @binding(3) var<storage, read> in2: array<f32>;
@group(0) @binding(4) var<storage, read> in3: array<f32>;
@group(0) @binding(5) var<storage, read_write> out0: array<f32>;
@group(0) @binding(6) var<storage, read_write> out1: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.count {
        return;
    }

    let a = in0[idx];
    let b = in1[idx];
    let c = in2[idx];
    let d = in3[idx];

    if params.direction == 0u {
        // Forward
        if params.pass_idx == 0u {
            // Predict: d0 = f1 - 0.5*(f0 + f2), d1 = f3 - f2
            let d0 = b - 0.5 * (a + c);
            let d1 = d - c;
            out0[idx] = d0;
            out1[idx] = d1;
        } else {
            // Update: s0 = f0 + 0.5*d0, s1 = f2 + 0.25*(d0 + d1)
            // in0=f0, in1=d0, in2=f2, in3=d1
            out0[idx] = a + 0.5 * b;
            out1[idx] = c + 0.25 * (b + d);
        }
    } else {
        // Inverse
        if params.pass_idx == 0u {
            // Undo update: x0 = s0 - 0.5*d0, x2 = s1 - 0.25*(d0+d1)
            // in0=s0, in1=d0, in2=s1, in3=d1
            out0[idx] = a - 0.5 * b;
            out1[idx] = c - 0.25 * (b + d);
        } else {
            // Undo predict: x1 = d0 + 0.5*(x0+x2), x3 = d1 + x2
            // in0=x0, in1=d0, in2=x2, in3=d1
            out0[idx] = b + 0.5 * (a + c);
            out1[idx] = d + c;
        }
    }
}
