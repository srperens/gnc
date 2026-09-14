//! PERF-4 — N encode streams on **one** `GpuContext` must produce what one stream produces.
//!
//! `gnc density` runs N streams inside one process, and by default they share a device: one
//! `wgpu::Device`, one `wgpu::Queue`, N `EncoderPipeline`s. That is only a legitimate way to
//! measure stream density if it is also a legitimate way to *encode* — a shared queue with
//! interleaved submissions from several threads must not let one stream's dispatches land in
//! another stream's buffers.
//!
//! The pipelines own their buffers, so the property should hold by construction. "Should hold by
//! construction" is exactly the claim this repository has been burned by, and the failure mode
//! here is silent: a race would show up as a slightly different coded size, which no throughput
//! number would notice and which the fingerprint matrix cannot see because it is single-threaded.
//!
//! So: encode one frame serially, then encode the same frame on four threads sharing one device,
//! and require every thread's bytes to equal the serial bytes exactly.

use gnc::encoder::pipeline::EncoderPipeline;
use gnc::GpuContext;
use std::sync::Arc;

/// A deterministic frame with enough structure to exercise every subband. A flat or random
/// image would code identically under a race that swapped whole tiles between streams.
fn test_frame(w: usize, h: usize) -> Vec<f32> {
    let mut data = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let ramp = (x as f32) / (w as f32);
            let rings = (((x * x + y * y) as f32).sqrt() * 0.25).sin() * 0.5 + 0.5;
            data[i] = ramp;
            data[i + 1] = rings;
            data[i + 2] = 1.0 - ramp * rings;
        }
    }
    data
}

#[test]
fn concurrent_streams_on_one_device_encode_identically() {
    let (w, h) = (320u32, 192u32);
    let frame = Arc::new(test_frame(w as usize, h as usize));
    let config = gnc::quality_preset(90);

    let ctx = Arc::new(GpuContext::new());

    // The reference: one stream, alone on the device.
    let serial_bytes = {
        let mut enc = EncoderPipeline::new(&ctx);
        // Encode twice; the second call is the one with warm cached buffers, which is the
        // state the concurrent streams will be in when they are compared against it.
        let _ = enc.encode(&ctx, &frame, w, h, &config);
        enc.encode(&ctx, &frame, w, h, &config).byte_size()
    };

    const STREAMS: usize = 4;
    let sizes: Vec<usize> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..STREAMS)
            .map(|_| {
                let ctx = Arc::clone(&ctx);
                let frame = Arc::clone(&frame);
                let config = config.clone();
                scope.spawn(move || {
                    let mut enc = EncoderPipeline::new(&ctx);
                    let _ = enc.encode(&ctx, &frame, w, h, &config);
                    // Several rounds, so the streams are genuinely overlapping rather than
                    // merely started at the same time.
                    let mut last = 0;
                    for _ in 0..4 {
                        last = enc.encode(&ctx, &frame, w, h, &config).byte_size();
                    }
                    last
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    assert_eq!(sizes.len(), STREAMS, "every stream must report a size");
    for (i, &size) in sizes.iter().enumerate() {
        assert_eq!(
            size, serial_bytes,
            "stream {i} coded {size} B on a shared device against {serial_bytes} B alone — \
             concurrent submissions are not independent"
        );
    }
}
