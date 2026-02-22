use serde::Serialize;
use std::time::{Duration, Instant};

/// Throughput measurement result
#[derive(Debug, Clone, Serialize)]
pub struct ThroughputMetrics {
    pub encode_time_ms: f64,
    pub decode_time_ms: f64,
    pub encode_fps: f64,
    pub decode_fps: f64,
    pub width: u32,
    pub height: u32,
    pub megapixels_per_sec_encode: f64,
    pub megapixels_per_sec_decode: f64,
}

impl std::fmt::Display for ThroughputMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Encode: {:.2} ms ({:.1} fps, {:.1} MP/s) | Decode: {:.2} ms ({:.1} fps, {:.1} MP/s)",
            self.encode_time_ms,
            self.encode_fps,
            self.megapixels_per_sec_encode,
            self.decode_time_ms,
            self.decode_fps,
            self.megapixels_per_sec_decode,
        )
    }
}

/// Time a closure, returning the result and elapsed duration.
pub fn timed<F, R>(f: F) -> (R, Duration)
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed();
    (result, elapsed)
}

/// Run encode/decode multiple times and compute average throughput.
pub fn measure_throughput<E, D>(
    encode_fn: E,
    decode_fn: D,
    width: u32,
    height: u32,
    iterations: u32,
) -> ThroughputMetrics
where
    E: Fn() -> (),
    D: Fn() -> (),
{
    // Warmup
    encode_fn();
    decode_fn();

    // Measure encode
    let start = Instant::now();
    for _ in 0..iterations {
        encode_fn();
    }
    let encode_total = start.elapsed();

    // Measure decode
    let start = Instant::now();
    for _ in 0..iterations {
        decode_fn();
    }
    let decode_total = start.elapsed();

    let encode_ms = encode_total.as_secs_f64() * 1000.0 / iterations as f64;
    let decode_ms = decode_total.as_secs_f64() * 1000.0 / iterations as f64;
    let mpixels = (width as f64 * height as f64) / 1_000_000.0;

    ThroughputMetrics {
        encode_time_ms: encode_ms,
        decode_time_ms: decode_ms,
        encode_fps: 1000.0 / encode_ms,
        decode_fps: 1000.0 / decode_ms,
        width,
        height,
        megapixels_per_sec_encode: mpixels * 1000.0 / encode_ms,
        megapixels_per_sec_decode: mpixels * 1000.0 / decode_ms,
    }
}
