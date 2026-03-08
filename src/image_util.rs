use crate::{ChromaFormat, WaveletType};

/// Load an image as f32 RGB.
/// For 8-bit PNGs values are in [0, 255];
/// for 16-bit PNGs (10-bit content in high bits) values are in [0, 1023].
pub fn load_image_rgb_f32_bits(path: &str, bit_depth: u32) -> (Vec<f32>, u32, u32) {
    let img = image::open(path).expect("Failed to open image");
    if bit_depth == 10 {
        let img16 = img.to_rgb16();
        let (w, h) = img16.dimensions();
        // 10-bit content stored in the high bits of a 16-bit PNG: shift down 6 bits
        let pixels: Vec<f32> = img16.as_raw().iter().map(|&v| (v >> 6) as f32).collect();
        (pixels, w, h)
    } else {
        let img8 = img.to_rgb8();
        let (w, h) = img8.dimensions();
        let pixels: Vec<f32> = img8.as_raw().iter().map(|&v| v as f32).collect();
        (pixels, w, h)
    }
}

pub fn load_image_rgb_f32(path: &str) -> (Vec<f32>, u32, u32) {
    load_image_rgb_f32_bits(path, 8)
}

/// Save f32 RGB data as a PNG.
/// For bit_depth == 8: clamp to [0, 255], save as RGB8 PNG.
/// For bit_depth == 10: clamp to [0, 1023], shift up to 16-bit (<<6), save as RGB16 PNG.
pub fn save_image_rgb_f32_bits(path: &str, data: &[f32], width: u32, height: u32, bit_depth: u32) {
    if bit_depth == 10 {
        let samples: Vec<u16> = data
            .iter()
            .map(|&v| (v.round().clamp(0.0, 1023.0) as u16) << 6)
            .collect();
        let img: image::ImageBuffer<image::Rgb<u16>, Vec<u16>> =
            image::ImageBuffer::from_raw(width, height, samples)
                .expect("Failed to create 16-bit image");
        img.save(path).expect("Failed to save image");
    } else {
        let bytes: Vec<u8> = data
            .iter()
            .map(|&v| v.round().clamp(0.0, 255.0) as u8)
            .collect();
        let img = image::RgbImage::from_raw(width, height, bytes).expect("Failed to create image");
        img.save(path).expect("Failed to save image");
    }
}

pub fn save_image_rgb_f32(path: &str, data: &[f32], width: u32, height: u32) {
    save_image_rgb_f32_bits(path, data, width, height, 8);
}

pub fn parse_wavelet_type(s: &str) -> WaveletType {
    match s {
        "53" => WaveletType::LeGall53,
        "97" => WaveletType::CDF97,
        other => {
            eprintln!("Unknown wavelet type: {}. Use: 53 or 97", other);
            std::process::exit(1);
        }
    }
}

pub fn parse_chroma_format(s: &str) -> ChromaFormat {
    match s {
        "444" => ChromaFormat::Yuv444,
        "422" => ChromaFormat::Yuv422,
        "420" => ChromaFormat::Yuv420,
        other => {
            eprintln!("Unknown chroma format: {}. Use: 444, 422, or 420", other);
            std::process::exit(1);
        }
    }
}
