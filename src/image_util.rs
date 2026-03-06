use crate::{ChromaFormat, WaveletType};

pub fn load_image_rgb_f32(path: &str) -> (Vec<f32>, u32, u32) {
    let img = image::open(path).expect("Failed to open image").to_rgb8();
    let (w, h) = img.dimensions();
    let pixels: Vec<f32> = img.as_raw().iter().map(|&v| v as f32).collect();
    (pixels, w, h)
}

pub fn save_image_rgb_f32(path: &str, data: &[f32], width: u32, height: u32) {
    let bytes: Vec<u8> = data
        .iter()
        .map(|&v| v.round().clamp(0.0, 255.0) as u8)
        .collect();
    let img = image::RgbImage::from_raw(width, height, bytes).expect("Failed to create image");
    img.save(path).expect("Failed to save image");
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
