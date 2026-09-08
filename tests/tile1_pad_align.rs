//! TILE-1 stage 1: pad the plane to 32, not to the tile size.
//!
//! Interior tiles stay 256. The last row and column are shorter, every extent a multiple of
//! 32. Resolutions that already fill the 256 grid must not grow.

use gnc::{ChromaFormat, FrameInfo, PLANE_PAD_ALIGN};

fn info(w: u32, h: u32) -> FrameInfo {
    FrameInfo::new(w, h, 8, 256, ChromaFormat::Yuv444)
}

#[test]
fn pad_align_is_thirty_two() {
    assert_eq!(PLANE_PAD_ALIGN, 32);
}

#[test]
fn whole_tiles_do_not_grow() {
    for (w, h) in [(256, 256), (512, 512), (1024, 512), (256, 512)] {
        let i = info(w, h);
        assert_eq!(i.padded_width(), w, "{w}x{h} width");
        assert_eq!(i.padded_height(), h, "{w}x{h} height");
        assert_eq!(i.tiles_x() * i.tile_size, w);
        assert_eq!(i.tiles_y() * i.tile_size, h);
    }
}

#[test]
fn broadcast_heights_pad_to_thirty_two() {
    let i = info(1920, 1080);
    assert_eq!(i.padded_width(), 1920);
    assert_eq!(i.padded_height(), 1088);
    assert_eq!(i.tiles_x(), 8);
    assert_eq!(i.tiles_y(), 5);
    assert_eq!(i.tile_extent(7, 4), (128, 64));
    // 8 of 1088 rows are outside the picture = 0.7% of coded samples.
    let coded = i.padded_width() * i.padded_height();
    let extra = coded - 1920 * 1080;
    assert!((extra as f64 / coded as f64 - 0.0074).abs() < 0.0002);
}

#[test]
fn gp18_legacy_still_pads_to_the_tile() {
    let mut i = info(1920, 1080);
    i.plane_pad_align = i.tile_size;
    assert_eq!(i.padded_width(), 2048);
    assert_eq!(i.padded_height(), 1280);
}

#[test]
fn chroma_pads_independently() {
    // 1366x768 4:2:0: luma 1366 -> 1376, chroma 683 -> 704. 1376/2 = 688, not 704.
    let i = FrameInfo::new(1366, 768, 8, 256, ChromaFormat::Yuv420);
    assert_eq!(i.padded_width(), 1376);
    assert_eq!(i.chroma_width(), 683);
    assert_eq!(i.chroma_padded_width(), 704);
    assert_ne!(i.chroma_padded_width(), i.padded_width() / 2);
}
