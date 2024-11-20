use anyhow::Result;
use dicom::object::open_file;
use dicom::pixeldata::image;
use dicom::pixeldata::image::ImageBuffer;
use dicom_pixeldata::{ndarray::Array4, ConvertOptions, VoiLutOption};
use ndarray::Axis;

use devscol::{create_slice, trimming_box_with_resample, ImageFilter, PredicateSource};
use dicom::pixeldata::PixelDecoder;
use log::{debug, warn};
use std::{ffi::OsStr, path::PathBuf, sync::Once};
use walkdir::WalkDir;

static INIT: Once = Once::new();
fn setup() {
    INIT.call_once(|| {
        env_logger::init();
    });
}

fn test_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests")
}

fn dicom_directory() -> PathBuf {
    test_directory().join("data/dicom")
}

fn tmp_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
}

#[ignore]
#[test]
fn test_dicom_trim() -> Result<()> {
    setup();
    let out_dir = tmp_directory();
    let dcm_dir = dicom_directory();
    for entry in WalkDir::new(dcm_dir) {
        let entry = entry?;
        if entry.path().extension() == Some(OsStr::new("dcm")) {
            println!("entry: {:?}", entry);
            let obj = open_file(entry.path())?;
            let pixel_data = obj.decode_pixel_data()?;
            if pixel_data.number_of_frames() > 1 {
                warn!(
                    "Using first frame although there are {} frames in the dicom",
                    pixel_data.number_of_frames()
                );
            }
            let options = ConvertOptions::new()
                .with_voi_lut(VoiLutOption::Normalize)
                .force_8bit();
            let u8arr4d: Array4<u8> = pixel_data.to_ndarray_with_options(&options)?;
            let now = std::time::Instant::now();

            // Remove frame and channel dimensions
            let u8arr2d = u8arr4d
                .index_axis_move(Axis(0), 0)
                .index_axis_move(Axis(2), 0);

            let img = u8arr2d.mapv(|e| e as i16);

            let resample_step = img.ncols().max(img.nrows()) / 1000 + 1;
            debug!("resample_step: {:?}", resample_step);
            let bbox = trimming_box_with_resample(
                img.view(),
                vec![PredicateSource {
                    filter: ImageFilter::Original,
                    quantile_min: Some(0.01),
                    quantile_max: Some(0.99),
                    raw: false,
                }],
                resample_step,
            )?;
            let u8arr = u8arr2d
                .slice(create_slice(&bbox, &(u8arr2d.nrows(), u8arr2d.ncols())))
                .to_owned();
            debug!("Done trimming: {:?}", now.elapsed());
            let img: image::GrayImage = ImageBuffer::from_vec(
                u8arr.ncols() as u32,
                u8arr.nrows() as u32,
                u8arr.into_raw_vec(),
            )
            .unwrap();
            let jpeg_fn = out_dir.join(
                entry
                    .path()
                    .to_owned()
                    .with_extension("trimmed.jpg")
                    .file_name()
                    .unwrap(),
            );
            img.save(jpeg_fn)?;
        }
    }
    Ok(())
}
