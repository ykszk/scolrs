use anyhow::Result;
use dicom::object::open_file;
use dicom::pixeldata::image;
use dicom::pixeldata::image::ImageBuffer;
use ndarray::Axis;

use devscol::{create_slice, trimming_box_with_resample, Normalize};
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
            let now = std::time::Instant::now();
            let u8arr3d = match pixel_data.bits_allocated() {
                8 => pixel_data.to_ndarray_frame::<u8>(0)?,
                16 => {
                    let arr = pixel_data.to_ndarray_frame::<i16>(0)?;
                    arr.minmax_normalize().unwrap() // arr is not empty
                }
                i => {
                    anyhow::bail!("Invalid bits allocated: {}", i)
                }
            };
            if u8arr3d.shape()[2] > 1 {
                warn!(
                    "Using first channel of {}-channel image",
                    u8arr3d.shape()[2]
                )
            }
            let u8arr2d = u8arr3d.index_axis(Axis(2), 0);

            let img = u8arr2d.mapv(|e| e as i16);

            let resample_step = img.ncols().max(img.nrows()) / 1000 + 1;
            debug!("resample_step: {:?}", resample_step);
            let bbox = trimming_box_with_resample(img.view(), resample_step)?;
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
