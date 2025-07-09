use anyhow::Result;
use ndarray::{s, Array3, CowArray, NewAxis};
use ort::{Environment, GraphOptimizationLevel, SessionBuilder, Value};

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    // get arguments: first is the model path, second is the image path, and third is the output path
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "Usage: {} <model_path> <input_image_path> <output_image_path>",
            args[0]
        );
        std::process::exit(1);
    }
    let model_path = &args[1];
    let image_path = &args[2];
    let output_path = &args[3];
    log::info!("Model path: {}", model_path);
    log::info!("Image path: {}", image_path);
    log::info!("Output path: {}", output_path);

    // Load the model
    let environment = Environment::builder().build()?.into_arc();
    let session = SessionBuilder::new(&environment)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .with_model_from_file(model_path)?;
    log::info!("Model loaded successfully");
    // Load the image
    let image = image::open(image_path).expect("Failed to open image");
    log::info!("Image loaded successfully");
    let image = image.thumbnail(10000, 768);
    // Convert the image to a tensor
    let image = Array3::from_shape_vec(
        (image.height() as usize, image.width() as usize, 1),
        image.to_luma8().into_raw(),
    )?;
    // convert to float32
    let image = image.mapv(|x| x as f32 / 255.0);
    log::info!("Image converted to tensor shape: {:?}", image.shape());
    // pad image width to be divisible by 256
    let padded_width = ((image.shape()[1] + 255) / 256) * 256;
    let mut padded_image: Array3<f32> =
        Array3::zeros((image.shape()[0], padded_width, image.shape()[2]));
    padded_image
        .slice_mut(s![.., ..image.shape()[1], ..])
        .assign(&image);
    let image = padded_image;
    // to channel first
    let image = image.permuted_axes([2, 0, 1]);

    let input = CowArray::from(image.slice(s![NewAxis, .., .., ..]).into_dyn());
    log::info!(
        "Image converted to tensor successfully with shape: {:?}",
        input.shape()
    );
    // Run the model
    let inputs = vec![Value::from_array(session.allocator(), &input)?];
    let mut outputs: Vec<Value> = session.run(inputs)?;

    // convert to ndarray
    let output = outputs.pop().unwrap();
    let output_array = output.try_extract::<f32>()?.view().to_owned();
    log::info!("Output tensor shape: {:?}", output_array.shape());
    let rgb = output_array.slice(s![0, ..3, .., ..]).to_owned();
    // apply sigmoid
    let rgb = rgb.mapv(|x| (1.0 / (1.0 + (-x).exp())));
    log::info!("First channel shape: {:?}", rgb.shape());
    // save the first channel as an image
    let first_channel_image =
        image::ImageBuffer::from_fn(rgb.shape()[2] as u32, rgb.shape()[1] as u32, |x, y| {
            let r = rgb[[0, y as usize, x as usize]];
            let r = (r * 255.0) as u8;
            let g = rgb[[1, y as usize, x as usize]];
            let g = (g * 255.0) as u8;
            let b = rgb[[2, y as usize, x as usize]];
            let b = (b * 255.0) as u8;

            image::Rgb([r, g, b])
        });
    first_channel_image
        .save(output_path)
        .expect("Failed to save output image");
    Ok(())
}
