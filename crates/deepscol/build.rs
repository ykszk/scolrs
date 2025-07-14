use anyhow::Result;
use std::path::PathBuf;

fn main() -> Result<()> {
    // initialize env logger
    env_logger::init();

    let url = "https://drive.google.com/uc?export=download&id=1Ephh5ySR7e9VVVg_rAHKwKOzEcmScUlg";
    let file_name = "spine_mobileone_s1.onnx";
    let output_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models");
    std::fs::create_dir_all(&output_dir)?;

    let output_file = output_dir.join(file_name);
    if output_file.exists() {
        // println!("Model already exists at {}", output_file.display());
        return Ok(());
    } else {
        log::info!("Downloading model from {}", url);
    }

    // download file using reqwest
    let client = reqwest::blocking::Client::new();
    let response = client.get(url).send()?;
    if response.status().is_success() {
        let mut file = std::fs::File::create(output_dir.join(file_name))?;
        let content = response.bytes()?;
        std::io::copy(&mut content.as_ref(), &mut file)?;
        println!(
            "Model downloaded successfully to {}",
            output_dir.join(file_name).display()
        );
    } else {
        return Err(anyhow::anyhow!(
            "Failed to download model: {}",
            response.status()
        ));
    }

    Ok(())
}
