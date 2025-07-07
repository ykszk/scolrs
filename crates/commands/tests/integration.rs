use anyhow::Result;
use assert_cmd::Command;
use labelme_rs::{LabelMeData, LabelMeDataLine};
use std::{env, io::Write, path::PathBuf};

#[test]
fn test_redirect() -> Result<()> {
    let mut cmd = Command::cargo_bin("scolrs").unwrap();
    let data_dir = PathBuf::from("../../tests/data/");
    let output_dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    cmd.arg("svg")
        .arg(data_dir.join("case1/frontal.json"))
        .arg("-")
        .arg("--labelme")
        .arg("coronal")
        .assert()
        .success();

    let cmd_output = cmd.output()?;

    let mut cmd = Command::cargo_bin("scolrs").unwrap();
    cmd.arg("html")
        .arg("-")
        .arg(output_dir.join("case1_frontal.html"))
        .write_stdin(cmd_output.stdout)
        .assert()
        .success();

    // create ndjson
    let ndjson = output_dir.join("case5.ndjson");
    let mut json_files = Vec::new();
    for entry in glob::glob(data_dir.join("case5/frontal*.json").to_str().unwrap()).unwrap() {
        let entry = entry.unwrap();
        if entry.is_file() {
            json_files.push(entry);
        }
    }
    json_files.sort();
    let mut ndjson_file = std::fs::File::create(&ndjson)?;
    for json_file in json_files.into_iter() {
        let data = LabelMeData::try_from(json_file.as_path())?;
        let data = data.to_absolute_path(json_file.canonicalize()?.parent().unwrap());
        let data_line = LabelMeDataLine {
            content: data,
            filename: json_file.to_string_lossy().into_owned(),
        };
        ndjson_file.write_all(serde_json::to_string(&data_line)?.as_bytes())?;
        ndjson_file.write_all(b"\n")?;
    }

    ndjson_file.flush()?;
    ndjson_file.sync_all()?;

    // convert ndjson
    let mut cmd: Command = Command::cargo_bin("scolrs").unwrap();
    let ndjson_native = output_dir.join("case5_frontal_native.ndjson");
    cmd.arg("conv")
        .arg(&ndjson)
        .arg("--ndjson")
        .arg("--from")
        .arg("Labelme")
        .arg("--to")
        .arg("ScoliosisCoronal")
        .assert()
        .success();

    let mut ndjson_file = std::fs::File::create(&ndjson_native)?;
    ndjson_file.write_all(&cmd.output()?.stdout)?;

    ndjson_file.sync_all()?;

    // test  combination of svg-ndjson and catalog
    let mut cmd = Command::cargo_bin("scolrs").unwrap();
    cmd.arg("svg-ndjson")
        .arg(&ndjson_native)
        .arg("-")
        .arg("coronal")
        .assert()
        .success();

    let cmd_output = cmd.output()?;

    let mut cmd = Command::cargo_bin("scolrs").unwrap();
    cmd.arg("catalog")
        .arg("-")
        .arg(output_dir.join("catalog_case5_frontal.html"))
        .write_stdin(cmd_output.stdout)
        .assert()
        .success();

    Ok(())
}
