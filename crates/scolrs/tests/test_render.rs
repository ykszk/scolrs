use anyhow::Result;
use std::path::PathBuf;
use std::process::Command;

fn test_data_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data")
}

fn tmp_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
}

#[ignore]
#[test]
fn test_render_cases() -> Result<()> {
    // case 1
    let bin_path = PathBuf::from(env!("CARGO_BIN_EXE_scolrs"));
    let data_dir = test_data_directory();
    let tmp_dir = tmp_directory();

    // save curve set from frontal image
    let output = Command::new(&bin_path)
        .arg("curve")
        .arg(data_dir.join("case1/frontal.json"))
        .output()?;
    let curve_set_path = tmp_dir.join("case1_frontal.json");
    std::fs::write(&curve_set_path, output.stdout)?;
    let colors = data_dir.join("colors.yaml");
    let line_colors = data_dir.join("line_colors.csv");
    let common_args: Vec<&std::ffi::OsStr> = vec![
        "render".as_ref(),
        "--label-colors".as_ref(),
        colors.as_os_str(),
        "--line-colors".as_ref(),
        line_colors.as_os_str(),
        "--resize".as_ref(),
        "1024x1024".as_ref(),
    ];

    // render
    for stem in ["frontal", "left_lateral_bend", "right_lateral_bend"] {
        println!("stem: {:?}", stem);
        let output = Command::new(&bin_path)
            .args(
                [
                    common_args.clone(),
                    vec![
                        "--curve-set".as_ref(),
                        curve_set_path.as_os_str(),
                        data_dir.join(format!("case1/{stem}.json")).as_os_str(),
                        tmp_dir.join(format!("case1_{stem}.svg")).as_os_str(),
                    ],
                ]
                .concat(),
            )
            .output()
            .unwrap();
        println!("stdout:{}", String::from_utf8(output.stdout)?);
        println!("stderr:{}", String::from_utf8(output.stderr)?);
        assert!(output.status.success());
    }
    let stem = "lateral";
    let output = Command::new(&bin_path)
        .args(
            [
                common_args.clone(),
                vec![
                    "--curve-set".as_ref(),
                    curve_set_path.as_os_str(),
                    data_dir.join(format!("case1/{stem}.json")).as_os_str(),
                    tmp_dir.join(format!("case1_{stem}.svg")).as_os_str(),
                    "--direction".as_ref(),
                    "lateral".as_ref(),
                ],
            ]
            .concat(),
        )
        .output()
        .unwrap();
    println!("stdout:{}", String::from_utf8(output.stdout)?);
    println!("stderr:{}", String::from_utf8(output.stderr)?);
    assert!(output.status.success());

    for case in ["case2", "case3", "case4"] {
        for stem in ["frontal", "lateral"] {
            let output = if stem == "frontal" {
                Command::new(&bin_path)
                    .args(
                        [
                            common_args.clone(),
                            vec![
                                data_dir.join(format!("{case}/{stem}.json")).as_os_str(),
                                tmp_dir.join(format!("{case}_{stem}.svg")).as_os_str(),
                            ],
                        ]
                        .concat(),
                    )
                    .output()
            } else {
                Command::new(&bin_path)
                    .args(
                        [
                            common_args.clone(),
                            vec![
                                data_dir.join(format!("{case}/{stem}.json")).as_os_str(),
                                tmp_dir.join(format!("{case}_{stem}.svg")).as_os_str(),
                                "--direction".as_ref(),
                                "lateral".as_ref(),
                            ],
                        ]
                        .concat(),
                    )
                    .output()
            }
            .unwrap();
            println!("stdout:{}", String::from_utf8(output.stdout)?);
            println!("stderr:{}", String::from_utf8(output.stderr)?);
            assert!(output.status.success());
        }
    }

    Ok(())
}
