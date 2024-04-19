fn main() -> Result<(), Box<dyn std::error::Error>> {
    const TEXT_DATA_PROTO: &str = "../fish_speech/datasets/protos/text-data.proto";

    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed={}", TEXT_DATA_PROTO);

    tonic_build::compile_protos(TEXT_DATA_PROTO)?;
    Ok(())
}
