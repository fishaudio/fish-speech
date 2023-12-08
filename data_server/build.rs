fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::compile_protos("../fish_speech/datasets/protos/text-data.proto")?;
    Ok(())
}
