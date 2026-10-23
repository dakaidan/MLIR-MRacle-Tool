use mlir_mracle::protocol::Request;
mod ffi;

const BUILD_INFO: &str = include_str!(concat!(env!("OUT_DIR"), "/compiler-build.json"));
fn main() {
    if let Err(error) = run() {
        eprintln!("mracle-worker: {error}");
        std::process::exit(2);
    }
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args_os()
        .nth(1)
        .ok_or("usage: mracle-worker REQUEST.json | --build-info")?;
    if path == "--build-info" {
        println!("{BUILD_INFO}");
        return Ok(());
    }
    let request: Request = serde_json::from_slice(&std::fs::read(path)?)?;
    // Record worker provenance even when a different build was selected via --worker.
    std::fs::write(request.artifacts.join("compiler-build.json"), BUILD_INFO)?;
    let report = ffi::run(&request)?;
    // An atomic response distinguishes complete reports from crashes during writing.
    std::fs::write(request.artifacts.join("response.tmp"), report)?;
    std::fs::rename(
        request.artifacts.join("response.tmp"),
        request.artifacts.join("response.json"),
    )?;
    Ok(())
}
