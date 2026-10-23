pub mod confidence;
pub mod coverage;
pub mod protocol;
pub mod supervisor;

/// Keep debug/release and relocated binary pairs independent of the working directory.
pub fn default_worker() -> std::path::PathBuf {
    std::env::current_exe()
        .expect("resolve current executable")
        .with_file_name("mracle-worker")
}
