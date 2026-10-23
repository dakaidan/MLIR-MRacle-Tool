use std::{collections::BTreeSet, path::Path};

#[path = "../../build-support/native.rs"]
mod native;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(mracle_tsan)");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../build-support");
    if std::env::var_os("CARGO_FEATURE_NATIVE").is_none() {
        native::write_metadata(serde_json::json!({"native": false}));
        return;
    }
    let dir = native::build();
    let manifest = dir.join("native-link.txt");
    println!("cargo:rerun-if-changed={}", manifest.display());
    let libraries = std::fs::read_to_string(&manifest)
        .expect("build the native static libraries first; missing native-link.txt");
    let mut runtime_paths = BTreeSet::new();
    worker_arg("-Wl,--start-group");
    for library in libraries.lines().filter(|line| !line.is_empty()) {
        let path = Path::new(library);
        if path.is_absolute() {
            println!("cargo:rerun-if-changed={library}");
            assert!(path.is_file(), "native library missing: {library}");
            worker_arg(library);
            if library.contains(".so") {
                runtime_paths.insert(path.parent().unwrap().to_owned());
            }
        } else if library.starts_with('-') {
            worker_arg(library);
        } else {
            worker_arg(&format!("-l{library}"));
        }
    }
    worker_arg("-Wl,--end-group");

    let runtime_file = dir.join("tsan-library.txt");
    println!("cargo:rerun-if-changed={}", runtime_file.display());
    let runtime = std::fs::read_to_string(runtime_file).expect("missing tsan-library.txt");
    if !runtime.trim().is_empty() {
        println!("cargo:rustc-cfg=mracle_tsan");
        let runtime = Path::new(runtime.trim());
        assert!(runtime.is_file(), "missing ThreadSanitizer runtime");
        let runtime_dir = runtime.parent().unwrap();
        let runtime_name = runtime
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .strip_prefix("lib")
            .unwrap()
            .strip_suffix(".so")
            .unwrap();
        // rustc-link-lib puts TSan before Rust's libc dependencies. Appending it
        // as a late link argument lets libc's pthread_create bypass its interceptor.
        println!("cargo:rustc-link-search=native={}", runtime_dir.display());
        println!("cargo:rustc-link-lib=dylib={runtime_name}");
        runtime_paths.insert(runtime_dir.to_owned());
    }
    for path in runtime_paths {
        worker_arg(&format!("-Wl,-rpath,{}", path.display()));
    }
    // LLJIT and TSan resolve runtime symbols through the current process;
    // static native symbols must now be visible from the Rust executable.
    worker_arg("-Wl,--export-dynamic");
}

fn worker_arg(argument: &str) {
    println!("cargo:rustc-link-arg-bin=mracle-worker={argument}");
}
