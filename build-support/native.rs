//! Cargo owns the complete native build. CMake/Ninja own incremental C++ work.
use std::{
    collections::BTreeSet,
    env,
    fs::{self, File},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

const LLVM_REVISION: &str = "373c54d763c98487df27599247412f80d933e177";
const COVERAGE: &str = "-fsanitize-coverage=trace-pc-guard";

fn setting(name: &str) -> Option<String> {
    println!("cargo:rerun-if-env-changed={name}");
    env::var(name).ok()
}

fn run(command: &mut Command, log: &Path) {
    let output = File::create(log).expect("create native build log");
    eprintln!("Running {command:?}; log: {}", log.display());
    let status = command
        .stdout(Stdio::from(output.try_clone().unwrap()))
        .stderr(Stdio::from(output))
        .status()
        .unwrap_or_else(|error| panic!("cannot start {command:?}: {error}"));
    if !status.success() {
        let text = fs::read_to_string(log).unwrap_or_default();
        let tail: Vec<_> = text.lines().rev().take(60).collect();
        for line in tail.into_iter().rev() {
            eprintln!("{line}");
        }
        panic!(
            "native build failed ({status}); full log: {}",
            log.display()
        );
    }
}

fn output(command: &mut Command) -> String {
    let output = command.output().expect("run build tool");
    assert!(output.status.success(), "failed {command:?}");
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}

pub fn write_metadata(value: serde_json::Value) {
    let path = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("compiler-build.json");
    fs::write(path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
}

pub fn build() -> PathBuf {
    assert_eq!(
        env::var("HOST").unwrap(),
        env::var("TARGET").unwrap(),
        "cross-compilation is not supported by the native JIT build"
    );
    assert_eq!(
        env::var("CARGO_CFG_TARGET_OS").unwrap(),
        "linux",
        "the native runtime currently requires Linux"
    );
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap())
        .join("../..")
        .canonicalize()
        .unwrap();
    let cache = root.join(setting("MRACLE_BUILD_DIR").unwrap_or_else(|| "build/unified".into()));
    fs::create_dir_all(&cache).unwrap();
    let cache = cache.canonicalize().unwrap();
    // Separate Cargo profiles/target directories may enter this cache together.
    let lock = File::create(cache.join("build.lock")).unwrap();
    lock.lock().expect("lock shared native build cache");
    let cc = setting("CC").unwrap_or_else(|| "clang-21".into());
    let cxx = setting("CXX").unwrap_or_else(|| "clang++-21".into());
    let jobs = setting("MRACLE_BUILD_JOBS").unwrap_or_else(|| "2".into());
    let sanitizer = setting("MRACLE_SANITIZER").unwrap_or_else(|| "none".into());
    assert!(
        matches!(sanitizer.as_str(), "none" | "thread"),
        "MRACLE_SANITIZER must be none or thread"
    );
    assert!(
        jobs.parse::<usize>().is_ok_and(|n| n > 0),
        "MRACLE_BUILD_JOBS must be positive"
    );
    let source_override = setting("MRACLE_LLVM_SOURCE");
    let prefix_override = setting("MRACLE_LLVM_PREFIX");
    assert!(
        source_override.is_none() || prefix_override.is_none(),
        "set only one of MRACLE_LLVM_SOURCE and MRACLE_LLVM_PREFIX"
    );
    let compiler = cache.join("compiler");
    let native = cache.join(if sanitizer == "thread" {
        "native-thread"
    } else {
        "native"
    });
    let mut revision = None;
    let mut source_path = None;
    let prefix = if let Some(prefix) = prefix_override {
        println!(
            "cargo:warning=Using an external LLVM prefix; compiler instrumentation is not guaranteed by this build."
        );
        root.join(prefix)
            .canonicalize()
            .expect("MRACLE_LLVM_PREFIX must exist")
    } else {
        let source = if let Some(source) = source_override {
            let source = root
                .join(source)
                .canonicalize()
                .expect("MRACLE_LLVM_SOURCE must exist");
            println!("cargo:rerun-if-changed={}", source.join("llvm").display());
            println!("cargo:rerun-if-changed={}", source.join("mlir").display());
            println!("cargo:rerun-if-changed={}", source.join("cmake").display());
            println!(
                "cargo:rerun-if-changed={}",
                source.join("third-party").display()
            );
            source
        } else {
            let source = cache.join(format!("llvm-project-{LLVM_REVISION}"));
            println!(
                "cargo:rerun-if-changed={}",
                source.join(".git/HEAD").display()
            );
            println!(
                "cargo:rerun-if-changed={}",
                source.join(".mracle-checkout-complete").display()
            );
            if !source.join(".mracle-checkout-complete").exists() {
                println!(
                    "cargo:warning=Fetching pinned LLVM source {LLVM_REVISION}; see {}/source.log",
                    cache.display()
                );
                fs::create_dir_all(&source).unwrap();
                run(
                    Command::new("git").arg("init").arg(&source),
                    &cache.join("source-init.log"),
                );
                run(
                    Command::new("git").arg("-C").arg(&source).args([
                        "fetch",
                        "--depth=1",
                        "https://github.com/llvm/llvm-project.git",
                        LLVM_REVISION,
                    ]),
                    &cache.join("source.log"),
                );
                run(
                    Command::new("git").arg("-C").arg(&source).args([
                        "checkout",
                        "--detach",
                        "FETCH_HEAD",
                    ]),
                    &cache.join("source-checkout.log"),
                );
                fs::write(source.join(".mracle-checkout-complete"), LLVM_REVISION).unwrap();
            }
            let actual = output(
                Command::new("git")
                    .arg("-C")
                    .arg(&source)
                    .args(["rev-parse", "HEAD"]),
            );
            assert_eq!(
                actual, LLVM_REVISION,
                "managed LLVM checkout changed; use MRACLE_LLVM_SOURCE for a custom compiler"
            );
            revision = Some(actual);
            source
        };
        assert!(
            source.join("llvm/CMakeLists.txt").is_file(),
            "expected an llvm-project source tree"
        );
        println!(
            "cargo:warning=Configuring instrumented LLVM/MLIR; logs in {}",
            cache.display()
        );
        run(
            Command::new("cmake")
                .args(["-G", "Ninja", "-S"])
                .arg(source.join("llvm"))
                .arg("-B")
                .arg(&compiler)
                .args([
                    "-DCMAKE_BUILD_TYPE=Release",
                    &format!("-DCMAKE_C_COMPILER={cc}"),
                    &format!("-DCMAKE_CXX_COMPILER={cxx}"),
                    &format!("-DCMAKE_C_FLAGS={COVERAGE}"),
                    &format!("-DCMAKE_CXX_FLAGS={COVERAGE}"),
                    "-DLLVM_ENABLE_PROJECTS=mlir",
                    "-DLLVM_TARGETS_TO_BUILD=Native",
                    "-DLLVM_ENABLE_PIC=ON",
                    "-DBUILD_SHARED_LIBS=OFF",
                    "-DLLVM_BUILD_LLVM_DYLIB=OFF",
                    "-DLLVM_LINK_LLVM_DYLIB=OFF",
                    "-DLLVM_INCLUDE_TESTS=OFF",
                    "-DLLVM_INCLUDE_EXAMPLES=OFF",
                    "-DLLVM_INCLUDE_BENCHMARKS=OFF",
                    "-DMLIR_INCLUDE_TESTS=OFF",
                    "-DMLIR_ENABLE_BINDINGS_PYTHON=OFF",
                    "-DLLVM_ENABLE_ZSTD=OFF",
                    "-DLLVM_PARALLEL_LINK_JOBS=1",
                ]),
            &cache.join("compiler-configure.log"),
        );
        source_path = Some(source);
        compiler.clone()
    };
    println!("cargo:rerun-if-changed={}", root.join("native").display());
    run(
        Command::new("cmake")
            .args(["-G", "Ninja", "-S"])
            .arg(root.join("native"))
            .arg("-B")
            .arg(&native)
            .args([
                "-DCMAKE_BUILD_TYPE=Release",
                "-DBUILD_TESTING=OFF",
                &format!(
                    "-DMRACLE_ENABLE_TSAN={}",
                    if sanitizer == "thread" { "ON" } else { "OFF" }
                ),
                &format!("-DCMAKE_C_COMPILER={cc}"),
                &format!("-DCMAKE_CXX_COMPILER={cxx}"),
                &format!("-DLLVM_DIR={}/lib/cmake/llvm", prefix.display()),
                &format!("-DMLIR_DIR={}/lib/cmake/mlir", prefix.display()),
            ]),
        &cache.join("native-configure.log"),
    );
    if source_path.is_some() {
        // Build precisely the compiler archives exported by the native target's
        // transitive link graph, plus generated public headers. No LLVM tools
        // or separate native CLI are required: the compiler executes inside the worker.
        let manifest = fs::read_to_string(native.join("ffi/native-link.txt")).unwrap();
        let mut targets = BTreeSet::from(["llvm-headers".to_owned(), "mlir-headers".to_owned()]);
        for library in manifest.lines() {
            let path = Path::new(library);
            if path.starts_with(&compiler) && path.extension().is_some_and(|ext| ext == "a") {
                let name = path.file_stem().unwrap().to_str().unwrap();
                targets.insert(name.strip_prefix("lib").unwrap_or(name).to_owned());
            }
        }
        assert!(targets.len() > 2, "no LLVM/MLIR static dependencies found");
        println!(
            "cargo:warning=Building instrumented compiler (first build is substantial); progress: {}/compiler-build.log",
            cache.display()
        );
        run(
            Command::new("cmake")
                .arg("--build")
                .arg(&compiler)
                .args(["--parallel", &jobs, "--target"])
                .args(targets),
            &cache.join("compiler-build.log"),
        );
    }
    println!("cargo:warning=Building the static MLIR-MRacle engine");
    run(
        Command::new("cmake").arg("--build").arg(&native).args([
            "--target",
            "mracle_ffi",
            "--parallel",
            &jobs,
        ]),
        &cache.join("native-build.log"),
    );
    let tsan = fs::read_to_string(native.join("ffi/tsan-library.txt")).unwrap();
    let linker = fs::read_to_string(native.join("ffi/native-cc.txt")).unwrap();
    write_metadata(serde_json::json!({
        "native": true, "llvm_revision": revision, "llvm_source": source_path,
        "sanitizer": sanitizer, "execution_backends": ["jit", "native"],
        "native_linker": linker.trim(),
        "llvm_prefix": prefix, "native_build": native,
        "compiler_coverage": if source_path.is_some() {Some(COVERAGE)} else {None},
        "host_cxx": output(Command::new(&cxx).arg("--version")),
        "tsan_runtime": tsan.trim(),
    }));
    native.join("ffi")
}
