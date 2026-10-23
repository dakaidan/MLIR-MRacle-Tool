use serde_json::Value;
use std::{
    fs,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("mracle-campaign-{}-{nonce}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn json(&self, path: &str) -> Value {
        serde_json::from_slice(&fs::read(self.0.join(path)).unwrap()).unwrap()
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn campaign(dir: &Directory, seed: &str, iterations: u64) {
    fs::write(dir.0.join("seed.mlir"), seed).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_mracle-fuzz"))
        .args([
            "--allow-uninstrumented",
            "--input",
            dir.0.join("seed.mlir").to_str().unwrap(),
            "--output",
            dir.0.join("out").to_str().unwrap(),
            "--worker",
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/worker.py"),
            "--iterations",
            &iterations.to_string(),
            "--max-trials",
            "20",
            "--block-size",
            "8",
            "--epsilon",
            "0.6",
        ])
        .output()
        .unwrap();
    assert_eq!(
        result.status.success(),
        !seed.contains("warn-fixture"),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

#[test]
fn sibling_worker_is_found_after_relocation_from_another_directory() {
    let dir = Directory::new();
    let binaries = dir.0.join("bin");
    fs::create_dir(&binaries).unwrap();
    fs::copy(
        env!("CARGO_BIN_EXE_mracle-fuzz"),
        binaries.join("mracle-fuzz"),
    )
    .unwrap();
    fs::copy(
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/worker.py"),
        binaries.join("mracle-worker"),
    )
    .unwrap();
    fs::write(dir.0.join("seed.mlir"), "module {}").unwrap();
    let result = Command::new(binaries.join("mracle-fuzz"))
        .current_dir(&dir.0)
        .args([
            "--allow-uninstrumented",
            "--input",
            "seed.mlir",
            "--output",
            "out",
            "--iterations",
            "1",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        dir.json("out/config.json")["worker"],
        binaries.join("mracle-worker").to_str().unwrap()
    );
    assert!(dir.0.join("out/driver-build.json").is_file());
}

#[test]
fn execution_options_survive_campaign_and_replay() {
    let dir = Directory::new();
    fs::write(dir.0.join("seed.mlir"), "module {}").unwrap();
    let fixture = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/worker.py");
    let status = Command::new(env!("CARGO_BIN_EXE_mracle-fuzz"))
        .args([
            "--input",
            dir.0.join("seed.mlir").to_str().unwrap(),
            "--output",
            dir.0.join("out").to_str().unwrap(),
            "--worker",
            fixture,
            "--allow-uninstrumented",
            "--iterations",
            "1",
            "--backend",
            "native",
            "--tsan-triage",
        ])
        .output()
        .unwrap();
    assert!(
        status.status.success(),
        "{}",
        String::from_utf8_lossy(&status.stderr)
    );
    let request = dir.0.join("out/runs/00000000/request.json");
    let result = Command::new(env!("CARGO_BIN_EXE_mracle-replay"))
        .arg(&request)
        .arg("--output")
        .arg(dir.0.join("replay"))
        .args(["--worker", fixture])
        .args([
            "--reps",
            "3",
            "--replay-reps",
            "4",
            "--max-runs",
            "5",
            "--max-apply",
            "2",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    for path in ["out/runs/00000000/request.json", "replay/request.json"] {
        assert_eq!(dir.json(path)["backend"], "native");
        assert_eq!(dir.json(path)["tsan_triage"], true);
    }
    assert_eq!(dir.json("replay/request.json")["reps"], 3);
    assert_eq!(dir.json("replay/request.json")["replay_reps"], 4);
    assert_eq!(dir.json("replay/request.json")["max_runs"], 5);
    assert_eq!(dir.json("replay/request.json")["max_apply"], 2);
    let mut old = dir.json("replay/request.json");
    old.as_object_mut().unwrap().remove("backend");
    old.as_object_mut().unwrap().remove("tsan_triage");
    let old: mlir_mracle::protocol::Request = serde_json::from_value(old).unwrap();
    assert!(matches!(
        old.backend,
        mlir_mracle::protocol::Backend::Jit
    ));
    assert!(!old.tsan_triage);
}

#[test]
fn novel_children_are_queued_and_source_is_held_until_validation() {
    let dir = Directory::new();
    campaign(&dir, "module {}", 100);
    let summary = dir.json("out/summary.json");
    assert_eq!(
        dir.json("out/config.json")["confidence_features"],
        "mr-only"
    );
    assert_eq!(dir.json("out/config.json")["coverage_artifacts"], "summary");
    assert_eq!(summary["corpus_size"], 2);
    assert_eq!(summary["saturated_sources"], 2);
    assert_eq!(summary["attempts"], 32);
    assert_eq!(dir.json("out/queue/000001.json")["parent"], 0);
    assert_eq!(dir.json("out/sources/000000.json")["state"], "saturated");
    let event_text = fs::read_to_string(dir.0.join("out/events.jsonl")).unwrap();
    let events: Vec<Value> = event_text
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    assert!(events[..16].iter().all(|e| e["parent"] == 0));
    assert!(events[16..].iter().all(|e| e["parent"] == 1));
    assert_eq!(events[0]["report"]["applied"][0]["name"], "fixture");
    assert_eq!(events[0]["confidence_target_samples"], 8);
    assert_eq!(events[0]["confidence_novel_features"], 1);
    let coverage = dir.json("out/runs/00000000/coverage.json");
    assert_eq!(coverage["edge_count"], 0);
    assert!(coverage.get("edges").is_none());
    let result = Command::new(env!("CARGO_BIN_EXE_mracle-replay"))
        .arg(dir.0.join("out/runs/00000000/request.json"))
        .arg("--output")
        .arg(dir.0.join("replay"))
        .args([
            "--worker",
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/worker.py"),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(dir.json("replay/response.json")["status"], "OK");
}

#[test]
fn budget_and_warn_are_inconclusive_and_existing_output_is_preserved() {
    let dir = Directory::new();
    campaign(&dir, "module {}", 1);
    assert_eq!(dir.json("out/sources/000000.json")["state"], "inconclusive");
    assert_eq!(
        dir.json("out/summary.json")["stop_reason"],
        "campaign-budget"
    );
    let warn = Directory::new();
    campaign(&warn, "// warn-fixture\nmodule {}", 20);
    assert_eq!(warn.json("out/summary.json")["saturated_sources"], 0);
    assert_eq!(warn.json("out/summary.json")["findings"], 20);
    assert_eq!(warn.json("out/sources/000000.json")["tainted"], true);
    let saved = fs::read(warn.0.join("out/config.json")).unwrap();
    campaign(&warn, "// warn-fixture\nmodule {}", 1);
    assert_eq!(fs::read(warn.0.join("out/config.json")).unwrap(), saved);
}

#[test]
fn new_compiler_edges_admit_children_without_new_semantic_features() {
    let dir = Directory::new();
    campaign(&dir, "// coverage-fixture\nmodule {}", 3);
    let summary = dir.json("out/summary.json");
    assert_eq!(summary["corpus_size"], 3);
    assert_eq!(summary["compiler_edges"], 2);
    assert_eq!(dir.json("out/queue/000002.json")["new_features"], 1);
}

#[test]
fn coverage_artifact_retention_is_optional() {
    let dir = Directory::new();
    fs::write(dir.0.join("seed.mlir"), "// coverage-fixture\nmodule {}").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_mracle-fuzz"))
        .args([
            "--allow-uninstrumented",
            "--input",
            dir.0.join("seed.mlir").to_str().unwrap(),
            "--output",
            dir.0.join("out").to_str().unwrap(),
            "--worker",
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/worker.py"),
            "--iterations",
            "3",
            "--coverage-artifacts",
            "none",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(dir.json("out/summary.json")["compiler_edges"], 2);
    assert_eq!(dir.json("out/summary.json")["corpus_size"], 3);
    assert!(!dir.0.join("out/runs/00000000/coverage.json").exists());
    let event_text = fs::read_to_string(dir.0.join("out/events.jsonl")).unwrap();
    let first_event: Value = serde_json::from_str(event_text.lines().next().unwrap()).unwrap();
    assert_eq!(first_event["compiler_edges"], 1);
}

#[test]
fn uninstrumented_compiler_requires_explicit_opt_out() {
    let dir = Directory::new();
    fs::write(dir.0.join("seed.mlir"), "module {}").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_mracle-fuzz"))
        .arg("--input")
        .arg(dir.0.join("seed.mlir"))
        .arg("--output")
        .arg(dir.0.join("out"))
        .args([
            "--worker",
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/worker.py"),
            "--iterations",
            "1",
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("no compiler coverage observed"));
}

#[test]
fn time_budget_finishes_the_active_trial_and_saves_a_summary() {
    let dir = Directory::new();
    fs::write(dir.0.join("seed.mlir"), "module {} // slow-fixture").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_mracle-fuzz"))
        .args(["--allow-uninstrumented", "--input"])
        .arg(dir.0.join("seed.mlir"))
        .arg("--output")
        .arg(dir.0.join("out"))
        .args([
            "--worker",
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/worker.py"),
            "--duration-secs",
            "1",
            "--max-corpus",
            "1",
            "--iterations",
            "100",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let summary = dir.json("out/summary.json");
    assert_eq!(summary["stop_reason"], "time-budget");
    let attempts = summary["attempts"].as_u64().unwrap();
    assert!((1..=2).contains(&attempts));
    assert_eq!(summary["usable_trials"], attempts);
    assert_eq!(summary["findings"], 0);
    assert!(summary["elapsed_seconds"].as_f64().unwrap() >= 1.0);
}

#[test]
fn external_stop_request_preserves_a_summary_without_starting_a_worker() {
    let dir = Directory::new();
    fs::write(dir.0.join("seed.mlir"), "module {}").unwrap();
    fs::write(dir.0.join("stop"), "stop").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_mracle-fuzz"))
        .args(["--allow-uninstrumented", "--input"])
        .arg(dir.0.join("seed.mlir"))
        .arg("--output")
        .arg(dir.0.join("out"))
        .args([
            "--worker",
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/worker.py"),
        ])
        .arg("--stop-file")
        .arg(dir.0.join("stop"))
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert_eq!(dir.json("out/summary.json")["stop_reason"], "stop-request");
    assert_eq!(dir.json("out/summary.json")["attempts"], 0);
    assert_eq!(fs::read_dir(dir.0.join("out/runs")).unwrap().count(), 0);
}
