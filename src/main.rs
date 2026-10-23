use clap::Parser;
use libafl::{
    corpus::{Corpus, InMemoryCorpus, Testcase},
    inputs::BytesInput,
};
use libafl_bolts::rands::{Rand, StdRand};
use mlir_mracle::{
    confidence::Confidence,
    coverage,
    protocol::{
        Comparison, DEFAULT_BOOTSTRAP_REPLICATES, DEFAULT_CAMPAIGNS, DEFAULT_CONFIDENCE_LEVEL,
        DEFAULT_EXECUTIONS_PER_CAMPAIGN, DEFAULT_MAX_CAMPAIGNS, DEFAULT_MIN_CHANGE,
        DEFAULT_TEAM_SIZE, DEFAULT_TVD_THRESHOLD, Report, Request,
    },
    supervisor::{self, Outcome},
};
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    collections::{BTreeSet, VecDeque},
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Budgets of the outcome method, which the campaign methods replace with
/// per-campaign budgets.
const DEFAULT_REPS: i32 = 100;
const DEFAULT_REPLAY_REPS: i32 = 100;
const DEFAULT_MAX_RUNS: i32 = 10000;

/// Feedback-driven metamorphic testing of concurrent MLIR programs.
#[derive(Debug, Parser, Serialize)]
#[command(version, about)]
struct Options {
    /// Explicitly permit a smoke test with a compiler built without coverage.
    #[arg(long)]
    allow_uninstrumented: bool,
    /// Seed MLIR file or directory (recursively loaded in sorted order).
    #[arg(short, long)]
    input: PathBuf,
    /// New campaign directory. Existing directories are never overwritten.
    #[arg(short, long)]
    output: PathBuf,
    /// Native worker executable; defaults to the sibling of this binary.
    #[arg(long, default_value_os_t = mlir_mracle::default_worker())]
    worker: PathBuf,
    /// Execute via LLJIT or emitted and linked native executables.
    #[arg(long, value_enum, default_value = "jit")]
    backend: mlir_mracle::protocol::Backend,
    /// Enable final race-triage probes (requires MRACLE_SANITIZER=thread).
    #[arg(long)]
    tsan_triage: bool,
    /// Comparison method the engine runs; tolerance/adaptive measure outcome
    /// distributions instead of sweeping the agitation configurations.
    #[arg(long, value_enum, default_value = "outcome")]
    comparison: Comparison,
    /// Campaigns collected before the first verdict (campaign methods).
    #[arg(long, default_value_t = DEFAULT_CAMPAIGNS)]
    campaigns: i32,
    /// Campaign cap while the verdict stays inconclusive (campaign methods).
    #[arg(long, default_value_t = DEFAULT_MAX_CAMPAIGNS)]
    max_campaigns: i32,
    /// Executions per campaign and variant (campaign methods).
    #[arg(long, default_value_t = DEFAULT_EXECUTIONS_PER_CAMPAIGN)]
    executions_per_campaign: i32,
    /// OpenMP team size of every campaign execution (campaign methods).
    #[arg(long, default_value_t = DEFAULT_TEAM_SIZE)]
    team_size: i32,
    /// Resamples building the uncertainty estimate (campaign methods).
    #[arg(long, default_value_t = DEFAULT_BOOTSTRAP_REPLICATES)]
    bootstrap_replicates: i32,
    /// Practical tolerance of the tolerance method, in total variation distance.
    #[arg(long, default_value_t = DEFAULT_TVD_THRESHOLD)]
    tvd_threshold: f64,
    /// Smallest distributional change the adaptive method reports.
    #[arg(long, default_value_t = DEFAULT_MIN_CHANGE)]
    min_change: f64,
    /// Confidence of the campaign interval estimate.
    #[arg(long, default_value_t = DEFAULT_CONFIDENCE_LEVEL)]
    confidence_level: f64,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Total mutation/execution attempts, including failed attempts.
    #[arg(long, default_value_t = 10000)]
    iterations: u64,
    /// Stop starting trials after this many seconds; finish the active trial normally.
    #[arg(long)]
    duration_secs: Option<u64>,
    /// Stop after the active trial when this file exists (for external campaign control).
    #[arg(long)]
    stop_file: Option<PathBuf>,
    /// Hard attempt budget while retaining the same source program.
    #[arg(long, default_value_t = 1024)]
    max_trials: u64,
    /// Minimum successful trials in an adaptive novelty validation block.
    #[arg(long, default_value_t = 128)]
    block_size: u64,
    /// Upper novelty probability required before retiring a source.
    #[arg(long, default_value_t = 0.05)]
    epsilon: f64,
    /// Per-source error budget, spent across successive validation blocks.
    #[arg(long, default_value_t = 0.05)]
    alpha: f64,
    /// Feature family used for per-source saturation confidence.
    #[arg(long, value_enum, default_value = "mr-only")]
    confidence_features: ConfidenceFeatures,
    /// Per-trial compiler coverage artifact retention.
    #[arg(long, value_enum, default_value = "summary")]
    coverage_artifacts: CoverageArtifacts,
    /// Wall-clock limit for one complete native pipeline invocation.
    #[arg(long, default_value_t = 60000)]
    timeout_ms: u64,
    #[arg(long, default_value_t = DEFAULT_REPS)]
    reps: i32,
    #[arg(long, default_value_t = DEFAULT_REPLAY_REPS)]
    replay_reps: i32,
    #[arg(long, default_value_t = DEFAULT_MAX_RUNS)]
    max_runs: i32,
    #[arg(long, default_value_t = 1)]
    max_apply: i32,
    /// Comma-separated transform names; empty means the full registry.
    #[arg(long, default_value = "")]
    transform: String,
    /// Memory-model gate used by the transform registry.
    #[arg(long, default_value = "")]
    model: String,
    #[arg(long, default_value_t = 10000)]
    max_corpus: usize,
    #[arg(long, default_value_t = 1048576)]
    max_input_bytes: usize,
}

#[derive(Debug, Clone, Copy, Serialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum ConfidenceFeatures {
    /// Only transform sequence and expected MR relation.
    MrOnly,
    /// MR features plus MLIR structure and observed outcome values.
    Semantic,
    /// Semantic features plus compiler coverage edges.
    All,
}

#[derive(Debug, Clone, Copy, Serialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum CoverageArtifacts {
    /// Write guards and the exact edge list to coverage.json.
    Full,
    /// Write guards and edge count, but omit the bulky exact edge list.
    Summary,
    /// Do not write per-trial coverage.json.
    None,
}

impl ConfidenceFeatures {
    fn select(&self, report: &Report, compiler_edges: &BTreeSet<String>) -> BTreeSet<String> {
        match self {
            Self::MrOnly => report.mr_features(),
            Self::Semantic => report.features(),
            Self::All => {
                let mut features = report.features();
                features.extend(compiler_edges.iter().cloned());
                features
            }
        }
    }
}

impl CoverageArtifacts {
    fn write(&self, dir: &Path, compiler_coverage: &coverage::Coverage) -> Result<()> {
        match self {
            Self::Full => write_json(
                dir.join("coverage.json"),
                &json!({
                    "guards": compiler_coverage.guards,
                    "edges": compiler_coverage.edges,
                    "edge_count": compiler_coverage.edges.len(),
                }),
            ),
            Self::Summary => write_json(
                dir.join("coverage.json"),
                &json!({
                    "guards": compiler_coverage.guards,
                    "edge_count": compiler_coverage.edges.len(),
                }),
            ),
            Self::None => Ok(()),
        }
    }
}

fn main() {
    if let Err(error) = run(Options::parse()) {
        eprintln!("mracle-fuzz: {error}");
        std::process::exit(1);
    }
}

fn write_json(path: impl AsRef<Path>, value: &impl Serialize) -> Result<()> {
    let path = path.as_ref();
    let temporary = path.with_extension("tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

/// Flags of the unused method are rejected rather than silently ignored: the
/// two methods disagree about what a repetition and a budget mean.
fn validate_comparison(options: &Options) -> Result<()> {
    let campaign_defaults = options.campaigns == DEFAULT_CAMPAIGNS
        && options.max_campaigns == DEFAULT_MAX_CAMPAIGNS
        && options.executions_per_campaign == DEFAULT_EXECUTIONS_PER_CAMPAIGN
        && options.team_size == DEFAULT_TEAM_SIZE
        && options.bootstrap_replicates == DEFAULT_BOOTSTRAP_REPLICATES
        && options.tvd_threshold == DEFAULT_TVD_THRESHOLD
        && options.min_change == DEFAULT_MIN_CHANGE
        && options.confidence_level == DEFAULT_CONFIDENCE_LEVEL;
    if options.comparison == Comparison::Agitation {
        return if campaign_defaults {
            Ok(())
        } else {
            Err("campaign flags belong to --comparison tolerance or adaptive".into())
        };
    }
    if options.reps != DEFAULT_REPS
        || options.replay_reps != DEFAULT_REPLAY_REPS
        || options.max_runs != DEFAULT_MAX_RUNS
    {
        return Err(
            "--reps, --replay-reps and --max-runs belong to the outcome method; \
             use --executions-per-campaign, --campaigns and --max-campaigns"
                .into(),
        );
    }
    if options.tsan_triage {
        return Err("--tsan-triage belongs to the outcome method".into());
    }
    if options.campaigns < 2 {
        return Err("--campaigns must be at least 2".into());
    }
    if options.max_campaigns < options.campaigns {
        return Err("--max-campaigns must be at least --campaigns".into());
    }
    if options.executions_per_campaign <= 0
        || options.team_size <= 0
        || options.bootstrap_replicates <= 0
    {
        return Err(
            "campaign executions, team size and bootstrap replicates must be positive".into(),
        );
    }
    let thresholds_valid = 0.0 < options.tvd_threshold
        && options.tvd_threshold <= 1.0
        && 0.0 < options.min_change
        && options.min_change <= 1.0
        && 0.0 < options.confidence_level
        && options.confidence_level < 1.0;
    if !thresholds_valid {
        return Err(
            "--tvd-threshold and --min-change must be in (0, 1]; --confidence-level must lie \
             strictly between 0 and 1"
                .into(),
        );
    }
    Ok(())
}

fn seed_files(path: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    if path.is_file() {
        if path.extension().is_some_and(|ext| ext == "mlir") {
            files.push(path.to_owned());
        }
    } else if path.is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            if !entry.file_type()?.is_symlink() {
                seed_files(&entry.path(), files)?;
            }
        }
    } else {
        return Err(format!("input does not exist: {}", path.display()).into());
    }
    Ok(())
}

fn report_event(report: &Report) -> Value {
    json!({
        "status": &report.status,
        "transform_applied": report.transform_applied,
        "applied": &report.applied,
        "relation": &report.relation,
        "source_runs": report.source_runs,
        "transformed_runs": report.transformed_runs,
        "source_outcomes": report.source_outcomes.len(),
        "transformed_outcomes": report.transformed_outcomes.len(),
    })
}

fn run(mut options: Options) -> Result<()> {
    if options.iterations == 0
        || options.duration_secs == Some(0)
        || options.max_trials == 0
        || options.block_size == 0
        || options.timeout_ms == 0
        || options.max_corpus == 0
        || options.max_input_bytes == 0
        || options.reps <= 0
        || options.replay_reps <= 0
        || options.max_runs <= 0
        || options.max_apply <= 0
        || !(0.0 < options.alpha && options.alpha < 1.0)
        || !(0.0 < options.epsilon && options.epsilon < 1.0)
    {
        return Err(
            "budgets must be positive; alpha and epsilon must lie strictly between 0 and 1".into(),
        );
    }
    options.input = options.input.canonicalize()?;
    options.worker = options.worker.canonicalize()?;
    validate_comparison(&options)?;
    let mut files = Vec::new();
    seed_files(&options.input, &mut files)?;
    files.sort();
    if files.is_empty() {
        return Err("no .mlir seeds found".into());
    }
    if files.len() > options.max_corpus {
        return Err("seed count exceeds --max-corpus".into());
    }
    for path in &files {
        if fs::metadata(path)?.len() > options.max_input_bytes as u64 {
            return Err(format!("seed exceeds --max-input-bytes: {}", path.display()).into());
        }
    }
    fs::create_dir(&options.output)?;
    options.output = options.output.canonicalize()?;
    for name in ["queue", "runs", "sources", "findings"] {
        fs::create_dir(options.output.join(name))?;
    }
    write_json(options.output.join("config.json"), &options)?;
    write_json(
        options.output.join("driver-build.json"),
        &json!({"version": env!("CARGO_PKG_VERSION"), "arch": std::env::consts::ARCH,
            "debug_assertions": cfg!(debug_assertions)}),
    )?;
    write_json(
        options.output.join("environment.json"),
        &json!({
            "TSAN_OPTIONS": std::env::var("TSAN_OPTIONS").unwrap_or_else(|_| "ignore_noninstrumented_modules=1".into()),
            "OMP_TOOL_LIBRARIES": std::env::var("OMP_TOOL_LIBRARIES").ok(),
            "ARCHER_OPTIONS": std::env::var("ARCHER_OPTIONS").ok(),
            "OMP_PROC_BIND": std::env::var("OMP_PROC_BIND").ok(),
            "OMP_PLACES": std::env::var("OMP_PLACES").ok(),
            "LD_LIBRARY_PATH": std::env::var("LD_LIBRARY_PATH").ok()
        }),
    )?;
    let mut events = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(options.output.join("events.jsonl"))?;
    let mut corpus = InMemoryCorpus::<BytesInput>::new();
    let mut pending = VecDeque::new();
    let mut inputs = BTreeSet::new();
    for path in files {
        let bytes = fs::read(&path)?;
        if inputs.insert(bytes.clone()) {
            let index = corpus.count();
            let id = corpus.add(Testcase::new(BytesInput::new(bytes.clone())))?;
            fs::write(options.output.join(format!("queue/{index:06}.mlir")), bytes)?;
            write_json(
                options.output.join(format!("queue/{index:06}.json")),
                &json!({"seed_file": path, "parent": null}),
            )?;
            pending.push_back((index, id));
        }
    }
    let mut rng = StdRand::with_seed(options.seed);
    let mut global_features = BTreeSet::new();
    let mut attempts = 0u64;
    let mut usable = 0u64;
    let mut finding_count = 0u64;
    let mut saturated_count = 0u64;
    let start = Instant::now();
    let time_expired = || {
        options
            .duration_secs
            .is_some_and(|seconds| start.elapsed() >= Duration::from_secs(seconds))
    };
    let stop_requested = || options.stop_file.as_ref().is_some_and(|path| path.exists());
    while attempts < options.iterations && !time_expired() && !stop_requested() {
        let Some((index, id)) = pending.pop_front() else {
            break;
        };
        // Validate against a fixed source distribution; queued children run later.
        let testcase = corpus.get(id)?.borrow();
        let source = testcase.input().as_ref().ok_or("missing corpus input")?;
        let source_bytes: &[u8] = source.as_ref();
        let source_bytes = source_bytes.to_vec();
        drop(testcase);
        let mut confidence = Confidence::new(options.block_size, options.alpha, options.epsilon);
        let mut tainted = false;
        let mut source_attempts = 0;
        while source_attempts < options.max_trials
            && attempts < options.iterations
            && !time_expired()
            && !stop_requested()
        {
            let trial = attempts;
            attempts += 1;
            source_attempts += 1;
            let dir = options.output.join(format!("runs/{trial:08}"));
            fs::create_dir(&dir)?;
            fs::write(dir.join("input.mlir"), &source_bytes)?;
            let request = Request {
                backend: options.backend,
                tsan_triage: options.tsan_triage,
                comparison: options.comparison,
                campaigns: options.campaigns,
                max_campaigns: options.max_campaigns,
                executions_per_campaign: options.executions_per_campaign,
                team_size: options.team_size,
                bootstrap_replicates: options.bootstrap_replicates,
                tvd_threshold: options.tvd_threshold,
                min_change: options.min_change,
                confidence_level: options.confidence_level,
                input: dir.join("input.mlir"),
                artifacts: dir.clone(),
                seed: (rng.next() & 0x7fff_ffff) as u32,
                transform: options.transform.clone(),
                model: options.model.clone(),
                reps: options.reps,
                replay_reps: options.replay_reps,
                max_runs: options.max_runs,
                max_apply: options.max_apply,
            };
            write_json(dir.join("request.json"), &request)?;
            let outcome = supervisor::execute(
                &options.worker,
                &dir.join("request.json"),
                &dir,
                Duration::from_millis(options.timeout_ms),
            )?;
            let mut category = match outcome {
                Outcome::Completed => "protocol-error",
                Outcome::Timeout => "timeout",
                Outcome::Crash(_) => "crash",
                Outcome::Error(66) => "sanitizer-report",
                Outcome::Error(_) => "worker-error",
            };
            let compiler_coverage = coverage::read(&dir)?;
            options.coverage_artifacts.write(&dir, &compiler_coverage)?;
            // The raw guard map is 16 MiB; coverage.json/events keep the useful summary.
            fs::remove_file(dir.join("coverage.map"))?;
            if outcome == Outcome::Completed
                && !options.allow_uninstrumented
                && (compiler_coverage.guards == 0 || compiler_coverage.edges.is_empty())
            {
                return Err("no compiler coverage observed: use the default Cargo compiler build (see README.md), or explicitly use --allow-uninstrumented for a smoke test with an external compiler".into());
            }
            let mut admitted = None;
            let mut new_features = 0;
            let mut confidence_novel_features = 0;
            let mut confidence_feature_count = 0;
            let mut feedback_feature_count = 0;
            let mut report_log = Value::Null;
            if outcome == Outcome::Completed {
                match fs::read(dir.join("response.json"))
                    .map_err(|e| e.to_string())
                    .and_then(|bytes| {
                        serde_json::from_slice::<Report>(&bytes).map_err(|e| e.to_string())
                    }) {
                    Ok(report) if report.usable() => {
                        report_log = report_event(&report);
                        category = "ok";
                        usable += 1;
                        let confidence_features = options
                            .confidence_features
                            .select(&report, &compiler_coverage.edges);
                        confidence_feature_count = confidence_features.len();
                        confidence_novel_features = confidence.novelty_count(&confidence_features);
                        confidence.observe(confidence_features);
                        let mut feedback_features = report.features();
                        feedback_features.extend(compiler_coverage.edges.iter().cloned());
                        feedback_feature_count = feedback_features.len();
                        new_features = feedback_features.difference(&global_features).count();
                        global_features.extend(feedback_features);
                        let child = report.transformed_mlir.into_bytes();
                        if new_features > 0
                            && child.len() <= options.max_input_bytes
                            && corpus.count() < options.max_corpus
                            && !inputs.contains(&child)
                        {
                            let child_index = corpus.count();
                            let child_id =
                                corpus.add(Testcase::new(BytesInput::new(child.clone())))?;
                            fs::write(
                                options.output.join(format!("queue/{child_index:06}.mlir")),
                                &child,
                            )?;
                            write_json(
                                options.output.join(format!("queue/{child_index:06}.json")),
                                &json!({"parent": index, "trial": trial, "request": request,
                                    "new_features": new_features,
                                    "confidence_novel_features": confidence_novel_features}),
                            )?;
                            inputs.insert(child);
                            pending.push_back((child_index, child_id));
                            admitted = Some(child_index);
                        }
                    }
                    Ok(report) => {
                        report_log = report_event(&report);
                        category = match report.status.as_str() {
                            "WARN" => "inconclusive",
                            "ERROR" if report.source_runs > 0 && report.transformed_runs > 0 => {
                                "oracle-failure"
                            }
                            "ERROR" if !report.transform_applied => "rejected",
                            "ERROR" => "pipeline-error",
                            _ => "protocol-error",
                        };
                    }
                    Err(error) => {
                        fs::write(dir.join("protocol-error.txt"), error)?;
                    }
                }
            }
            if category != "ok" {
                tainted = true;
            }
            if !matches!(category, "ok" | "rejected") {
                finding_count += 1;
                write_json(
                    options.output.join(format!("findings/{trial:08}.json")),
                    &json!({"category": category, "parent": index, "request": request,
                        "artifacts": dir, "worker_outcome": format!("{outcome:?}")}),
                )?;
            }
            let event = json!({"trial": trial, "parent": index, "seed": request.seed,
                "compiler_guards": compiler_coverage.guards, "compiler_edges": compiler_coverage.edges.len(),
                "category": category, "admitted": admitted, "new_features": new_features,
                "feedback_features": feedback_feature_count,
                "confidence_features": confidence_feature_count,
                "confidence_novel_features": confidence_novel_features,
                "confidence_block": confidence.block,
                "confidence_samples": confidence.samples,
                "confidence_target_samples": confidence.target_samples,
                "confidence_upper_novelty_probability": confidence.upper_novelty_probability,
                "report": report_log,
                "corpus_size": corpus.count(), "elapsed_seconds": start.elapsed().as_secs_f64()});
            writeln!(events, "{event}")?;
            events.flush()?;
            let saturated = confidence.saturated && !tainted;
            write_json(
                options.output.join(format!("sources/{index:06}.json")),
                &json!({"source": index, "attempts": source_attempts, "tainted": tainted,
                    "state": if saturated { "saturated" } else { "inconclusive" },
                    "confidence": confidence}),
            )?;
            eprintln!(
                "trial={trial} source={index} {category} queue={} feedback_features={} confidence_novel={} bound={:?}",
                corpus.count(),
                global_features.len(),
                confidence_novel_features,
                confidence.upper_novelty_probability
            );
            if saturated {
                saturated_count += 1;
                break;
            }
        }
    }
    let summary = json!({"attempts": attempts, "usable_trials": usable,
        "findings": finding_count, "corpus_size": corpus.count(),
        "feedback_features": global_features.len(), "saturated_sources": saturated_count,
        "compiler_edges": global_features.iter().filter(|f| f.starts_with("compiler-edge:")).count(),
        "pending_sources": pending.len(), "elapsed_seconds": start.elapsed().as_secs_f64(),
        "stop_reason": if stop_requested() { "stop-request" }
            else if time_expired() { "time-budget" }
            else if attempts >= options.iterations { "campaign-budget" } else { "queue-drained" }});
    write_json(options.output.join("summary.json"), &summary)?;
    println!("{summary}");
    if usable == 0 {
        return Err(
            "campaign produced no usable MR trials; inspect worker logs and findings".into(),
        );
    }
    Ok(())
}
