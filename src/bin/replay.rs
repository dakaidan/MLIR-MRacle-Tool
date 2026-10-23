use clap::Parser;
use mlir_mracle::protocol::{Comparison, Request};
use mlir_mracle::supervisor;
use std::{path::PathBuf, time::Duration};

#[derive(Parser)]
#[command(about = "Replay a saved MR trial into a new artifact directory")]
struct Options {
    request: PathBuf,
    #[arg(short, long)]
    output: PathBuf,
    #[arg(long, default_value_os_t = mlir_mracle::default_worker())]
    worker: PathBuf,
    #[arg(long, default_value_t = 60000)]
    timeout_ms: u64,
    #[arg(long)]
    reps: Option<i32>,
    #[arg(long)]
    replay_reps: Option<i32>,
    #[arg(long)]
    max_runs: Option<i32>,
    #[arg(long)]
    max_apply: Option<i32>,
    #[arg(long)]
    executions_per_campaign: Option<i32>,
    #[arg(long)]
    campaigns: Option<i32>,
    #[arg(long)]
    max_campaigns: Option<i32>,
}

fn main() {
    if let Err(error) = run(Options::parse()) {
        eprintln!("mracle-replay: {error}");
        std::process::exit(1);
    }
}

fn run(options: Options) -> Result<(), Box<dyn std::error::Error>> {
    let saved = options.request.canonicalize()?;
    let worker = options.worker.canonicalize()?;
    if options.timeout_ms == 0
        || options.reps.is_some_and(|value| value <= 0)
        || options.replay_reps.is_some_and(|value| value <= 0)
        || options.max_runs.is_some_and(|value| value <= 0)
        || options.max_apply.is_some_and(|value| value <= 0)
        || options
            .executions_per_campaign
            .is_some_and(|value| value <= 0)
        || options.campaigns.is_some_and(|value| value < 2)
        || options.max_campaigns.is_some_and(|value| value < 2)
    {
        return Err("timeout and replay budget overrides must be positive".into());
    }
    let mut request: Request = serde_json::from_slice(&std::fs::read(&saved)?)?;
    let campaign_method = request.comparison != Comparison::Agitation;
    if campaign_method
        && (options.reps.is_some() || options.replay_reps.is_some() || options.max_runs.is_some())
    {
        return Err("--reps, --replay-reps and --max-runs belong to the outcome method".into());
    }
    if !campaign_method
        && (options.executions_per_campaign.is_some()
            || options.campaigns.is_some()
            || options.max_campaigns.is_some())
    {
        return Err("campaign overrides belong to the tolerance and adaptive methods".into());
    }
    if let Some(reps) = options.reps {
        request.reps = reps;
    }
    if let Some(replay_reps) = options.replay_reps {
        request.replay_reps = replay_reps;
    }
    if let Some(max_runs) = options.max_runs {
        request.max_runs = max_runs;
    }
    if let Some(max_apply) = options.max_apply {
        request.max_apply = max_apply;
    }
    if let Some(executions) = options.executions_per_campaign {
        request.executions_per_campaign = executions;
    }
    if let Some(campaigns) = options.campaigns {
        request.campaigns = campaigns;
    }
    if let Some(max_campaigns) = options.max_campaigns {
        request.max_campaigns = max_campaigns;
    }
    if request.max_campaigns < request.campaigns {
        return Err("--max-campaigns must be at least --campaigns".into());
    }
    let input = std::fs::read(
        saved
            .parent()
            .ok_or("request has no parent")?
            .join("input.mlir"),
    )?;
    std::fs::create_dir(&options.output)?;
    request.artifacts = options.output.canonicalize()?;
    request.input = request.artifacts.join("input.mlir");
    std::fs::write(&request.input, input)?;
    let path = request.artifacts.join("request.json");
    std::fs::write(&path, serde_json::to_vec_pretty(&request)?)?;
    let outcome = supervisor::execute(
        &worker,
        &path,
        &request.artifacts,
        Duration::from_millis(options.timeout_ms),
    )?;
    println!("{outcome:?}: {}", request.artifacts.display());
    if outcome != supervisor::Outcome::Completed {
        return Err("worker did not complete; see logs".into());
    }
    let response = std::fs::read(request.artifacts.join("response.json"))?;
    let report: mlir_mracle::protocol::Report = serde_json::from_slice(&response)?;
    println!("oracle status: {}", report.status);
    Ok(())
}
