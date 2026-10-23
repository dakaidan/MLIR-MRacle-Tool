use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, path::PathBuf};

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
#[repr(u32)]
pub enum Backend {
    #[default]
    Jit = 0,
    Native = 1,
}

/// Which comparison the engine runs. The outcome method replays and triages
/// individual outcomes; the tolerance and adaptive methods compare outcome
/// distributions instead.
#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize, clap::ValueEnum)]
#[repr(u32)]
pub enum Comparison {
    #[default]
    #[serde(rename = "outcome", alias = "agitation")]
    #[value(name = "outcome", alias = "agitation")]
    Agitation = 0,
    #[serde(rename = "tolerance", alias = "tvd")]
    #[value(name = "tolerance", alias = "tvd")]
    Tvd = 1,
    #[serde(rename = "adaptive", alias = "calibrated")]
    #[value(name = "adaptive", alias = "calibrated")]
    Calibrated = 2,
}

/// Budgets of the campaign methods, matching the engine's own defaults.
pub const DEFAULT_CAMPAIGNS: i32 = 10;
pub const DEFAULT_MAX_CAMPAIGNS: i32 = 50;
pub const DEFAULT_EXECUTIONS_PER_CAMPAIGN: i32 = 2000;
pub const DEFAULT_TEAM_SIZE: i32 = 4;
pub const DEFAULT_BOOTSTRAP_REPLICATES: i32 = 2000;
pub const DEFAULT_TVD_THRESHOLD: f64 = 0.05;
pub const DEFAULT_MIN_CHANGE: f64 = 0.005;
pub const DEFAULT_CONFIDENCE_LEVEL: f64 = 0.95;

fn default_campaigns() -> i32 {
    DEFAULT_CAMPAIGNS
}
fn default_max_campaigns() -> i32 {
    DEFAULT_MAX_CAMPAIGNS
}
fn default_executions_per_campaign() -> i32 {
    DEFAULT_EXECUTIONS_PER_CAMPAIGN
}
fn default_team_size() -> i32 {
    DEFAULT_TEAM_SIZE
}
fn default_bootstrap_replicates() -> i32 {
    DEFAULT_BOOTSTRAP_REPLICATES
}
fn default_tvd_threshold() -> f64 {
    DEFAULT_TVD_THRESHOLD
}
fn default_min_change() -> f64 {
    DEFAULT_MIN_CHANGE
}
fn default_confidence_level() -> f64 {
    DEFAULT_CONFIDENCE_LEVEL
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    pub input: PathBuf,
    pub artifacts: PathBuf,
    pub seed: u32,
    pub transform: String,
    pub model: String,
    pub reps: i32,
    pub replay_reps: i32,
    pub max_runs: i32,
    pub max_apply: i32,
    #[serde(default)]
    pub backend: Backend,
    #[serde(default)]
    pub tsan_triage: bool,
    #[serde(default)]
    pub comparison: Comparison,
    #[serde(default = "default_campaigns")]
    pub campaigns: i32,
    #[serde(default = "default_max_campaigns")]
    pub max_campaigns: i32,
    #[serde(default = "default_executions_per_campaign")]
    pub executions_per_campaign: i32,
    #[serde(default = "default_team_size")]
    pub team_size: i32,
    #[serde(default = "default_bootstrap_replicates")]
    pub bootstrap_replicates: i32,
    #[serde(default = "default_tvd_threshold")]
    pub tvd_threshold: f64,
    #[serde(default = "default_min_change")]
    pub min_change: f64,
    #[serde(default = "default_confidence_level")]
    pub confidence_level: f64,
}

// Parse strictly at the worker boundary: missing observations cannot mean OK.
#[derive(Debug, Serialize, Deserialize)]
pub struct Report {
    pub status: String,
    pub transform_applied: bool,
    pub source_runs: u64,
    pub transformed_runs: u64,
    pub applied: Vec<AppliedTransform>,
    pub relation: Relation,
    pub source_outcomes: Vec<ObservedOutcome>,
    pub transformed_outcomes: Vec<ObservedOutcome>,
    pub transformed_mlir: String,
    pub ir_features: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AppliedTransform {
    pub name: String,
    pub target_function: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Relation {
    #[serde(rename = "type")]
    pub kind: RelationKind,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RelationKind {
    Equality,
    Subset,
    Superset,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ObservedOutcome {
    pub outcome: Vec<i64>,
    pub count: u64,
}

impl Report {
    pub fn usable(&self) -> bool {
        self.status == "OK"
            && self.transform_applied
            && !self.applied.is_empty()
            && self.source_runs > 0
            && self.transformed_runs > 0
            && !self.source_outcomes.is_empty()
            && !self.transformed_outcomes.is_empty()
            && !self.transformed_mlir.is_empty()
            && !self.ir_features.is_empty()
            && self
                .applied
                .iter()
                .all(|t| !t.name.is_empty() && !t.target_function.is_empty())
            && self.source_outcomes.iter().all(|o| o.count > 0)
            && self.transformed_outcomes.iter().all(|o| o.count > 0)
    }
    fn feature_context(&self) -> String {
        format!(
            "{}:{}",
            serde_json::json!(self.applied),
            serde_json::json!(self.relation)
        )
    }
    /// Metamorphic-relation feedback: transform choice and expected relation.
    pub fn mr_features(&self) -> BTreeSet<String> {
        let mut features = BTreeSet::new();
        let context = self.feature_context();
        features.insert(format!("mr:{context}"));
        features
    }
    /// Semantic feedback, not compiler edge coverage. Counts are excluded:
    /// ordinary sampling noise must not produce endlessly new corpus entries.
    pub fn features(&self) -> BTreeSet<String> {
        let mut features = self.mr_features();
        let context = self.feature_context();
        features.extend(
            self.ir_features
                .iter()
                .map(|feature| format!("ir:{feature}")),
        );
        for (side, outcomes) in [
            ("source", &self.source_outcomes),
            ("transformed", &self.transformed_outcomes),
        ] {
            for outcome in outcomes {
                let value = serde_json::json!(outcome.outcome);
                features.insert(format!("{context}:{side}:{value}"));
            }
        }
        features
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn report() -> Report {
        serde_json::from_value(serde_json::json!({
            "status": "OK", "transform_applied": true,
            "source_runs": 10, "transformed_runs": 10,
            "applied": [{"name": "insert-fence", "target_function": "main"}],
            "relation": {"type": "subset"},
            "source_outcomes": [{"outcome": [0], "count": 10}],
            "transformed_outcomes": [{"outcome": [0], "count": 10}],
            "transformed_mlir": "module {}",
            "ir_features": ["count:builtin.module:1"]
        }))
        .unwrap()
    }
    #[test]
    fn frequency_noise_is_not_novel_but_outcomes_are() {
        let mut r = report();
        let features = r.features();
        let mr_features = r.mr_features();
        r.source_outcomes[0].count = 20;
        assert_eq!(features, r.features());
        assert_eq!(mr_features, r.mr_features());
        r.source_outcomes[0].outcome = vec![1];
        assert_ne!(features, r.features());
        assert_eq!(mr_features, r.mr_features());
    }
    #[test]
    fn warnings_and_empty_observations_cannot_pass() {
        let mut r = report();
        assert!(r.usable());
        r.status = "WARN".into();
        assert!(!r.usable());
        r.status = "OK".into();
        r.source_outcomes.clear();
        assert!(!r.usable());
        assert!(serde_json::from_str::<Report>("{}").is_err());
    }
}
