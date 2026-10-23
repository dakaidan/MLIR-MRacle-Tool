//! Validation blocks with an alpha-spending correction for repeated looks.
use serde::Serialize;
use std::collections::BTreeSet;

#[derive(Debug, Serialize)]
pub struct Confidence {
    pub seen: BTreeSet<String>,
    frozen: BTreeSet<String>,
    pub block: u64,
    pub samples: u64,
    pub target_samples: u64,
    pub novel_samples: u64,
    pub upper_novelty_probability: Option<f64>,
    pub saturated: bool,
    min_block_size: u64,
    alpha: f64,
    epsilon: f64,
}
impl Confidence {
    pub fn new(min_block_size: u64, alpha: f64, epsilon: f64) -> Self {
        assert!(min_block_size > 0 && alpha > 0.0 && alpha < 1.0 && epsilon > 0.0 && epsilon < 1.0);
        Self {
            seen: BTreeSet::new(),
            frozen: BTreeSet::new(),
            block: 1,
            samples: 0,
            target_samples: required_samples(1, min_block_size, alpha, epsilon),
            novel_samples: 0,
            upper_novelty_probability: None,
            saturated: false,
            min_block_size,
            alpha,
            epsilon,
        }
    }
    pub fn novelty_count(&self, features: &BTreeSet<String>) -> usize {
        features.difference(&self.frozen).count()
    }
    pub fn observe(&mut self, features: BTreeSet<String>) {
        if !features.is_subset(&self.frozen) {
            self.novel_samples += 1;
        }
        self.seen.extend(features);
        self.samples += 1;
        if self.samples >= self.target_samples {
            // Sum_{k>=1} alpha/(k(k+1)) = alpha. For zero successes,
            // the exact one-sided binomial upper limit is 1-alpha_k^(1/n).
            let alpha_k = self.alpha / (self.block as f64 * (self.block as f64 + 1.0));
            let upper = if self.novel_samples == 0 {
                -((alpha_k.ln() / self.samples as f64).exp_m1())
            } else {
                1.0
            };
            self.upper_novelty_probability = Some(upper);
            self.saturated = upper <= self.epsilon;
            self.frozen = self.seen.clone();
            self.block += 1;
            self.samples = 0;
            self.target_samples =
                required_samples(self.block, self.min_block_size, self.alpha, self.epsilon);
            self.novel_samples = 0;
        }
    }
}

fn required_samples(block: u64, min_block_size: u64, alpha: f64, epsilon: f64) -> u64 {
    let alpha_k = alpha / (block as f64 * (block as f64 + 1.0));
    let needed = (alpha_k.ln() / (1.0 - epsilon).ln()).ceil() as u64;
    needed.max(min_block_size).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn features(s: &str) -> BTreeSet<String> {
        [s.to_owned()].into()
    }
    #[test]
    fn discovery_then_independent_validation() {
        let mut c = Confidence::new(100, 0.05, 0.05);
        for _ in 0..100 {
            c.observe(features("a"));
        }
        assert!(!c.saturated);
        for _ in 0..99 {
            c.observe(features("a"));
        }
        assert!(!c.saturated);
        c.observe(features("a"));
        assert!(c.saturated);
        assert!(
            (c.upper_novelty_probability.unwrap() - (1.0 - (0.05_f64 / 6.0).powf(0.01))).abs()
                < 1e-12
        );
    }
    #[test]
    fn late_novelty_prevents_stopping_and_empty_budget_does_not_pass() {
        let mut c = Confidence::new(100, 0.05, 0.05);
        assert!(!c.saturated);
        for _ in 0..199 {
            c.observe(features("a"));
        }
        c.observe(features("b"));
        assert!(!c.saturated);
        assert_eq!(c.upper_novelty_probability, Some(1.0));
    }
    #[test]
    fn later_blocks_grow_to_keep_zero_novelty_decisive() {
        let mut c = Confidence::new(128, 0.05, 0.05);
        assert_eq!(c.target_samples, 128);
        for _ in 0..128 {
            c.observe(features("a"));
        }
        for name in ["b", "c", "d", "e"] {
            for _ in 0..127 {
                c.observe(features("a"));
            }
            c.observe(features(name));
        }
        assert_eq!(c.block, 6);
        assert!(c.target_samples > 128);
        for _ in 0..c.target_samples {
            c.observe(features("a"));
        }
        assert!(c.saturated);
        assert!(c.upper_novelty_probability.unwrap() <= 0.05);
    }
}
