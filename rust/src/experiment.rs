// Copyright © 2026 Hanzo AI. MIT License.

//! The falsifiable frame: a hypothesis, a prediction, the attempts, and the verdict.

use crate::provenance::Host;
use serde::{Deserialize, Serialize};

/// The epistemic outcome of a falsifiable experiment — distinct from execution status.
///
/// A refutation is a first-class, durable result, recorded as clearly as a proof; that is
/// the whole point of an evidence plane. Being an enum rather than a string means an
/// out-of-set verdict cannot be constructed at all, which is stricter than the Go SDK's
/// validate-at-`Conclude` and costs nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    Proven,
    Refuted,
    Inconclusive,
}

impl Verdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Verdict::Proven => "proven",
            Verdict::Refuted => "refuted",
            Verdict::Inconclusive => "inconclusive",
        }
    }
}

/// One measured attempt's outcome.
///
/// Named `Attempt` rather than the Go SDK's `Result` to avoid shadowing `std::result::Result`
/// at every call site — the wire contract is unchanged, only the Rust-side name.
///
/// `status` of `faulted` or `failed` retains a negative result without counting it toward
/// the running accuracy: a crashed arm is evidence, but it is not a wrong answer.
#[derive(Debug, Clone, Default)]
pub struct Attempt {
    pub answer: String,
    pub correct: bool,
    pub response: String,
    pub gold: String,
    /// Defaults to `hanzo-measured` when blank.
    pub source: String,
    /// Defaults to `complete` when blank.
    pub status: String,
}

impl Attempt {
    /// A correct attempt with just the answer recorded.
    pub fn ok(answer: impl Into<String>) -> Self {
        Attempt { answer: answer.into(), correct: true, ..Default::default() }
    }

    /// An incorrect attempt, carrying what was produced and what was expected.
    pub fn wrong(answer: impl Into<String>, gold: impl Into<String>) -> Self {
        Attempt { answer: answer.into(), gold: gold.into(), correct: false, ..Default::default() }
    }

    /// An arm that did not complete. Retained as evidence, excluded from accuracy.
    pub fn faulted(response: impl Into<String>) -> Self {
        Attempt {
            response: response.into(),
            correct: false,
            status: "faulted".into(),
            ..Default::default()
        }
    }

    pub(crate) fn normalized(mut self) -> Self {
        if self.source.is_empty() {
            self.source = "hanzo-measured".into();
        }
        if self.status.is_empty() {
            self.status = "complete".into();
        }
        self
    }

    /// Whether this attempt counts toward the running accuracy denominator.
    pub(crate) fn counts(&self) -> bool {
        !matches!(self.status.as_str(), "faulted" | "failed")
    }
}

/// The scientific frame plus the self-documenting narrative.
///
/// These json names are the cross-language contract, shared verbatim with the Go SDK and
/// the Python producer. The server keys and dedupes semantically, so a Rust harness and a
/// Python one land in the same row.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct Meta {
    pub doc: String,
    pub commits: Vec<String>,
    pub note: String,
    pub host: Host,
    pub hypothesis: String,
    pub predict: String,
    pub verdict: String,
    pub because: String,
    pub log: Vec<String>,
}

/// One experiment version on the wire. `project`, `ts` and `cost` are server-stamped.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ExpPayload {
    pub id: String,
    pub kind: String,
    pub subject: String,
    pub task: String,
    pub metric: String,
    pub value: f64,
    pub n: i64,
    pub n_total: i64,
    pub status: String,
    pub git_sha: String,
    pub git_branch: String,
    pub git_dirty: bool,
    pub lib_versions: std::collections::BTreeMap<String, String>,
    pub meta: Meta,
}

/// One measured attempt on the wire.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct AttemptPayload {
    pub benchmark: String,
    pub item: String,
    pub model: String,
    pub answer: String,
    pub correct: bool,
    pub response: String,
    pub gold: String,
    pub source: String,
    pub status: String,
}

/// One diary artifact. Bytes are base64 in `content`; the server content-addresses by sha256.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ArtifactPayload {
    pub content: String,
    pub sha256: String,
    pub kind: String,
    pub run_id: String,
    pub git_sha: String,
    pub git_branch: String,
    pub git_dirty: bool,
    pub lib_versions: std::collections::BTreeMap<String, String>,
}

/// One upload batch: experiments and their attempts as two flat arrays.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct IngestReq {
    pub experiments: Vec<ExpPayload>,
    pub attempts: Vec<AttemptPayload>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdict_serializes_to_the_cross_language_strings() {
        assert_eq!(serde_json::to_string(&Verdict::Proven).unwrap(), "\"proven\"");
        assert_eq!(serde_json::to_string(&Verdict::Refuted).unwrap(), "\"refuted\"");
        assert_eq!(
            serde_json::to_string(&Verdict::Inconclusive).unwrap(),
            "\"inconclusive\""
        );
    }

    #[test]
    fn a_faulted_arm_is_evidence_but_not_a_wrong_answer() {
        assert!(!Attempt::faulted("oom").counts());
        assert!(Attempt::wrong("4", "5").counts());
        assert!(Attempt::ok("5").counts());
    }

    #[test]
    fn blank_source_and_status_take_the_documented_defaults() {
        let a = Attempt::ok("x").normalized();
        assert_eq!(a.source, "hanzo-measured");
        assert_eq!(a.status, "complete");
    }

    #[test]
    fn ingest_body_always_carries_both_arrays() {
        // The Go SDK converts nil to []; an absent key is a different wire shape and the
        // server distinguishes them, so an empty batch must still emit both arrays.
        let body = IngestReq { experiments: vec![], attempts: vec![] };
        let j = serde_json::to_string(&body).unwrap();
        assert_eq!(j, r#"{"experiments":[],"attempts":[]}"#);
    }
}
