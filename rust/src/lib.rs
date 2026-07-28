// Copyright © 2026 Hanzo AI. MIT License.

//! Rust SDK for the Hanzo `/v1/research` evidence plane.
//!
//! The ONE way a Rust program records R&D evidence. Mirrors the Go SDK
//! (`github.com/hanzoai/research`) and the Python producer verb-for-verb: the server keys
//! each experiment on `(project, id = kind:subject:task)`, so a Rust harness, a Python
//! notebook and a Go service upsert the same row.
//!
//! Rust is where the GPU harnesses live, and until this crate existed they had nowhere to
//! record — which is why kernel results survived as loose `.log` files on the box that ran
//! them instead of as evidence anyone could query.
//!
//! # Inert until configured
//!
//! With no API key the client is **inert**: every verb is a cheap no-op that returns `Ok`.
//! That is what makes it safe to leave a `record` call on a production path — unconfigured
//! it costs a branch, and configured it records. Recording is best-effort by design: a
//! transport error is returned, never panicked, and never interrupts the work being measured.
//!
//! # `kind` is open, on purpose
//!
//! `kind` is a free string, never an enum: `benchmark`, `kernel-perf`, `training`,
//! `ablation` **and** `marketing-experiment`, `ad-test`, `pricing-test`, `growth-experiment`.
//! A marketing A/B records identically to a kernel A/B — state a hypothesis, record the
//! arms, conclude proven or refuted. That symmetry is the point: one plane measures whether
//! a change helped, whatever the change was.
//!
//! ```no_run
//! use hanzo_research::{Research, Config, Verdict, Attempt};
//!
//! let c = Research::new(Config { project: "engine-bench".into(), ..Default::default() });
//! let mut exp = c
//!     .experiment("kernel-perf", "gfx1151", "q4k-decode")
//!     .metric("ratio")
//!     .hypothesis("the DSL i8 coopmat arm beats the dp4a hand tile")
//!     .predict("ratio > 1.0 across ffn/attn shapes")
//!     .start()?;
//!
//! exp.record("n=4096", "mmq_q4k_rt", Attempt::wrong("0.62", "1.0"))?;
//! exp.log("i8 matrix-core does not reach f16 matrix-core throughput here");
//! exp.conclude(Verdict::Refuted, "0.51-0.78x of the hand tile; int8 alone will not close it", 0.62)?;
//! # Ok::<(), hanzo_research::Error>(())
//! ```

mod experiment;
mod provenance;

pub use experiment::{Attempt, Verdict};
pub use provenance::{find_repo, lib_version, Host, Vcs};

use experiment::{ArtifactPayload, AttemptPayload, ExpPayload, IngestReq, Meta};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::Duration;

const DEFAULT_BASE: &str = "https://api.hanzo.ai";
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
/// Bound the response read so a hostile or broken server cannot stream an unbounded body.
const MAX_RESPONSE: usize = 8 << 20;

/// What can go wrong recording evidence. Every variant is returned, never panicked.
#[derive(Debug)]
pub enum Error {
    /// Transport failed — the experiment still happened, only the recording did not.
    Transport(String),
    /// The server rejected the request.
    Server { method: &'static str, path: String, status: u16, message: String },
    /// The response exceeded the read cap.
    ResponseTooLarge { path: String, cap: usize },
    /// The response was not the JSON we expect.
    Decode(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Transport(e) => write!(f, "research: transport: {e}"),
            Error::Server { method, path, status, message } => {
                write!(f, "research: {method} {path}: {status}: {message}")
            }
            Error::ResponseTooLarge { path, cap } => {
                write!(f, "research: {path}: response exceeds {cap}-byte cap")
            }
            Error::Decode(e) => write!(f, "research: decode: {e}"),
        }
    }
}

impl std::error::Error for Error {}

/// Client configuration. Every unset field falls back to the environment or a default.
#[derive(Debug, Clone, Default)]
pub struct Config {
    /// Defaults to `$HANZO_API_BASE`, else `https://api.hanzo.ai`.
    pub base: String,
    /// Defaults to `$HANZO_API_KEY`. Empty leaves the client inert.
    pub key: String,
    /// Project sub-scope. Defaults to `$HANZO_PROJECT`.
    pub project: String,
    /// Repo to read provenance from. Defaults to the nearest enclosing git repo.
    pub repo: String,
    /// Library names whose `<NAME>_VERSION` env var is captured with each record.
    pub libs: Vec<String>,
    pub timeout: Option<Duration>,
}

/// A configured client: base URL, per-org key, project sub-scope, provenance repo.
#[derive(Debug, Clone)]
pub struct Research {
    base: String,
    key: String,
    project: String,
    repo: String,
    libs: Vec<String>,
    timeout: Duration,
}

fn env_or(explicit: &str, var: &str, fallback: &str) -> String {
    if !explicit.is_empty() {
        return explicit.to_string();
    }
    std::env::var(var)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| fallback.to_string())
}

impl Research {
    /// Builds a client, filling every unset field from the environment or its default.
    ///
    /// The key is read from `$HANZO_API_KEY` — sourced from KMS through the process
    /// environment, never hardcoded.
    pub fn new(cfg: Config) -> Self {
        let repo = if cfg.repo.is_empty() { find_repo("") } else { cfg.repo.clone() };
        Research {
            base: env_or(&cfg.base, "HANZO_API_BASE", DEFAULT_BASE)
                .trim_end_matches('/')
                .to_string(),
            key: env_or(&cfg.key, "HANZO_API_KEY", ""),
            project: env_or(&cfg.project, "HANZO_PROJECT", ""),
            repo,
            libs: cfg.libs,
            timeout: cfg.timeout.unwrap_or(DEFAULT_TIMEOUT),
        }
    }

    /// Whether this client will actually transmit.
    ///
    /// Unconfigured clients are inert rather than failing, so a `record` call can be left
    /// in a production path and simply cost nothing where no key is provisioned.
    pub fn enabled(&self) -> bool {
        !self.key.is_empty()
    }

    /// Opens an experiment handle for `kind:subject:task`.
    ///
    /// `kind` is an open string — `kernel-perf` and `marketing-experiment` are equally valid.
    pub fn experiment<'a>(
        &'a self,
        kind: &str,
        subject: &str,
        task: &str,
    ) -> ExperimentBuilder<'a> {
        ExperimentBuilder {
            client: self,
            kind: kind.into(),
            subject: subject.into(),
            task: task.into(),
            metric: String::new(),
            n_total: 0,
            note: String::new(),
            hypothesis: String::new(),
            predict: String::new(),
        }
    }

    /// Lists canonical experiments, optionally narrowed by project and kind.
    pub fn query(&self, project: &str, kind: &str) -> Result<Vec<Run>, Error> {
        if !self.enabled() {
            return Ok(Vec::new());
        }
        let mut path = String::from("/v1/research/experiments");
        let mut q = Vec::new();
        if !project.is_empty() {
            q.push(format!("project={}", urlencode(project)));
        }
        if !kind.is_empty() {
            q.push(format!("kind={}", urlencode(kind)));
        }
        if !q.is_empty() {
            path.push('?');
            path.push_str(&q.join("&"));
        }
        self.get(&path)
    }

    /// The headline aggregate plus per-kind totals.
    pub fn totals(&self, project: &str) -> Result<Totals, Error> {
        if !self.enabled() {
            return Ok(Totals::default());
        }
        let mut path = String::from("/v1/research/totals");
        if !project.is_empty() {
            path.push_str(&format!("?project={}", urlencode(project)));
        }
        self.get(&path)
    }

    /// Sets visibility/consent for a stable id. Records are private by default; publication
    /// is a separate authorized grant, never implied by recording.
    pub fn grant(&self, g: GrantRequest) -> Result<i64, Error> {
        if !self.enabled() {
            return Ok(0);
        }
        let out: GrantResult = self.post("/v1/research/grants", &g)?;
        Ok(out.updated)
    }

    fn ingest(&self, exps: Vec<ExpPayload>, atts: Vec<AttemptPayload>) -> Result<(), Error> {
        if !self.enabled() {
            return Ok(());
        }
        let _: serde_json::Value =
            self.post("/v1/research/experiments", &IngestReq { experiments: exps, attempts: atts })?;
        Ok(())
    }

    fn artifact(&self, a: ArtifactPayload) -> Result<(), Error> {
        if !self.enabled() {
            return Ok(());
        }
        let _: serde_json::Value = self.post("/v1/research/artifacts", &a)?;
        Ok(())
    }

    /// Auth is ONLY the per-org key; the gateway mints the validated principal and project
    /// scope from it. The client never sends `X-User-Id` / `X-Org-Id` — a cross-tenant forge
    /// the gateway strips anyway.
    fn request(&self, req: ureq::Request) -> ureq::Request {
        let r = req
            .set("Content-Type", "application/json")
            .set("X-Project-Id", &self.project);
        if self.key.is_empty() {
            r
        } else {
            r.set("Authorization", &format!("Bearer {}", self.key))
        }
    }

    fn agent(&self) -> ureq::Agent {
        ureq::AgentBuilder::new().timeout(self.timeout).build()
    }

    fn post<B: Serialize, T: for<'de> Deserialize<'de>>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T, Error> {
        let url = format!("{}{}", self.base, path);
        let resp = self
            .request(self.agent().post(&url))
            .send_json(serde_json::to_value(body).map_err(|e| Error::Decode(e.to_string()))?);
        self.finish("POST", path, resp)
    }

    fn get<T: for<'de> Deserialize<'de>>(&self, path: &str) -> Result<T, Error> {
        let url = format!("{}{}", self.base, path);
        let resp = self.request(self.agent().get(&url)).call();
        self.finish("GET", path, resp)
    }

    fn finish<T: for<'de> Deserialize<'de>>(
        &self,
        method: &'static str,
        path: &str,
        resp: Result<ureq::Response, ureq::Error>,
    ) -> Result<T, Error> {
        let response = match resp {
            Ok(r) => r,
            Err(ureq::Error::Status(status, r)) => {
                let msg = r.into_string().unwrap_or_default();
                return Err(Error::Server {
                    method,
                    path: path.to_string(),
                    status,
                    message: server_error(&msg),
                });
            }
            Err(e) => return Err(Error::Transport(e.to_string())),
        };
        let mut buf = String::new();
        use std::io::Read;
        response
            .into_reader()
            .take((MAX_RESPONSE + 1) as u64)
            .read_to_string(&mut buf)
            .map_err(|e| Error::Transport(e.to_string()))?;
        if buf.len() > MAX_RESPONSE {
            return Err(Error::ResponseTooLarge { path: path.to_string(), cap: MAX_RESPONSE });
        }
        if buf.trim().is_empty() {
            buf.push_str("null");
        }
        serde_json::from_str(&buf).map_err(|e| Error::Decode(e.to_string()))
    }
}

/// Pulls the server's message out of an error body, falling back to the raw text.
fn server_error(body: &str) -> String {
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(body) {
        for k in ["error", "message", "msg"] {
            if let Some(s) = v.get(k).and_then(|x| x.as_str()) {
                return s.to_string();
            }
        }
    }
    body.chars().take(400).collect()
}

fn urlencode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}

/// Builds an experiment before it is opened.
pub struct ExperimentBuilder<'a> {
    client: &'a Research,
    kind: String,
    subject: String,
    task: String,
    metric: String,
    n_total: i64,
    note: String,
    hypothesis: String,
    predict: String,
}

impl<'a> ExperimentBuilder<'a> {
    /// Names the headline number's unit (`accuracy` | `tok/s` | `ratio` | `net-lift` | …).
    pub fn metric(mut self, s: impl Into<String>) -> Self {
        self.metric = s.into();
        self
    }
    /// The denominator this run is measured against.
    pub fn total(mut self, n: i64) -> Self {
        self.n_total = n;
        self
    }
    /// A one-shot note carried with the record.
    pub fn note(mut self, s: impl Into<String>) -> Self {
        self.note = s.into();
        self
    }
    /// The falsifiable claim under test.
    pub fn hypothesis(mut self, s: impl Into<String>) -> Self {
        self.hypothesis = s.into();
        self
    }
    /// What the hypothesis predicts will be observed.
    pub fn predict(mut self, s: impl Into<String>) -> Self {
        self.predict = s.into();
        self
    }

    /// Opens the experiment and posts it in-flight as `running`.
    pub fn start(self) -> Result<Experiment<'a>, Error> {
        let vcs = Vcs::detect(&self.client.repo);
        let libs: BTreeMap<String, String> = self
            .client
            .libs
            .iter()
            .filter_map(|l| lib_version(l).map(|v| (l.clone(), v)))
            .collect();
        let mut e = Experiment {
            client: self.client,
            id: format!("{}:{}:{}", self.kind, self.subject, self.task),
            kind: self.kind,
            subject: self.subject,
            task: self.task,
            metric: self.metric,
            n_total: self.n_total,
            n_ok: 0,
            n_done: 0,
            note: self.note,
            hypothesis: self.hypothesis,
            predict: self.predict,
            because: String::new(),
            log: Vec::new(),
            vcs,
            libs,
            host: Host::detect(),
            attempts: Vec::new(),
            sealed: false,
        };
        e.post("running", 0.0)?;
        Ok(e)
    }
}

/// A handle to one run. Not `Sync`: an experiment is a single narrative.
pub struct Experiment<'a> {
    client: &'a Research,
    id: String,
    kind: String,
    subject: String,
    task: String,
    metric: String,
    n_total: i64,
    n_ok: i64,
    n_done: i64,
    note: String,
    hypothesis: String,
    predict: String,
    because: String,
    log: Vec<String>,
    vcs: Vcs,
    libs: BTreeMap<String, String>,
    host: Host,
    attempts: Vec<AttemptPayload>,
    sealed: bool,
}

impl<'a> Experiment<'a> {
    /// The stable id this run upserts: `kind:subject:task`.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Files one measured attempt. Idempotent server-side on `(benchmark, item, model)`.
    pub fn record(
        &mut self,
        item: impl Into<String>,
        model: impl Into<String>,
        attempt: Attempt,
    ) -> Result<(), Error> {
        let a = attempt.normalized();
        if a.counts() {
            self.n_done += 1;
            if a.correct {
                self.n_ok += 1;
            }
        }
        self.attempts.push(AttemptPayload {
            benchmark: self.id.clone(),
            item: item.into(),
            model: model.into(),
            answer: a.answer,
            correct: a.correct,
            response: a.response,
            gold: a.gold,
            source: a.source,
            status: a.status,
        });
        Ok(())
    }

    /// Appends to the running narrative.
    pub fn log(&mut self, line: impl Into<String>) -> &mut Self {
        self.log.push(line.into());
        self
    }

    /// The accuracy implied by the attempts recorded so far.
    pub fn accuracy(&self) -> f64 {
        if self.n_done == 0 {
            0.0
        } else {
            self.n_ok as f64 / self.n_done as f64
        }
    }

    /// Seals the run with a verdict, the reasoning, and the headline number.
    pub fn conclude(
        &mut self,
        verdict: Verdict,
        because: impl Into<String>,
        value: f64,
    ) -> Result<(), Error> {
        self.because = because.into();
        self.post_verdict(Some(verdict), value)
    }

    /// Seals the run without an explicit verdict.
    ///
    /// A run that stated a hypothesis but never concluded is recorded `Inconclusive` rather
    /// than silently complete — an untested claim is not a proven one.
    pub fn finish(&mut self, value: f64) -> Result<(), Error> {
        let v = if self.hypothesis.is_empty() { None } else { Some(Verdict::Inconclusive) };
        self.post_verdict(v, value)
    }

    /// Files a diary artifact (a plot, a report). The server content-addresses it by sha256.
    pub fn artifact(&self, kind: &str, bytes: &[u8]) -> Result<(), Error> {
        self.client.artifact(ArtifactPayload {
            content: b64(bytes),
            sha256: sha256_hex(bytes),
            kind: kind.to_string(),
            run_id: self.id.clone(),
            git_sha: self.vcs.sha.clone(),
            git_branch: self.vcs.branch.clone(),
            git_dirty: self.vcs.dirty,
            lib_versions: self.libs.clone(),
        })
    }

    fn post_verdict(&mut self, verdict: Option<Verdict>, value: f64) -> Result<(), Error> {
        if let Some(v) = verdict {
            self.log.push(format!("verdict: {}", v.as_str()));
        }
        let v = verdict.map(|v| v.as_str().to_string()).unwrap_or_default();
        self.sealed = true;
        self.post_with("complete", value, &v)
    }

    fn post(&mut self, status: &str, value: f64) -> Result<(), Error> {
        self.post_with(status, value, "")
    }

    fn post_with(&mut self, status: &str, value: f64, verdict: &str) -> Result<(), Error> {
        let exp = ExpPayload {
            id: self.id.clone(),
            kind: self.kind.clone(),
            subject: self.subject.clone(),
            task: self.task.clone(),
            metric: self.metric.clone(),
            value,
            n: self.n_done,
            n_total: if self.n_total > 0 { self.n_total } else { self.n_done },
            status: status.to_string(),
            git_sha: self.vcs.sha.clone(),
            git_branch: self.vcs.branch.clone(),
            git_dirty: self.vcs.dirty,
            lib_versions: self.libs.clone(),
            meta: Meta {
                doc: String::new(),
                commits: Vec::new(),
                note: std::mem::take(&mut self.note),
                host: self.host.clone(),
                hypothesis: self.hypothesis.clone(),
                predict: self.predict.clone(),
                verdict: verdict.to_string(),
                because: self.because.clone(),
                log: self.log.clone(),
            },
        };
        let atts = std::mem::take(&mut self.attempts);
        self.client.ingest(vec![exp], atts)
    }
}

/// One stored experiment version returned by [`Research::query`].
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Run {
    #[serde(default)]
    pub project: String,
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub subject: String,
    #[serde(default)]
    pub task: String,
    #[serde(default)]
    pub metric: String,
    #[serde(default)]
    pub value: f64,
    #[serde(default)]
    pub n: i64,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub git_sha: String,
    #[serde(default)]
    pub meta: serde_json::Value,
}

/// Headline aggregate plus per-kind totals.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Totals {
    #[serde(default)]
    pub experiments: i64,
    #[serde(default)]
    pub attempts: i64,
    #[serde(default)]
    pub cost_usd: f64,
    #[serde(default)]
    pub kinds: Vec<KindTotal>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct KindTotal {
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub experiments: i64,
    #[serde(default)]
    pub attempts: i64,
}

/// Visibility/consent for a stable id.
#[derive(Debug, Clone, Default, Serialize)]
pub struct GrantRequest {
    pub id: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub project: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub visibility: String,
    pub trainable: bool,
    pub publishable: bool,
}

#[derive(Debug, Deserialize)]
struct GrantResult {
    #[serde(default)]
    updated: i64,
}

// --- small self-contained encodings, so the crate stays dependency-light ---

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn b64(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for c in data.chunks(3) {
        let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(B64[(n >> 18 & 63) as usize] as char);
        out.push(B64[(n >> 12 & 63) as usize] as char);
        out.push(if c.len() > 1 { B64[(n >> 6 & 63) as usize] as char } else { '=' });
        out.push(if c.len() > 2 { B64[(n & 63) as usize] as char } else { '=' });
    }
    out
}

fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bits = (data.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bits.to_be_bytes());
    for block in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (i, v) in [a, b, c, d, e, f, g, hh].iter().enumerate() {
            h[i] = h[i].wrapping_add(*v);
        }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inert() -> Research {
        Research::new(Config { base: "http://127.0.0.1:1".into(), ..Default::default() })
    }

    #[test]
    fn an_unconfigured_client_is_inert_not_broken() {
        // The property that makes it safe on a production path: with no key, every verb
        // returns Ok without touching the network. The base points at a closed port, so a
        // real request would error — proving these never dialled.
        let c = inert();
        assert!(!c.enabled());
        assert!(c.query("p", "k").unwrap().is_empty());
        assert_eq!(c.totals("p").unwrap().experiments, 0);
        assert_eq!(c.grant(GrantRequest::default()).unwrap(), 0);
        let mut e = c.experiment("kernel-perf", "gfx1151", "q4k").start().unwrap();
        e.record("n=4096", "hand", Attempt::ok("1.0")).unwrap();
        e.conclude(Verdict::Proven, "because", 1.0).unwrap();
    }

    #[test]
    fn accuracy_excludes_faulted_arms() {
        let c = inert();
        let mut e = c.experiment("benchmark", "s", "t").start().unwrap();
        e.record("a", "m", Attempt::ok("x")).unwrap();
        e.record("b", "m", Attempt::wrong("y", "x")).unwrap();
        e.record("c", "m", Attempt::faulted("oom")).unwrap();
        assert_eq!(e.accuracy(), 0.5, "2 counted, 1 correct; the faulted arm is excluded");
    }

    #[test]
    fn id_is_the_cross_language_key() {
        let c = inert();
        let e = c.experiment("kernel-perf", "gfx1151", "q4k-decode").start().unwrap();
        assert_eq!(e.id(), "kernel-perf:gfx1151:q4k-decode");
    }

    #[test]
    fn sha256_matches_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn base64_matches_known_vectors() {
        assert_eq!(b64(b""), "");
        assert_eq!(b64(b"f"), "Zg==");
        assert_eq!(b64(b"fo"), "Zm8=");
        assert_eq!(b64(b"foo"), "Zm9v");
        assert_eq!(b64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn marketing_and_kernels_share_one_plane() {
        // kind is an open string: the same call shape records an ad test and a kernel A/B.
        let c = inert();
        let a = c.experiment("marketing-experiment", "launch-email", "subject-line").start().unwrap();
        let b = c.experiment("kernel-perf", "gfx1151", "q4k-decode").start().unwrap();
        assert_eq!(a.id(), "marketing-experiment:launch-email:subject-line");
        assert_eq!(b.id(), "kernel-perf:gfx1151:q4k-decode");
    }
}
