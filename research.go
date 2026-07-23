// Copyright © 2026 Hanzo AI. MIT License.

package research

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"
)

const (
	defaultBase    = "https://api.hanzo.ai"
	defaultProject = "default"
	defaultTimeout = 120 * time.Second
	narrativeDepth = 10      // commits to record when the last run's sha is unknown
	maxResponse    = 8 << 20 // response-body cap (8 MiB): a hostile server cannot OOM the client
)

// Config configures a client. The zero value is valid: Base defaults to api.hanzo.ai (or
// $RESEARCH_BASE), Key to $HANZO_API_KEY, Project to $RESEARCH_PROJECT or "default", and
// Repo to the git toplevel of the process. Libs are module-path prefixes to fold into the
// lib-version provenance beyond the Go toolchain and main module.
type Config struct {
	Base    string
	Key     string
	Project string
	Repo    string
	Libs    []string
	Timeout time.Duration
}

// Research is a configured client: base URL, per-org key, project sub-scope, and the
// auto-detected provenance repo. Safe for concurrent use by multiple goroutines.
type Research struct {
	base    string
	key     string
	project string
	repo    string
	libs    []string
	http    *http.Client
}

// New builds a client from cfg, filling every unset field from the environment or the
// documented default. The key is read from $HANZO_API_KEY when unset — sourced from KMS
// through the process environment, never hardcoded.
func New(cfg Config) *Research {
	repo := cfg.Repo
	if repo == "" {
		repo = findRepo("")
	}
	timeout := cfg.Timeout
	if timeout == 0 {
		timeout = defaultTimeout
	}
	return &Research{
		base:    strings.TrimRight(firstNonEmpty(cfg.Base, os.Getenv("RESEARCH_BASE"), defaultBase), "/"),
		key:     firstNonEmpty(cfg.Key, os.Getenv("HANZO_API_KEY")),
		project: firstNonEmpty(cfg.Project, os.Getenv("RESEARCH_PROJECT"), defaultProject),
		repo:    repo,
		libs:    cfg.Libs,
		http: &http.Client{
			Timeout: timeout,
			// Never follow redirects: a /v1/research endpoint has no legitimate redirect, and
			// refusing them denies a hostile 30x any chance to bounce the request (and its
			// Bearer key) to another host or an SSRF target. Return the 30x as-is.
			CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
		},
	}
}

// String renders the client for logs and debug dumps WITHOUT its key. The api key is a
// secret and must never reach a log line, so implementing Stringer + GoStringer makes every
// fmt verb (%v, %+v, %#v, %s) redact it — closing the plaintext-key leak a struct dump
// would otherwise open. Mirrors the Rust SDK's Debug redaction. Value receiver, so a
// *Research and a Research value are both covered.
func (c Research) String() string {
	return fmt.Sprintf("research.Research{base:%q, project:%q, repo:%q, key:<redacted>}",
		c.base, c.project, c.repo)
}

// GoString redacts the key under the %#v (Go-syntax) verb, which does not consult String.
func (c Research) GoString() string { return c.String() }

// Experiment gets or creates the handle for (kind, subject, task) and posts it in-flight
// so the ops board sees it immediately. kind is an OPEN string — benchmark, kernel-perf,
// training, ablation, policy-eval AND marketing-experiment, ad-test, pricing-test,
// growth-experiment, … — a marketing A/B test records identically to a kernel A/B. Options
// set the metric, item target, one-shot note, and the falsifiable frame (Hypothesis,
// Predict). Provenance — git sha/branch/dirty, the commit narrative since this experiment's
// last recorded run, lib versions, host, and the calling code site — is auto-captured; the
// caller supplies none.
func (c *Research) Experiment(kind, subject, task string, opts ...Option) (*Experiment, error) {
	e := &Experiment{
		c:       c,
		id:      kind + ":" + subject + ":" + task,
		kind:    kind,
		subject: subject,
		task:    task,
		metric:  "accuracy",
		log:     []string{},
	}
	for _, opt := range opts {
		opt(e)
	}
	// Zero-config provenance: git state, the calling code site, lib versions, host, and
	// the commit narrative SINCE this experiment's last recorded run (best-effort — an
	// unavailable read falls back to the recent window, never failing the experiment).
	e.git = gitState(c.repo)
	e.doc = callerDoc()
	e.libs = libVersions(c.libs)
	e.host = host()
	e.commits = commitNarrative(c.repo, c.lastRunSHA(e.id), narrativeDepth)
	// Post as in-flight so the board sees the run immediately.
	if _, err := e.post("running", 0.0); err != nil {
		return nil, err
	}
	return e, nil
}

// Query lists canonical experiments (the latest answered version per stable id), narrowed
// to a project (defaulting to the client's) and, when set, a kind.
func (c *Research) Query(project, kind string) ([]Run, error) {
	if project == "" {
		project = c.project
	}
	q := url.Values{}
	if project != "" {
		q.Set("project", project)
	}
	if kind != "" {
		q.Set("kind", kind)
	}
	path := "/v1/research/experiments"
	if len(q) > 0 {
		path += "?" + q.Encode()
	}
	var out struct {
		Data []Run `json:"data"`
	}
	err := c.do(http.MethodGet, path, nil, &out)
	return out.Data, err
}

// Totals reads the headline aggregate (canonical plus retained) with a per-kind breakdown,
// optionally scoped to one project.
func (c *Research) Totals(project string) (Totals, error) {
	path := "/v1/research/totals"
	if project != "" {
		path += "?" + url.Values{"project": {project}}.Encode()
	}
	var out Totals
	err := c.do(http.MethodGet, path, nil, &out)
	return out, err
}

// Grant sets visibility/consent for a run (by ID) or an artifact (by SHA256) — the separate
// authorization an upload never implies (uploads are private by default). Returns the row
// count updated.
func (c *Research) Grant(g GrantRequest) (int, error) {
	var out struct {
		Updated int `json:"updated"`
	}
	err := c.do(http.MethodPost, "/v1/research/grants", g, &out)
	return out.Updated, err
}

// ingest POSTs a batch of experiments and attempts (idempotent by stable id server-side).
// Both arrays are always sent (empty, never null) to match the cross-language wire shape.
func (c *Research) ingest(exps []expPayload, atts []attemptPayload) (IngestResult, error) {
	if exps == nil {
		exps = []expPayload{}
	}
	if atts == nil {
		atts = []attemptPayload{}
	}
	var out IngestResult
	err := c.do(http.MethodPost, "/v1/research/experiments",
		ingestReq{Experiments: exps, Attempts: atts}, &out)
	return out, err
}

// artifact POSTs one diary artifact (idempotent by sha256 content hash server-side).
func (c *Research) artifact(a artifactPayload) (ArtifactResult, error) {
	var out ArtifactResult
	err := c.do(http.MethodPost, "/v1/research/artifacts", a, &out)
	return out, err
}

// lastRunSHA is the git sha of this experiment's last recorded run, for the since-
// narrative. Best-effort: any read error yields "" (the recent-window fallback).
func (c *Research) lastRunSHA(id string) string {
	runs, err := c.Query("", "")
	if err != nil {
		return ""
	}
	for _, r := range runs {
		if r.ID == id {
			return r.GitSHA
		}
	}
	return ""
}

// headers builds the request headers. Auth is ONLY the per-org key (the gateway mints the
// validated principal and project scope from it); the client never sends X-User-Id /
// X-Org-Id — a cross-tenant forge the gateway strips anyway.
func (c *Research) headers() http.Header {
	h := http.Header{}
	h.Set("Content-Type", "application/json")
	h.Set("X-Project-Id", c.project)
	if c.key != "" {
		h.Set("Authorization", "Bearer "+c.key)
	}
	return h
}

// do performs one JSON request and decodes the JSON response into out when non-nil. A
// non-2xx status is returned as an error carrying the server's message.
func (c *Research) do(method, path string, body, out any) error {
	var r io.Reader
	if body != nil {
		b, err := encode(body)
		if err != nil {
			return err
		}
		r = bytes.NewReader(b)
	}
	req, err := http.NewRequest(method, c.base+path, r)
	if err != nil {
		return err
	}
	req.Header = c.headers()
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	// Bound the read: a hostile or broken server cannot stream an unbounded body to OOM the
	// client. Read one byte past the cap so an over-limit body is detected, not silently cut.
	data, _ := io.ReadAll(io.LimitReader(resp.Body, maxResponse+1))
	if len(data) > maxResponse {
		return fmt.Errorf("research: %s %s: response exceeds %d-byte cap", method, path, maxResponse)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("research: %s %s: %s: %s", method, path, resp.Status, serverError(data))
	}
	if out != nil && len(data) > 0 {
		return json.Unmarshal(data, out)
	}
	return nil
}

// ingestReq is one upload batch: experiments and their attempts as two flat arrays, in the
// Python producer's key order.
type ingestReq struct {
	Experiments []expPayload     `json:"experiments"`
	Attempts    []attemptPayload `json:"attempts"`
}

// Run is one stored experiment version returned by Query — the canonical (latest answered)
// view. Meta is the raw provenance object; git and lib fields populate as the server
// surfaces them.
type Run struct {
	Project     string            `json:"project"`
	ID          string            `json:"id"`
	Kind        string            `json:"kind"`
	Subject     string            `json:"subject"`
	Task        string            `json:"task"`
	Metric      string            `json:"metric"`
	Value       float64           `json:"value"`
	N           int               `json:"n"`
	NTotal      int               `json:"n_total"`
	CostUSD     float64           `json:"cost_usd"`
	Status      string            `json:"status"`
	Meta        json.RawMessage   `json:"meta,omitempty"`
	TS          int64             `json:"ts"`
	GitSHA      string            `json:"git_sha,omitempty"`
	GitBranch   string            `json:"git_branch,omitempty"`
	GitDirty    bool              `json:"git_dirty,omitempty"`
	LibVersions map[string]string `json:"lib_versions,omitempty"`
}

// Totals is the headline aggregate for an org or one project.
type Totals struct {
	Project             string      `json:"project"`
	Projects            int         `json:"projects"`
	Experiments         int         `json:"experiments"`
	ExperimentsRetained int         `json:"experiments_retained"`
	Attempts            int         `json:"attempts"`
	AttemptsRetained    int         `json:"attempts_retained"`
	Models              int         `json:"models"`
	Benchmarks          int         `json:"benchmarks"`
	CostUSD             float64     `json:"cost_usd"`
	ByKind              []KindTotal `json:"by_kind"`
}

// KindTotal is one discriminator's slice of the totals.
type KindTotal struct {
	Kind        string  `json:"kind"`
	Experiments int     `json:"experiments"`
	CostUSD     float64 `json:"cost_usd"`
}

// IngestResult is the count summary returned by an ingest.
type IngestResult struct {
	Project             string `json:"project"`
	ExperimentsIngested int    `json:"experiments_ingested"`
	AttemptsIngested    int    `json:"attempts_ingested"`
	ExperimentsTotal    int    `json:"experiments_total"`
	AttemptsTotal       int    `json:"attempts_total"`
	RolledUp            bool   `json:"rolled_up"`
}

// ArtifactResult is the outcome of recording a diary artifact.
type ArtifactResult struct {
	SHA256   string `json:"sha256"`
	Created  bool   `json:"created"`
	RolledUp bool   `json:"rolled_up"`
}

// GrantRequest sets visibility/consent for a run (ID) or an artifact (SHA256). A nil
// pointer field is left unchanged server-side.
type GrantRequest struct {
	Project     string  `json:"project,omitempty"`
	ID          string  `json:"id,omitempty"`
	SHA256      string  `json:"sha256,omitempty"`
	Visibility  *string `json:"visibility,omitempty"`
	Trainable   *bool   `json:"trainable,omitempty"`
	Publishable *bool   `json:"publishable,omitempty"`
}

// encode marshals v as compact JSON with HTML escaping OFF, so <, >, and & in notes and
// commit messages travel literally rather than as escaped sequences. The wire bytes are
// compact (Python/C++ emit spaced json.dumps) — immaterial to the store: the server keys
// each record on (project, id) and parses the JSON, so every language upserts the same row.
func encode(v any) ([]byte, error) {
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimRight(b.Bytes(), "\n"), nil
}

// serverError pulls the {error} message from a failed response, else the raw body.
func serverError(data []byte) string {
	var e struct {
		Error string `json:"error"`
	}
	if json.Unmarshal(data, &e) == nil && e.Error != "" {
		return e.Error
	}
	return strings.TrimSpace(string(data))
}

// firstNonEmpty returns the first non-empty string, or "".
func firstNonEmpty(v ...string) string {
	for _, s := range v {
		if s != "" {
			return s
		}
	}
	return ""
}
