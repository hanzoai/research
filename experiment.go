// Copyright © 2026 Hanzo AI. MIT License.

package research

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"math"
)

// Verdict is the epistemic outcome of a falsifiable experiment — distinct from the
// execution status. A refutation is a first-class, durable result, recorded as clearly as
// a proof; that is the whole point of an evidence plane. The defined type means a bare
// string will not compile as a verdict, and an out-of-set value is rejected at Conclude.
type Verdict string

const (
	Proven       Verdict = "proven"
	Refuted      Verdict = "refuted"
	Inconclusive Verdict = "inconclusive"
)

func (v Verdict) valid() bool {
	switch v {
	case Proven, Refuted, Inconclusive:
		return true
	}
	return false
}

// Result is one measured attempt's outcome. Source defaults to "hanzo-measured" and
// Status to "complete" when left blank; a "faulted" or "failed" status retains a negative
// result without counting toward the running accuracy.
type Result struct {
	Answer   string
	Correct  bool
	Response string
	Gold     string
	Source   string
	Status   string
}

// meta carries the scientific frame plus the self-documenting narrative. Its field order
// and json names are the cross-language contract — identical to the Python and TypeScript
// producers — so every language emits structurally identical records into the one store.
type meta struct {
	Doc        string   `json:"doc"`
	Commits    []string `json:"commits"`
	Note       string   `json:"note"`
	Host       box      `json:"host"`
	Hypothesis string   `json:"hypothesis"`
	Predict    string   `json:"predict"`
	Verdict    string   `json:"verdict"`
	Because    string   `json:"because"`
	Log        []string `json:"log"`
}

// expPayload is one experiment version on the wire, in the exact field order the Python
// producer emits. project, ts, and cost are server-stamped and so are absent here.
type expPayload struct {
	ID          string            `json:"id"`
	Kind        string            `json:"kind"`
	Subject     string            `json:"subject"`
	Task        string            `json:"task"`
	Metric      string            `json:"metric"`
	Value       float64           `json:"value"`
	N           int               `json:"n"`
	NTotal      int               `json:"n_total"`
	Status      string            `json:"status"`
	GitSHA      string            `json:"git_sha"`
	GitBranch   string            `json:"git_branch"`
	GitDirty    bool              `json:"git_dirty"`
	LibVersions map[string]string `json:"lib_versions"`
	Meta        meta              `json:"meta"`
}

// attemptPayload is one measured attempt on the wire, in the Python producer's field
// order.
type attemptPayload struct {
	Benchmark string `json:"benchmark"`
	Item      string `json:"item"`
	Model     string `json:"model"`
	Answer    string `json:"answer"`
	Correct   bool   `json:"correct"`
	Response  string `json:"response"`
	Gold      string `json:"gold"`
	Source    string `json:"source"`
	Status    string `json:"status"`
}

// artifactPayload is one diary artifact on the wire. The bytes are base64 in content; the
// server content-addresses them by sha256.
type artifactPayload struct {
	Content     string            `json:"content"`
	SHA256      string            `json:"sha256"`
	Kind        string            `json:"kind"`
	RunID       string            `json:"run_id"`
	GitSHA      string            `json:"git_sha"`
	GitBranch   string            `json:"git_branch"`
	GitDirty    bool              `json:"git_dirty"`
	LibVersions map[string]string `json:"lib_versions"`
}

// Experiment is a handle to one run (kind:subject:task). Record files attempts; Log
// appends to the running narrative; Snapshot and Report file diary artifacts; Conclude and
// Finish seal the run with its verdict, headline number, and the auto-captured provenance.
// Not safe for concurrent use by multiple goroutines.
type Experiment struct {
	c       *Research
	id      string
	kind    string
	subject string
	task    string
	metric  string
	nTotal  int

	// running accuracy, for the auto-computed value at Finish.
	nOK   int
	nDone int

	// the falsifiable frame and narrative that travel into meta.
	note       string
	hypothesis string
	predict    string
	verdict    Verdict
	because    string
	log        []string

	// zero-config provenance, captured at start.
	git     vcs
	doc     string
	commits []string
	libs    map[string]string
	host    box
}

// Option configures an Experiment at creation.
type Option func(*Experiment)

// Metric names the headline number's unit (accuracy | tok/s | ratio | net-lift | …).
func Metric(s string) Option { return func(e *Experiment) { e.metric = s } }

// Total sets the item target (n_total), so the board shows done versus remaining.
func Total(n int) Option { return func(e *Experiment) { e.nTotal = n } }

// Note sets the one-shot headline note (meta.note). The running trail is Log.
func Note(s string) Option { return func(e *Experiment) { e.note = s } }

// Hypothesis states the claim under test — the frame a later Conclude proves or refutes.
func Hypothesis(s string) Option { return func(e *Experiment) { e.hypothesis = s } }

// Predict states the observation that would confirm the hypothesis.
func Predict(s string) Option { return func(e *Experiment) { e.predict = s } }

// Log appends a line to the running narrative — the "what I saw / thought" trail that
// travels with the run into meta.log. Chainable.
func (e *Experiment) Log(text string) *Experiment {
	e.log = append(e.log, text)
	return e
}

// Record files one attempt. Idempotent server-side by (project, benchmark, item, model),
// so replaying a corpus records it exactly once.
func (e *Experiment) Record(item, model string, r Result) (IngestResult, error) {
	source := r.Source
	if source == "" {
		source = "hanzo-measured"
	}
	status := r.Status
	if status == "" {
		status = "complete"
	}
	if status != "faulted" && status != "failed" {
		e.nDone++
		if r.Correct {
			e.nOK++
		}
	}
	return e.c.ingest(nil, []attemptPayload{{
		Benchmark: e.task, Item: item, Model: model,
		Answer: r.Answer, Correct: r.Correct, Response: r.Response,
		Gold: r.Gold, Source: source, Status: status,
	}})
}

// Conclude seals the experiment with its verdict ∈ {Proven, Refuted, Inconclusive} and the
// reasoning that earns it, then finishes the run. An out-of-set verdict is rejected. A
// refutation is recorded as durably as a proof. value is the headline number, or omitted
// to compute it from the recorded attempts.
func (e *Experiment) Conclude(v Verdict, because string, value ...float64) (IngestResult, error) {
	if !v.valid() {
		return IngestResult{}, fmt.Errorf("research: verdict must be %s, %s, or %s, got %q",
			Proven, Refuted, Inconclusive, v)
	}
	e.verdict = v
	e.because = because
	return e.Finish(value...)
}

// Finish seals the run: posts the headline value (computed from the recorded attempts when
// omitted) with status complete. A stated hypothesis with no verdict defaults to
// Inconclusive — a finished run never silently reads as a proof.
func (e *Experiment) Finish(value ...float64) (IngestResult, error) {
	v := 0.0
	switch {
	case len(value) > 0:
		v = value[0]
	case e.nDone > 0:
		v = math.Round(100.0*float64(e.nOK)/float64(e.nDone)*100) / 100
	}
	if e.hypothesis != "" && e.verdict == "" {
		e.verdict = Inconclusive
	}
	return e.post("complete", v)
}

// Snapshot files a board-snapshot artifact (PNG bytes). The server content-addresses the
// bytes by sha256; the client hash travels only so the server can reject a mismatch.
func (e *Experiment) Snapshot(data []byte) (ArtifactResult, error) {
	return e.artifact("snapshot", data)
}

// Report files a generated-report artifact (HTML or Markdown bytes).
func (e *Experiment) Report(data []byte) (ArtifactResult, error) {
	return e.artifact("report", data)
}

func (e *Experiment) artifact(kind string, raw []byte) (ArtifactResult, error) {
	sum := sha256.Sum256(raw)
	return e.c.artifact(artifactPayload{
		Content:     base64.StdEncoding.EncodeToString(raw),
		SHA256:      hex.EncodeToString(sum[:]),
		Kind:        kind,
		RunID:       e.id,
		GitSHA:      e.git.SHA,
		GitBranch:   e.git.Branch,
		GitDirty:    e.git.Dirty,
		LibVersions: e.libs,
	})
}

// post files the experiment version at the given status and value.
func (e *Experiment) post(status string, value float64) (IngestResult, error) {
	return e.c.ingest([]expPayload{{
		ID: e.id, Kind: e.kind, Subject: e.subject, Task: e.task, Metric: e.metric,
		Value: value, N: e.nDone, NTotal: e.nTotal, Status: status,
		GitSHA: e.git.SHA, GitBranch: e.git.Branch, GitDirty: e.git.Dirty,
		LibVersions: e.libs,
		Meta: meta{
			Doc: e.doc, Commits: e.commits, Note: e.note, Host: e.host,
			Hypothesis: e.hypothesis, Predict: e.predict,
			Verdict: string(e.verdict), Because: e.because, Log: e.log,
		},
	}}, nil)
}
