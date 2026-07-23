// Copyright © 2026 Hanzo AI. MIT License.

package research

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
)

// metaOrder is the cross-language contract: the meta keys, in the exact order the Python
// and TypeScript producers emit them.
var metaOrder = []string{"doc", "commits", "note", "host", "hypothesis", "predict", "verdict", "because", "log"}

// expOrder is the experiment record's field order on the wire, shared by every kind.
var expOrder = []string{"id", "kind", "subject", "task", "metric", "value", "n", "n_total", "status", "git_sha", "git_branch", "git_dirty", "lib_versions", "meta"}

// TestMetaMatchesPythonBytes proves a Go-emitted meta is the SAME JSON document as the
// Python producer's json.dumps for identical inputs — same keys, same order, and (with
// HTML escaping off) the same literal &, <, > that commit messages and notes contain.
func TestMetaMatchesPythonBytes(t *testing.T) {
	m := meta{
		Doc:        "github.com/hanzoai/marketing/abtest.Checkout",
		Commits:    []string{"ship green CTA", "measure lift & p<0.05"},
		Note:       "blue vs green; lift & p<0.05",
		Host:       box{Hostname: "evo", Platform: "Linux"},
		Hypothesis: "green CTA lifts checkout conversion",
		Predict:    "green - blue >= 2% at p<0.05",
		Verdict:    "refuted",
		Because:    "green -0.4% vs blue, p=0.03",
		Log:        []string{"n=48120", "two-week holdout"},
	}
	got, err := encode(m)
	if err != nil {
		t.Fatal(err)
	}

	// The exact compact bytes our encoder emits, which the Python producer's json.dumps
	// reduces to once whitespace (JSON-insignificant) is removed. json.dumps escapes
	// neither & nor <, and our encoder sets EscapeHTML=false to match.
	want := `{"doc":"github.com/hanzoai/marketing/abtest.Checkout","commits":["ship green CTA","measure lift & p<0.05"],"note":"blue vs green; lift & p<0.05","host":{"hostname":"evo","platform":"Linux"},"hypothesis":"green CTA lifts checkout conversion","predict":"green - blue >= 2% at p<0.05","verdict":"refuted","because":"green -0.4% vs blue, p=0.03","log":["n=48120","two-week holdout"]}`
	if string(got) != want {
		t.Fatalf("meta bytes differ from the cross-language contract:\n got: %s\nwant: %s", got, want)
	}

	// And against Python's ACTUAL json.dumps output (spaced): the same document.
	pySpaced := `{"doc": "github.com/hanzoai/marketing/abtest.Checkout", "commits": ["ship green CTA", "measure lift & p<0.05"], "note": "blue vs green; lift & p<0.05", "host": {"hostname": "evo", "platform": "Linux"}, "hypothesis": "green CTA lifts checkout conversion", "predict": "green - blue >= 2% at p<0.05", "verdict": "refuted", "because": "green -0.4% vs blue, p=0.03", "log": ["n=48120", "two-week holdout"]}`
	if canon(t, got) != canon(t, []byte(pySpaced)) {
		t.Fatalf("go meta and python json.dumps are not the same document:\n go: %s\n py: %s", canon(t, got), canon(t, []byte(pySpaced)))
	}

	// The definitive cross-language check: Go's canonical form (map keys sorted by the
	// json package, HTML-escaping off) byte-equals REAL CPython json.dumps(meta,
	// sort_keys=True, separators=(",",":")) — verified against the stdlib json module.
	pyCanon := `{"because":"green -0.4% vs blue, p=0.03","commits":["ship green CTA","measure lift & p<0.05"],"doc":"github.com/hanzoai/marketing/abtest.Checkout","host":{"hostname":"evo","platform":"Linux"},"hypothesis":"green CTA lifts checkout conversion","log":["n=48120","two-week holdout"],"note":"blue vs green; lift & p<0.05","predict":"green - blue >= 2% at p<0.05","verdict":"refuted"}`
	var parsed map[string]any
	if err := json.Unmarshal(got, &parsed); err != nil {
		t.Fatal(err)
	}
	goCanon, _ := encode(parsed)
	if string(goCanon) != pyCanon {
		t.Fatalf("go canonical != python canonical:\n go: %s\n py: %s", goCanon, pyCanon)
	}

	// The EscapeHTML=false is load-bearing: Go's default marshal HTML-escapes & and <,
	// diverging from the Python producer, which emits them literally. This documents why
	// encode exists — the default would break byte-compatibility on common note/commit text.
	def, _ := json.Marshal(m)
	if strings.ContainsRune(string(def), '&') {
		t.Fatal("expected Go's default marshal to escape & (HTML-escaping on)")
	}
	if !strings.ContainsRune(string(got), '&') {
		t.Fatal("encode must keep & literal (HTML-escaping off) to match the Python producer")
	}

	// Key order is exactly the contract.
	if k := topKeys(t, got); !reflect.DeepEqual(k, metaOrder) {
		t.Fatalf("meta key order = %v, want %v", k, metaOrder)
	}
}

// TestUniversalKinds runs a kernel-perf A/B and a marketing-experiment A/B end-to-end
// against a mock server and proves they emit byte-compatible records: identical field
// order at both the experiment and meta level, arrays never null, and a first-class
// verdict either way. A refutation is recorded as cleanly as a proof.
func TestUniversalKinds(t *testing.T) {
	m := newMock()
	defer m.Close()
	c := New(Config{Base: m.URL, Key: "hk-test-key", Project: "enso-bench"})

	// A kernel A/B, framed as a falsifiable test, concluded PROVEN.
	k, err := c.Experiment("kernel-perf", "matvec_q4k_f32_blk", "vulkan/6144x2048",
		Metric("ratio_vs_hand"),
		Total(3),
		Hypothesis("the DSL f32-direct matvec beats the hand kernel"),
		Predict("DSL/hand >= 1.0 cold in-engine at the dominant FFN shape"))
	if err != nil {
		t.Fatalf("kernel experiment: %v", err)
	}
	k.Record("6144x2048", "dsl-f32", Result{Answer: "1.022", Correct: true})
	k.Record("2048x2048", "dsl-f32", Result{Answer: "0.94", Correct: false})
	k.Log("cold in-engine A/B, evo gfx1151, quiet window, 3 runs, bit-exact 2.3e-6")
	if _, err := k.Conclude(Proven, "1.022x at 6144 rows (loses small shapes, gate >=4096)", 1.022); err != nil {
		t.Fatalf("kernel conclude: %v", err)
	}

	// A marketing A/B — SAME verbs, open kind — concluded REFUTED. Negative result, first
	// class.
	mk, err := c.Experiment("marketing-experiment", "checkout-cta", "blue-vs-green-button",
		Metric("net-lift"),
		Total(48120),
		Note("two-week holdout, US traffic only"),
		Hypothesis("the green CTA lifts checkout conversion"),
		Predict("green - blue >= 2% conversion at p<0.05"))
	if err != nil {
		t.Fatalf("marketing experiment: %v", err)
	}
	mk.Record("cohort-green", "variant-green", Result{Answer: "0.171", Correct: false})
	mk.Record("cohort-blue", "variant-blue", Result{Answer: "0.175", Correct: true})
	mk.Log("green -0.4% vs blue; p=0.03; blue wins")
	if _, err := mk.Conclude(Refuted, "green -0.4% vs blue, significant at p=0.03", -0.4); err != nil {
		t.Fatalf("marketing conclude: %v", err)
	}

	// Auth is the per-org key; project rides the header; no forged identity.
	if m.auth != "Bearer hk-test-key" {
		t.Fatalf("Authorization = %q, want Bearer hk-test-key", m.auth)
	}
	if m.project != "enso-bench" {
		t.Fatalf("X-Project-Id = %q, want enso-bench", m.project)
	}

	kExp := m.sealed(t, "kernel-perf")
	mExp := m.sealed(t, "marketing-experiment")

	// Both records carry the identical experiment field order — one shape, every kind.
	if got := topKeys(t, kExp); !reflect.DeepEqual(got, expOrder) {
		t.Fatalf("kernel experiment key order = %v, want %v", got, expOrder)
	}
	if got := topKeys(t, mExp); !reflect.DeepEqual(got, expOrder) {
		t.Fatalf("marketing experiment key order = %v, want %v", got, expOrder)
	}

	kMeta := metaOf(t, kExp)
	mMeta := metaOf(t, mExp)

	// Meta shape is identical across kinds, in the cross-language order.
	if got := topKeys(t, kMeta); !reflect.DeepEqual(got, metaOrder) {
		t.Fatalf("kernel meta key order = %v, want %v", got, metaOrder)
	}
	if !reflect.DeepEqual(topKeys(t, kMeta), topKeys(t, mMeta)) {
		t.Fatal("kernel and marketing meta shapes differ — universal kind broken")
	}

	// Verdicts are first-class, distinct, and durable.
	if v := field(t, kMeta, "verdict"); v != `"proven"` {
		t.Fatalf("kernel verdict = %s, want proven", v)
	}
	if v := field(t, mMeta, "verdict"); v != `"refuted"` {
		t.Fatalf("marketing verdict = %s, want refuted", v)
	}

	// commits and log are arrays, never null — so they serialize identically to the other
	// producers regardless of git state.
	for _, key := range []string{"commits", "log"} {
		for name, mt := range map[string]json.RawMessage{"kernel": kMeta, "marketing": mMeta} {
			if v := field(t, mt, key); !strings.HasPrefix(v, "[") {
				t.Fatalf("%s meta.%s = %s, want a JSON array", name, key, v)
			}
		}
	}

	// The running narrative traveled into meta.log.
	var kLog []string
	json.Unmarshal([]byte(field(t, kMeta, "log")), &kLog)
	if len(kLog) == 0 || !strings.Contains(kLog[0], "cold in-engine A/B") {
		t.Fatalf("kernel meta.log lost the narrative: %v", kLog)
	}
}

// TestVerdictValidation proves an out-of-set verdict is rejected and that a stated
// hypothesis with no verdict seals as Inconclusive, never a silent proof.
func TestVerdictValidation(t *testing.T) {
	m := newMock()
	defer m.Close()
	c := New(Config{Base: m.URL, Project: "p"})

	e, err := c.Experiment("ablation", "router", "top2-vs-top1", Hypothesis("top2 helps"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := e.Conclude(Verdict("maybe"), "unsure"); err == nil {
		t.Fatal("expected an invalid verdict to be rejected")
	}
	if _, err := e.Finish(); err != nil {
		t.Fatal(err)
	}
	sealed := m.sealed(t, "ablation")
	if v := field(t, metaOf(t, sealed), "verdict"); v != `"inconclusive"` {
		t.Fatalf("a stated hypothesis with no verdict sealed as %s, want inconclusive", v)
	}
}

// ── test helpers ────────────────────────────────────────────────────────────────────

// mock is a stand-in /v1/research server that records the ingest bodies it receives.
type mock struct {
	*httptest.Server
	mu          sync.Mutex
	experiments [][]byte
	auth        string
	project     string
}

func newMock() *mock {
	m := &mock{}
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/research/experiments", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			w.Write([]byte(`{"data":[],"total":0}`))
			return
		}
		body, _ := io.ReadAll(r.Body)
		m.mu.Lock()
		m.experiments = append(m.experiments, body)
		m.auth = r.Header.Get("Authorization")
		m.project = r.Header.Get("X-Project-Id")
		m.mu.Unlock()
		w.Write([]byte(`{"project":"p","experiments_ingested":1,"attempts_ingested":0,"experiments_total":1,"attempts_total":0,"rolled_up":true}`))
	})
	mux.HandleFunc("/v1/research/artifacts", func(w http.ResponseWriter, r *http.Request) {
		io.ReadAll(r.Body)
		w.Write([]byte(`{"sha256":"deadbeef","created":true,"rolled_up":true}`))
	})
	m.Server = httptest.NewServer(mux)
	return m
}

// sealed returns the raw bytes of the completed experiment record of the given kind.
func (m *mock) sealed(t *testing.T, kind string) json.RawMessage {
	t.Helper()
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, b := range m.experiments {
		var req struct {
			Experiments []json.RawMessage `json:"experiments"`
		}
		if json.Unmarshal(b, &req) != nil {
			continue
		}
		for _, er := range req.Experiments {
			var peek struct{ Kind, Status string }
			json.Unmarshal(er, &peek)
			if peek.Kind == kind && peek.Status == "complete" {
				return er
			}
		}
	}
	t.Fatalf("no sealed %q experiment captured", kind)
	return nil
}

// metaOf extracts the meta object from a raw experiment record.
func metaOf(t *testing.T, exp json.RawMessage) json.RawMessage {
	t.Helper()
	var e struct {
		Meta json.RawMessage `json:"meta"`
	}
	if err := json.Unmarshal(exp, &e); err != nil {
		t.Fatal(err)
	}
	return e.Meta
}

// field returns the raw JSON of one top-level key of obj.
func field(t *testing.T, obj json.RawMessage, key string) string {
	t.Helper()
	var mm map[string]json.RawMessage
	if err := json.Unmarshal(obj, &mm); err != nil {
		t.Fatal(err)
	}
	return string(mm[key])
}

// topKeys returns the top-level object keys of raw, in document order.
func topKeys(t *testing.T, raw json.RawMessage) []string {
	t.Helper()
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	if tok, err := dec.Token(); err != nil || tok != json.Delim('{') {
		t.Fatalf("not a JSON object: %v (%v)", tok, err)
	}
	var keys []string
	for dec.More() {
		tok, err := dec.Token()
		if err != nil {
			t.Fatal(err)
		}
		key, ok := tok.(string)
		if !ok {
			t.Fatalf("expected a string key, got %T", tok)
		}
		keys = append(keys, key)
		var skip json.RawMessage
		if err := dec.Decode(&skip); err != nil {
			t.Fatal(err)
		}
	}
	return keys
}

// canon reduces JSON to a whitespace- and order-insensitive canonical form for document
// equality.
func canon(t *testing.T, b []byte) string {
	t.Helper()
	var v any
	if err := json.Unmarshal(b, &v); err != nil {
		t.Fatalf("bad json: %v", err)
	}
	out, _ := json.Marshal(v)
	return string(out)
}
