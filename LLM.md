# research — Go SDK for /v1/research

The ONE way a Go service records and queries R&D evidence on the unified `/v1/research`
surface (HIP-0512). Stdlib-only (`net/http` + `encoding/json`), zero external modules, so
any service — cloud, tools, marketing/growth, ops — imports it without dragging a
dependency graph.

Mirrors the Python producer (`hanzo/python-sdk/pkg/hanzo-research`) verb-for-verb. Records
are semantically identical across languages: the server keys each experiment on `(project,
id = kind:subject:task)`, so every language upserts the same row regardless of JSON
serialization (Go emits compact bytes; Python/C++ emit spaced — immaterial to the store).

## Surface

```go
c := research.New(research.Config{Project: "enso-bench"})       // per-org key auth

exp, _ := c.Experiment(kind, subject, task, opts...)            // handle; posts in-flight
exp.Record(item, model, research.Result{...})                  // one attempt, idempotent
exp.Log(text)                                                  // append running narrative (chainable)
exp.Conclude(research.Proven, because, value)                  // verdict + seal complete
exp.Finish(value)                                              // seal; hypothesis w/o verdict -> Inconclusive
exp.Snapshot(pngBytes) / exp.Report(bytes)                     // diary artifacts (server hashes)

c.Query(project, kind) / c.Totals(project)                     // reads
c.Grant(research.GrantRequest{...})                            // visibility/consent (private by default)
```

Options (functional, single-word): `Metric`, `Total` (n_total), `Note` (one-shot),
`Hypothesis`, `Predict`. `Verdict` is a defined type — `Proven | Refuted | Inconclusive`;
an out-of-set value is rejected at `Conclude`.

## Open kinds — the whole company, not just eng

`kind` is an OPEN string, never an enum. `benchmark`, `kernel-perf`, `training`,
`ablation`, `policy-eval` AND `marketing-experiment`, `ad-test`, `creative-test`,
`growth-experiment`, `pricing-test`, … A marketing A/B records identically to a kernel
A/B: state a Hypothesis, Record the arms, Conclude proven/refuted. A refutation is a
first-class, durable result.

## Zero-config provenance

The caller supplies none. On `Experiment` and every seal the SDK captures: git
sha/branch/dirty (`git` subprocess, 5s-bounded), the commit narrative since this
experiment's last recorded run, Go toolchain + module versions (`debug.ReadBuildInfo`),
the host, and the calling code site (`runtime.Callers`). Serialized into `meta` in the
exact key order shared with Python/TS: `doc, commits, note, host, hypothesis, predict,
verdict, because, log`.

## Cross-language identity — semantic, not byte

Records are keyed by the server on `(project, id = kind:subject:task)` (experiments) and
`(project, benchmark, item, model)` (attempts), so every language upserts the same row
regardless of serialization — that is the identity guarantee, not byte-matching. `encode`
marshals compact with `SetEscapeHTML(false)` so `& < >` travel literally rather than as
escaped sequences; `research_test.go` proves Go's canonical meta is the SAME JSON document
as CPython `json.dumps(meta, sort_keys=True)`. Whitespace and numeric shortest-form (`0` vs
`0.0`) differ on the wire but are JSON-insignificant — the server parses and dedupes by
stable id, so records from every language interleave into one row cleanly.

## Auth

Only the per-org key (`Authorization: Bearer`), from `Config.Key` or `$HANZO_API_KEY`
(KMS-sourced through the environment, never hardcoded). The gateway mints the validated
principal + project scope; the client never sends `X-User-Id`/`X-Org-Id`. Base defaults to
`api.hanzo.ai`, no `/api/` prefix.

## Build

Standalone module. `GOWORK=off go test ./...` (not in the parent `go.work`).
