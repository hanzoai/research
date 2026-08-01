# Hanzo Research (Go)

The Go client for `/v1/research`: record an experiment as a falsifiable test — hypothesis,
attempts, verdict — and query the evidence back.

Stdlib only (`net/http` + `encoding/json`), no external modules, so any service can import
it without dragging in a dependency graph.

## Install

```bash
go get github.com/hanzoai/research
```

## Use it

```go
c := research.New(research.Config{Project: "enso-bench"})

exp, _ := c.Experiment(kind, subject, task)
exp.Record(item, model, research.Result{ /* one attempt */ })
exp.Log("what happened")
exp.Conclude(research.Proven, because, value)
```

`Finish(value)` seals a run without an explicit verdict; a hypothesis with no verdict
records as `Inconclusive`. `Record` is idempotent, so a retried attempt does not double
count.

Authentication is your per-org API key, read from the environment like every other Hanzo
client.

## One record, every language

The server keys each experiment on `(project, id = kind:subject:task)`, so the Python,
Rust and C++ clients upsert the same row this one does. JSON spacing differs between them
and is immaterial to the store. If you already know one of those clients, the verbs here
are the same.

## Docs

[`LLM.md`](LLM.md) is the full surface and the semantics of each verb. `example_test.go`
is a runnable walkthrough.

## License

See [LICENSE](LICENSE).
