// Copyright © 2026 Hanzo AI. MIT License.

// Package research is the ONE way a Go service records and queries R&D evidence on the
// unified /v1/research surface (HIP-0512). It mirrors the Python and TypeScript producers
// verb-for-verb and byte-for-byte, so every language emits structurally identical records
// into the one store.
//
// A tiny, zero-config surface makes hand-rolling the obviously worse choice:
//
//	c := research.New(research.Config{Project: "enso-bench"})
//
//	// A benchmark run — accumulate attempts, seal with the score.
//	exp, _ := c.Experiment("benchmark", "grok-4.5", "gpqa_diamond")
//	exp.Record("q1", "grok-4.5", research.Result{Answer: "A", Correct: true})
//	exp.Snapshot(pngBytes)
//	exp.Finish(94.3)
//
//	// A kernel-perf run framed as a falsifiable test — state the claim, log what you
//	// see, then PROVE or REFUTE it. A refutation is a first-class, durable result.
//	k, _ := c.Experiment("kernel-perf", "matvec_q4k_f32_blk", "vulkan/6144x2048",
//		research.Metric("ratio_vs_hand"),
//		research.Hypothesis("the DSL f32-direct matvec beats the hand kernel"),
//		research.Predict("DSL/hand >= 1.0 cold in-engine at the dominant FFN shape"))
//	k.Log("cold in-engine A/B, evo gfx1151, quiet window, 3 runs, bit-exact 2.3e-6")
//	k.Conclude(research.Proven, "1.022x at 6144 rows (loses small shapes)", 1.022)
//
//	rows, _ := c.Query("enso-bench", "benchmark")
//
// # Open kinds — the whole company logs here, not just engineering
//
// kind is an OPEN string, never an enum. Engineering records benchmark, kernel-perf,
// training, ablation, and policy-eval; marketing, growth, and ops record
// marketing-experiment, ad-test, creative-test, growth-experiment, pricing-test, and
// whatever else they run. A marketing A/B test records IDENTICALLY to a kernel A/B: state
// a Hypothesis, Record the arms, then Conclude proven or refuted. One evidence plane, one
// shape, every team. Do not reach for a kind enum — the value is the discriminator.
//
// # Zero-config provenance
//
// The caller supplies NO provenance. On Experiment and every seal the SDK captures the
// producing repo's git sha/branch/dirty, the commit-message narrative since this
// experiment's last recorded run (the "what changed" story), the Go toolchain + module
// versions (from the build info), the host, and the calling code site — and weaves them
// into the record. Research self-documents as a side effect of running.
//
// # Auth and privacy
//
// Auth is ONLY the per-org key (Authorization: Bearer), read from Config.Key or
// $HANZO_API_KEY — sourced from KMS through the process environment, never hardcoded. The
// gateway mints the validated principal and project scope from the key; the client never
// sends X-User-Id / X-Org-Id. Records are private by default; public visibility, training,
// and commons publication are each a separate Grant, never implied by an upload.
package research
