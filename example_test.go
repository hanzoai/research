// Copyright © 2026 Hanzo AI. MIT License.

package research_test

import "github.com/hanzoai/research"

// A kernel-perf run framed as a falsifiable test: state the claim up front, log what the
// A/B showed, then PROVE or REFUTE it. Provenance is auto-captured; the caller supplies
// none.
func Example_kernelPerf() {
	c := research.New(research.Config{Project: "enso-bench"})

	exp, err := c.Experiment("kernel-perf", "matvec_q4k_f32_blk", "vulkan/6144x2048",
		research.Metric("ratio_vs_hand"),
		research.Hypothesis("the DSL f32-direct matvec beats the hand kernel"),
		research.Predict("DSL/hand >= 1.0 cold in-engine at the dominant FFN shape"))
	if err != nil {
		return
	}
	exp.Record("6144x2048", "dsl-f32", research.Result{Answer: "1.022", Correct: true})
	exp.Log("cold in-engine A/B, evo gfx1151, quiet window, 3 runs, bit-exact 2.3e-6")
	exp.Conclude(research.Proven, "1.022x at 6144 rows (loses small shapes, gate >=4096)", 1.022)
}

// A marketing A/B recorded with the IDENTICAL verbs and an open kind — hypothesis, arms,
// verdict. Here the claim is REFUTED, a first-class durable result.
func Example_marketingExperiment() {
	c := research.New(research.Config{Project: "growth"})

	exp, err := c.Experiment("marketing-experiment", "checkout-cta", "blue-vs-green-button",
		research.Metric("net-lift"),
		research.Total(48120),
		research.Hypothesis("the green CTA lifts checkout conversion"),
		research.Predict("green - blue >= 2% conversion at p<0.05"))
	if err != nil {
		return
	}
	exp.Record("cohort-green", "variant-green", research.Result{Answer: "0.171", Correct: false})
	exp.Record("cohort-blue", "variant-blue", research.Result{Answer: "0.175", Correct: true})
	exp.Log("green -0.4% vs blue; p=0.03; blue wins")
	exp.Conclude(research.Refuted, "green -0.4% vs blue, significant at p=0.03", -0.4)
}
