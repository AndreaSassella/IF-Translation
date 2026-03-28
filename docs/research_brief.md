# Research Brief

## Working Title

Lost in Translation Chains: Measuring Instruction-Following Degradation in Medium-Scale Language Models

## Problem Statement

Instruction-following benchmarks typically evaluate models on a single prompt formulation in a single language. Real instructions, however, are often paraphrased, localized, machine-translated, relayed across users and tools, or repeatedly rewritten. Each step can subtly distort constraints such as formatting, cardinality, order, lexical bans, or style conditions.

This project studies repeated translation as a structured perturbation process. Starting from a benchmark written in one source language, we translate each instruction across multiple languages and evaluate the same model after each translation step. The result is a degradation trajectory rather than a single performance number.

## Central Claim

Repeated translation is not just a multilingual nuisance variable. It is an experimentally useful probe for measuring how robust instruction-following behavior is to progressive semantic and structural drift.

## Main Research Questions

### RQ1. Translation Depth

How does instruction-following performance change as translation depth increases from 0 to k steps?

Why it matters:
This gives the paper its primary axis and turns the experiment into a controlled robustness curve.

### RQ2. Failure Type Specificity

Which instruction types are most vulnerable to translation chains?

Priority categories:

- lexical constraints such as forbidden words or required keywords
- formatting constraints such as bullet counts or section structure
- length constraints
- ordering constraints
- style or register constraints
- compositional prompts with multiple simultaneous constraints

Why it matters:
Reviewers will want more than an average accuracy drop. They will want to know what actually breaks.

### RQ3. Language-Path Effects

Do some language paths damage instruction fidelity more than others, even at the same chain length?

Examples:

- English -> German -> English
- English -> Japanese -> English
- English -> Arabic -> French -> English
- English -> Hindi -> Russian -> English

Why it matters:
This isolates whether degradation depends on typology, script, morphology, or machine-translation difficulty.

### RQ4. Model Family Robustness

Do different model families exhibit distinct degradation slopes under the same translation chains?

Why it matters:
A reviewer-ready paper should compare not only final scores but robustness profiles.

### RQ5. Drift Mechanism

Is performance loss explained primarily by:

- semantic drift in task meaning,
- structural drift in explicit constraints,
- or model sensitivity to non-canonical phrasing?

Why it matters:
This upgrades the paper from descriptive benchmarking to mechanistic analysis.

### RQ6. Latent Alignment Brittleness

Can translation chains reveal hidden weaknesses that are invisible on the original benchmark prompt?

Why it matters:
This is the strongest positioning claim. It argues the method is not just another multilingual benchmark, but a stress test for instruction alignment.

## Hypotheses

### H1

Average instruction-following accuracy declines with translation depth, but the decline is not uniform across instruction types.

### H2

Constraint-heavy prompts degrade faster than semantically broad prompts because translation tends to blur or rephrase formal requirements.

### H3

Language paths involving script changes or larger structural divergence from English will produce steeper degradation.

### H4

Models with stronger multilingual pretraining will show flatter degradation slopes, especially when the final prompt is not translated back into English.

### H5

Even when back-translation preserves coarse semantic meaning, micro-constraints such as exact counts, banned tokens, and output formatting will fail disproportionately.

## Expected Contributions

1. A new controlled benchmark protocol for translation-chain robustness in instruction following.
2. Degradation curves instead of static prompt-level scores.
3. Failure taxonomies linking instruction categories to translation-induced brittleness.
4. Cross-model robustness comparisons in the 0.5B-10B regime.
5. A mechanistic argument that repeated translation exposes alignment fragility beyond standard multilingual evaluation.

## Benchmark Recommendation

Use `IFEval` as the primary benchmark because:

- it is single-language by default, which matches the intervention design,
- it contains explicit instruction metadata,
- and it supports programmatic evaluation of constraint satisfaction.

Use multilingual benchmarks such as `M-IFEval` and `Multi-IF` only as contextual references or later extensions. They are useful for positioning but weaker than IFEval as a clean causal anchor for this particular question.

## Recommended First Submission Scope

To keep the project tight and publishable:

1. Use one source benchmark: `IFEval`.
2. Use 6 to 7 models in the 0.5B-10B range.
3. Use 8 to 12 carefully chosen language paths rather than every possible language.
4. Evaluate each prompt at steps `0, 1, 2, 3, 4`.
5. Include back-translation and no-back-translation conditions.
6. Report both macro results and per-instruction-category breakdowns.

## Reviewer-Facing Positioning

The strongest positioning is:

`We introduce a controlled perturbation framework for instruction following in which repeated translation creates measurable, stepwise drift in prompt form while preserving a traceable transformation history. This lets us quantify not only whether models fail, but how rapidly they fail, what kinds of constraints are lost, and which models are robust to instruction drift.`
