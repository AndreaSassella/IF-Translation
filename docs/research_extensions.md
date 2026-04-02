# Research Extensions That Stay Novel

This project should avoid simply re-running existing multilingual instruction-following benchmarks. The closest prior work already includes:

- `M-IFEval` (Findings of NAACL 2025), which extends IFEval to French, Japanese, and Spanish and studies multilingual variation.
- `Multi-IF` (arXiv:2410.15553, October 2024), which studies multi-turn and multilingual instruction following.

That means plain "English versus translated prompt" benchmarking is less novel by itself. The strongest experiments are the ones that treat translation as a controlled perturbation process and analyze the resulting degradation curve.

## Added Models

To extend the evaluation without changing the current validated run, use the separate config:

- `configs/runtime.extended_models.json`
- `configs/runtime.qwen3_and_llama.json`

It adds two larger models on top of the current default model:

- `Qwen/Qwen2.5-7B-Instruct`
- `google/gemma-2-9b-it`

An additional comparison config adds:

- `Qwen/Qwen3-4B`
- `Qwen/Qwen3-8B`
- `meta-llama/Meta-Llama-3.1-8B-Instruct`

Why these two:

- `Qwen2.5-7B-Instruct` gives a larger model from the same family as the current `Qwen2.5-0.5B-Instruct`, which is ideal for measuring scaling effects under translation-chain perturbation.
- `Gemma-2-9b-it` is a strong open model with a different training recipe and a more English-leaning profile, making it a useful contrast case against Qwen's broader multilingual positioning.

Why the Qwen3 plus Llama comparison is useful:

- `Qwen3-4B` and `Qwen3-8B` let you test whether the newer Qwen generation changes the robustness slope relative to Qwen2.5.
- `Llama-3.1-8B-Instruct` is a strong multilingual open baseline with published IFEval numbers on its model card, making it a reviewer-friendly comparison point.

## Additional Research Questions Worth Pursuing

### 1. Scaling versus Robustness

Question:
Does scaling improve baseline instruction following more than it improves robustness to translation drift?

Why it matters:
Many papers show that larger models improve absolute scores. Fewer isolate whether larger models also flatten the degradation slope under controlled prompt drift.

Minimal experiment:

- compare `Qwen2.5-0.5B-Instruct` and `Qwen2.5-7B-Instruct`
- report both depth-0 accuracy and relative drop by depth

### 2. Family Effect at Similar Scale

Question:
At roughly similar model size, do family/training differences matter more than scale for translation robustness?

Minimal experiment:

- compare `Qwen2.5-7B-Instruct` and `Gemma-2-9b-it`
- keep the same dataset, paths, regimes, and evaluator

Why reviewers may care:
This helps separate scale effects from family-specific multilingual robustness.

### 3. Path-Order Effect

Question:
Does the order of languages in a multi-hop chain matter, even when the language set is the same?

Examples:

- `en -> fr -> ar -> en`
- `en -> ar -> fr -> en`

Why it matters:
If order matters, the result supports path-dependent error accumulation rather than generic multilingual difficulty.

### 4. Native versus Back-Translated Gap

Question:
How much degradation comes from multilingual understanding versus instruction corruption?

Minimal experiment:

- compare `native` and `back_translated`
- report the gap by model and by instruction category

Why it matters:
This is one of the strongest mechanistic questions in the whole setup.

### 5. Constraint-Class Fragility

Question:
Which instruction types are most vulnerable under repeated translation?

Priority classes:

- lexical requirements
- formatting requirements
- counting constraints
- structured output constraints
- ordering constraints

Why it matters:
This turns the paper from a benchmark table into a diagnostic analysis.

### 6. Translation-Engine Sensitivity

Question:
Do the main findings hold if you swap NLLB for a second translation system?

Why it matters:
This directly addresses the reviewer objection that the paper is only measuring artifacts of one MT model.

## Experiments To Deprioritize

These are closer to already-covered territory:

- plain one-hop multilingual benchmarking with no chain-depth analysis
- average multilingual score comparisons without category analysis
- direct reimplementation of M-IFEval language comparisons without perturbation framing
- multi-turn extensions before the single-turn translation-chain story is complete

## Suggested Next Benchmark Extension

The cleanest next run is:

1. keep the current default run unchanged
2. run `configs/runtime.extended_models.json`
3. compare:
   - baseline depth-0 accuracy
   - degradation slope by depth
   - native versus back-translated gap
   - category-specific fragility

That adds a scaling story and a cross-family story without invalidating the current code path or completed experiments.
