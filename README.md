# Translation Chains and Instruction-Following Robustness

This repository contains an executable research scaffold on how repeated translation alters instruction-following performance in medium-size language models. The core setup starts from a single-language benchmark, applies controlled multi-step translation chains to the original instruction, and evaluates the same model after every translation step.

The main scientific question is not just whether multilingual prompts are harder. It is whether semantic drift, formatting drift, and constraint drift accumulate as instructions are translated repeatedly, and whether different models fail in measurably different ways.

## Research Thesis

Repeated translation acts as a structured perturbation on an instruction. If a benchmark provides verifiable instruction-level constraints, we can measure:

- when performance begins to degrade,
- which instruction types degrade fastest,
- which language paths are most harmful,
- whether degradation is explained by translation quality alone,
- and whether some models are unusually robust to instruction drift.

## Upside

- `IFEval` is a good anchor benchmark because its constraints are programmatically checkable.
- Translation chains induce a controllable perturbation process instead of ad hoc prompt rewriting.
- Evaluating at every step gives a degradation curve, not a single before/after number.
- The design supports causal analysis through fixed prompts, fixed decoders, fixed models, and controlled language paths.

## Repository Layout

- `docs/research_brief.md`: reviewer-facing framing, questions, hypotheses, and contributions
- `docs/experimental_design.md`: benchmark choice, protocol, statistics, and threats to validity
- `docs/reviewer_faq.md`: likely reviewer objections and how the design addresses them
- `configs/models.yaml`: candidate model panel in the 0.5B-10B range
- `configs/language_paths.yaml`: translation-chain families and controls
- `configs/experiment.yaml`: default experimental settings
- `configs/runtime.experiment.json`: runnable local config
- `configs/runtime.hf.example.json`: example Hugging Face-backed config
- `data/sample/ifeval_like_sample.jsonl`: local `IFEval`-like sample dataset
- `src/translation_chains/`: executable pipeline for loading prompts, translating, evaluating, and aggregating
- `notebooks/`: step-by-step walkthroughs of the data flow, run status, and result inspection

## Recommended Core Benchmark

Primary benchmark:

- `IFEval` ([dataset card](https://huggingface.co/datasets/HuggingFaceH4/ifeval), linked to arXiv:2311.07911)

Contextual references:

- `M-IFEval` / multilingual instruction-following evaluation as a reference point for multilingual difficulty
- `Multi-IF` (2024) as evidence that multilingual and multi-turn instruction following are still underexplored and brittle

## Recommended Initial Model Panel

The project is strongest if the initial submission uses a compact but diverse panel:

- `Qwen2.5-0.5B-Instruct`
- `Qwen2.5-3B-Instruct`
- `Qwen2.5-7B-Instruct`
- `Llama-3.2-3B-Instruct`
- `Ministral-3B-Instruct-2410`
- `Ministral-8B-Instruct-2410`
- `Gemma-2-9b-it`

This mix gives scale variation, architecture variation, and different multilingual priors without exploding compute.

## Most Interesting Research Questions

The highest-value questions for a paper are:

1. Does instruction-following accuracy decay monotonically with translation depth, or do some chains show non-monotonic recovery?
2. Which instruction categories are most fragile under translation: formatting, lexical constraints, counting constraints, ordering constraints, or style constraints?
3. Is degradation better explained by translation quality metrics, by language-family transitions, or by model family?
4. Do multilingual-capable models preserve constraint satisfaction better than English-dominant models when the final prompt is translated back into English?
5. Are failures driven by semantic drift in the task meaning or by corruption of the instruction schema itself?
6. Does chain translation expose latent alignment brittleness that is not visible on standard one-shot instruction-following benchmarks?

## Quick Start

The core pipeline is standard-library-first, but the full research workflow now expects the packages listed in `requirements.txt` for Hugging Face loading and notebook usage.

1. Install Python 3.10+.
2. From the repository root, run:

```bash
python main.py inspect-dataset --dataset data/sample/ifeval_like_sample.jsonl
python main.py inspect-config --config configs/runtime.experiment.json
python main.py run --config configs/runtime.experiment.json
python main.py status --status-file outputs/sample_run/status.json
python main.py report --results outputs/sample_run/results.jsonl --output-dir outputs/sample_run
python main.py audit-results --results outputs/sample_run/results.jsonl --output-dir outputs/sample_run
```

Expected outputs:

- `outputs/sample_run/results.jsonl`
- `outputs/sample_run/summary.json`
- `outputs/sample_run/status.json`
- `outputs/sample_run/model_by_depth.csv`
- `outputs/sample_run/regime_summary.csv`
- `outputs/sample_run/category_scores.csv`
- `outputs/sample_run/REPORT.md`
- `outputs/sample_run/audit.json`

## Dataset Source

The current default experiment config does **not** use Hugging Face. It uses the local sample dataset in `data/sample/ifeval_like_sample.jsonl`.

If you want to run against a Hugging Face dataset, use `configs/runtime.hf.example.json` as a starting point. The repository now supports two dataset modes:

- `local_jsonl`
- `huggingface`

When a run finishes, `summary.json` records the detected dataset source so you can verify what was actually used.

Important caveat:
Loading `IFEval` from Hugging Face is now supported, but reproducing the official `IFEval` benchmark still requires integrating the real validator logic for raw instruction IDs. The repository includes a safety guard and will refuse to run those cases unless you explicitly set `allow_placeholder_ifeval_evaluator` to `true` for debugging only.

## Notebooks

The notebooks are meant to make the code path easy to follow:

- `notebooks/01_walkthrough_dataset_and_pipeline.ipynb`
- `notebooks/02_walkthrough_status_and_results.ipynb`

The first explains loading, translation, and per-step evaluation. The second explains how to inspect run status, tables, audit findings, and quick sanity plots.

## Adapter Model

The repository ships with executable local adapters so the full pipeline works immediately:

- translation adapters:
  - `identity`
  - `drift`
- model adapters:
  - `reference`
  - `fragile`
  - `echo`

These are intentionally simple. They are for smoke testing and development, not for publication-grade runs. The adapter interfaces are designed so you can later plug in real translation systems and real LLM backends.

## Experiment Status and Result Reasonableness

Every run now writes `status.json` with:

- run state
- dataset source
- dataset size
- expected rows
- completed rows
- timestamps

Every report build also writes `audit.json`, which checks:

- whether score values stay in `[0, 1]`
- whether baseline rows are present
- whether expected regimes are present
- whether perturbed averages suspiciously exceed baseline averages