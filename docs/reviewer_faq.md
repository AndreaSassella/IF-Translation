# Reviewer FAQ

## Why is this not just a machine translation benchmark?

Because the target variable is not translation fidelity alone. The target variable is instruction-following success under controlled prompt perturbation. The design also includes back-translation controls, translation-quality covariates, and human verification on a subset, which makes it possible to test whether model failures exceed what translation quality alone would predict.

## Why not evaluate multilingual benchmarks directly?

Existing multilingual instruction-following benchmarks are valuable, but they do not isolate the effect of progressive translation depth from the effect of prompt content, annotation style, or multilingual dataset construction. Starting from a single-language benchmark keeps the source instruction fixed and makes the perturbation process explicit.

## Why is repeated translation scientifically interesting?

Because it creates a traceable path of controlled drift. This supports stepwise analysis, degradation curves, failure taxonomy, and causal hypotheses about where and how alignment breaks.

## Why focus on 0.5B-10B models?

This size range is scientifically useful and practically relevant. These models are widely deployed in local, on-device, and cost-constrained settings, yet they are still under-characterized compared with frontier models on robustness-style evaluations.

## What result would make the paper strongest?

The strongest result is not merely that scores go down. It is a combination of:

- consistent degradation with depth,
- model-specific robustness slopes,
- category-specific fragility,
- residual degradation even after back-translation,
- and partial non-explanation by translation-quality metrics alone.

## What would make the project weak?

- using too many languages without a principled path design,
- reporting only aggregate accuracy,
- failing to separate multilingual weakness from instruction corruption,
- using a single translation engine without ablation,
- or omitting a failure analysis.

## What is the cleanest contribution statement?

We propose translation chains as a controlled stress test for instruction-following robustness and show that medium-size models differ not only in baseline accuracy but in how quickly they lose compliance as instructions undergo repeated translation.
