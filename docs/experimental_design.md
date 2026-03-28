# Experimental Design

## 1. Benchmark

Primary benchmark: `IFEval`

Reason:

- English-only starting point
- explicit constraint annotations
- programmatic validators
- widely recognized for instruction-following evaluation

Unit of analysis:

- prompt-level score
- instruction-level score
- instruction-category score

## 2. Treatment Definition

For an original instruction `x_0` in English, define a translation chain:

`x_0 -> x_1 -> x_2 -> ... -> x_t`

where `x_i` is produced by translating `x_{i-1}` into a target language chosen by a fixed path.

Two key regimes:

1. Native-chain regime:
   The model is evaluated directly on `x_t` in the current language.
2. Back-translation regime:
   After step `t`, `x_t` is translated back to English before evaluation.

This separation matters because it distinguishes multilingual understanding limitations from instruction-drift limitations.

## 3. Experimental Factors

### Independent Variables

- model identity
- model size
- translation depth
- language path
- evaluation regime: native vs back-translated
- instruction category

### Dependent Variables

- prompt-level strict accuracy
- prompt-level loose accuracy
- instruction-level strict accuracy
- instruction-level loose accuracy
- per-category pass rate
- degradation slope relative to step 0

## 4. Controls

To make the paper hard to dismiss, lock down the following:

- deterministic or near-deterministic decoding for evaluated models
- fixed prompt template across all conditions
- fixed translation system per main experiment
- same benchmark items across all models
- same language paths across all models
- same random seed and batching policy where possible

## 5. Recommended Language Paths

Use a balanced set rather than an exhaustive one.

### Family A: Low-drift Latin-script controls

- English -> French -> English
- English -> German -> English
- English -> Spanish -> English

### Family B: Script-shift paths

- English -> Russian -> English
- English -> Arabic -> English
- English -> Hindi -> English
- English -> Japanese -> English

### Family C: Multi-hop chains

- English -> German -> Russian -> English
- English -> French -> Arabic -> English
- English -> Japanese -> Hindi -> English
- English -> Spanish -> German -> French -> English

The paper is stronger if paths are chosen to cover:

- Latin vs non-Latin scripts
- fusional vs analytic vs morphologically richer languages
- high-resource translation directions
- structurally distant languages

## 6. Most Defensible Analyses

### A. Degradation Curves

Plot accuracy as a function of translation depth for each model.

Primary statistic:

- slope from mixed-effects regression with prompt and language-path random effects

### B. Category Fragility

Estimate which instruction classes lose performance fastest.

Primary statistic:

- model x category x depth interaction

### C. Back-Translation Gap

Compare native-chain evaluation with back-translated evaluation.

Interpretation:

- large native gap implies multilingual comprehension limitations
- large back-translation gap implies instruction corruption independent of final evaluation language

### D. Translation Quality Mediation

Measure whether translation quality explains model degradation.

Potential covariates:

- chrF
- COMET
- BLEU only as a secondary legacy metric
- lexical preservation of critical constraint tokens

Interpretation:

- if degradation persists after controlling for translation quality, the result is stronger than a simple MT-quality story

### E. Failure Mode Coding

For a subset of errors, annotate failure modes:

- constraint omitted
- count changed
- formatting collapsed
- prohibited token appears
- keyword requirement lost
- ordering violation
- style constraint ignored
- semantic task drift

This qualitative layer will help reviewers trust the quantitative story.

## 7. Statistical Plan

### Main Model

Use mixed-effects logistic regression on binary instruction success.

Suggested fixed effects:

- depth
- model
- language-path family
- regime
- instruction category
- relevant interactions

Suggested random effects:

- prompt item
- translation path

### Multiple Comparisons

Correct pairwise comparisons with Holm or Benjamini-Hochberg.

### Robustness Checks

- replicate with a second translation engine
- re-run on a reduced subset with stochastic decoding disabled
- compare one-hop paraphrase controls against one-hop translation
- evaluate whether longer prompts are more fragile independently of category

## 8. Critical Ablations

These are the ablations that make the paper much harder to reject.

### Ablation 1: Translation Engine

Repeat the main experiment with a second translation system.

Purpose:
Shows the result is not an artifact of a single MT model.

### Ablation 2: Back-Translation Only

Evaluate only English outputs after every chain step.

Purpose:
Separates multilingual ability from prompt corruption.

### Ablation 3: Constraint Token Preservation

Explicitly preserve known key tokens during translation for a subset of prompts.

Purpose:
Tests whether failures are caused by loss of anchor tokens rather than deeper semantic drift.

### Ablation 4: Human-Verified Subset

Human-check a small subset of translated prompts for fidelity.

Purpose:
Preempts the reviewer objection that the paper measures translation errors rather than model behavior.

## 9. Threats to Validity

### Threat

Performance drop may just reflect poor translation quality.

Mitigation:

- back-translation condition
- translation-quality covariates
- second translation engine
- human-verified subset

### Threat

The benchmark may overrepresent English-specific constraints.

Mitigation:

- analyze which constraint classes survive vs fail
- separate lexical from structural constraints
- discuss that the paper studies robustness of original instructions, not fairness across languages

### Threat

Model prompts in other languages may interact with tokenizer differences.

Mitigation:

- report tokenizer-level features as auxiliary covariates
- compare native-chain and back-translation regimes

### Threat

Repeated translation may create unnatural prompts.

Mitigation:

- this is a feature, not only a bug, because the paper studies robustness under iterative relay transformations
- still include human naturalness checks on a subset

## 10. What Reviewers Are Most Likely To Ask

1. Is this just measuring MT quality?
2. Why not use an existing multilingual benchmark directly?
3. Are the effects consistent across translation engines?
4. Which instruction types break first?
5. Does back-translation preserve the result?
6. Are failures monotonic with depth?

The protocol above is designed to answer all six.
