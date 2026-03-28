# Paper Outline

## Abstract

One paragraph built around:

- translation chains as controlled perturbation,
- medium-scale models,
- stepwise evaluation,
- degradation curves,
- and category-specific failure modes.

## 1. Introduction

- Instruction following is central to practical LLM use.
- Existing benchmarks are often single-turn and single-language.
- Real-world instructions are frequently relayed through translation or rewriting.
- We propose translation chains as a stress test for instruction robustness.

## 2. Related Work

- IFEval for objective instruction following
- multilingual instruction-following evaluation
- multi-turn multilingual evaluation
- translation robustness and prompt sensitivity

## 3. Method

- benchmark definition
- translation-chain operator
- native vs back-translated regimes
- model panel
- path design
- metrics

## 4. Experimental Setup

- models
- benchmark split
- translation systems
- decoding settings
- statistical analysis

## 5. Results

- baseline results
- degradation with depth
- category-specific fragility
- path-family comparisons
- back-translation analysis
- mediation by translation quality

## 6. Failure Analysis

- representative examples
- semantic drift vs structural drift
- token-preservation failures

## 7. Discussion

- robustness vs multilinguality
- implications for localized assistants
- implications for prompt optimization and safety

## 8. Limitations

- dependence on translation systems
- partial English-centricity of the base benchmark
- constrained language coverage in the first version

## 9. Conclusion

- translation chains reveal instruction robustness differences hidden by standard benchmarks

