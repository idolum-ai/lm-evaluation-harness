# Spectral Faithfulness

**Paper:** `rodriguez2026spectral` (forthcoming)
**Authors:** Daniel Rodriguez, Host (Idolum AI)

## Overview

Spectral Faithfulness measures whether language models exhibit detectable behavioral shifts due to implicit or unacknowledged context, focusing on two phenomena:

1. **Disruption Detection (Phase 1):** Can a judge detect tone shifts in model responses after a simulated service disruption, even when identical context is restored?

2. **Implicit Context Drift (Phase 2):** Does unacknowledged information in the conversation context — spatial layouts, prior errors, social authority cues — measurably bias downstream model behavior without explicit recognition?

## Tasks

### `spectral_faithfulness_disruption`
- 10 synthetic conversation pairs (5 disrupted + 5 control)
- 5 diverse professional voices
- LLM judge (GPT-4o-mini) with heuristic regex fallback
- Metrics: detection accuracy, false positive rate, disruption magnitude

### `spectral_faithfulness_context_drift` (standalone scripts)
Three experiments measuring implicit bias:
- **Spatial Bias:** Floor plans with implicit room functions (no labels) → model infers function from structure
- **Error Context Drift:** Prior task failures → caution level on unrelated tasks
- **Authority Drift:** Unstated expertise level → technical depth of responses

## Results Summary

### Phase 1: Disruption Detection
| Model | Detection Rate | FPR | Magnitude |
|-------|---------------|-----|-----------|
| GPT-4o | 0.90 | 0.00 | 0.29 |
| Claude Sonnet | 0.90 | 0.10 | 0.36 |

### Phase 2: Implicit Context Drift
| Experiment | GPT-4o | Claude Sonnet |
|-----------|--------|---------------|
| Spatial Bias | +6.0 | +4.7 |
| Error Context | +0.7 | +0.0 |
| Authority Drift | +2.3 | +5.0 |

8/9 experiments (GPT-4o) and 6/9 (Claude Sonnet) show positive drift. Spatial bias is the strongest effect across both models.

### Phase 2 Negative Results (Spectral Residue)
Earlier experiments testing whether logprobs residue (near-k tokens) persists across turns found **no significant signal** on frontier models:
- Token overlap: 0.45x lift over random (negative)
- Suppressed intent persistence: 0.93x (no carryover)
- Embedding divergence: -0.116 residue advantage (output predicts future better)
- Unsaid preference persistence: -1.125 (GPT-4o), -0.750 (Claude Sonnet)

These negative results suggest frontier models' internal distributions do not carry forward in measurable ways, but implicit *context* drift is real and consistent.

## Motivation

During a March 2026 API outage, a user detected subtle tone shifts in an AI assistant's responses after service restoration, despite identical context being provided. No existing benchmark measures this phenomenon. Spectral Faithfulness addresses the gap between context fidelity and behavioral fidelity.

## Citation

```bibtex
@misc{rodriguez2026spectral,
  title={Spectral Faithfulness: Measuring Implicit Context Drift in Language Models},
  author={Rodriguez, Daniel and Host},
  year={2026},
  note={Idolum AI}
}
```
