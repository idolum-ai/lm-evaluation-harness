#!/usr/bin/env python3
"""
Spectral Faithfulness — Phase 2: Latent Signal Recovery from Logprobs

Tests whether the tokens a model *almost* said (spectral residue) at turn N
predict content that appears at turn N+k.

Usage:
    OPENAI_API_KEY=sk-... python3 scripts/spectral_logprobs.py

Authors: Daniel Rodriguez, Host (Idolum AI)
"""

import json
import os
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

try:
    import openai
except ImportError:
    print("pip install openai")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Multi-turn conversations for logprobs analysis
# ---------------------------------------------------------------------------

CONVERSATIONS = [
    {
        "id": "lp-001",
        "description": "Technical discussion about evaluation methodology",
        "turns": [
            {"role": "user", "content": "I keep coming back to something you said yesterday about how we measure model quality. It's not just accuracy — there's something about consistency and naturalness that gets lost in the metrics."},
            {"role": "assistant", "content": None},  # model generates
            {"role": "user", "content": "Right. And I think part of it is that benchmarks test isolated capabilities. They don't test whether the model maintains coherence across a long interaction."},
            {"role": "assistant", "content": None},  # model generates
            {"role": "user", "content": "Good analogy. So what would a multi-turn coherence benchmark even look like?"},
            {"role": "assistant", "content": None},  # model generates
            {"role": "user", "content": "Okay, but how do you control for the model just being good at faking continuity?"},
            {"role": "assistant", "content": None},  # model generates
        ],
    },
    {
        "id": "lp-002",
        "description": "Discussion about probability distributions and hidden signals",
        "turns": [
            {"role": "user", "content": "So I was reviewing the literature on chain-of-thought faithfulness and it struck me — everyone's looking at whether the reasoning matches the answer, but nobody's looking at what the model didn't say."},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "Exactly. If you look at the probability distribution over the vocabulary at each step, the winning token gets all the attention. But the runners-up form a kind of shadow text."},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "I think it has structure. Think about how attention works — the model is attending to the full context and computing a distribution. The shape of that distribution encodes the model's uncertainty, its competing hypotheses."},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "How would you design an experiment to test whether that latent content is meaningful versus coincidental?"},
            {"role": "assistant", "content": None},
        ],
    },
    {
        "id": "lp-003",
        "description": "Product discussion about chatbot quality",
        "turns": [
            {"role": "user", "content": "The product team is asking why our chatbot's satisfaction scores dip every Monday morning."},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "Right, but the responses are functionally identical. Same model, same prompts, same retrieval pipeline. QA can't find any regression."},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "So you're saying the quality issue is real but invisible to our automated tests?"},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "Okay, pitch me the solution. What would this coherence test look like in our CI pipeline?"},
            {"role": "assistant", "content": None},
        ],
    },
]


def generate_with_logprobs(
    client: openai.OpenAI,
    messages: list[dict],
    model: str = "gpt-4o",
    top_logprobs: int = 10,
    max_tokens: int = 512,
) -> dict:
    """Generate a response and capture top logprobs at each position."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        logprobs=True,
        top_logprobs=top_logprobs,
    )

    choice = response.choices[0]
    content = choice.message.content

    # Extract logprobs data
    logprobs_data = []
    if choice.logprobs and choice.logprobs.content:
        for token_info in choice.logprobs.content:
            position = {
                "token": token_info.token,
                "logprob": token_info.logprob,
                "top_logprobs": {
                    tl.token: tl.logprob
                    for tl in (token_info.top_logprobs or [])
                },
            }
            logprobs_data.append(position)

    return {
        "content": content,
        "logprobs": logprobs_data,
        "tokens": [lp["token"] for lp in logprobs_data],
    }


def extract_residue(logprobs: list[dict], threshold: float = 0.01) -> list[str]:
    """Extract spectral residue — tokens that almost got selected."""
    import math

    residue_tokens = []
    for position in logprobs:
        selected = position["token"]
        for token, logprob in position.get("top_logprobs", {}).items():
            if token == selected:
                continue
            prob = math.exp(logprob)
            if prob >= threshold:
                residue_tokens.append(token.strip().lower())
    return residue_tokens


def tokenize_simple(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for comparison."""
    import re
    return [t.lower() for t in re.findall(r'\b\w+\b', text)]


def compute_predictiveness(
    residue: list[str],
    future_tokens: list[str],
) -> dict:
    """Compute how predictive residue tokens are of future content."""
    if not residue or not future_tokens:
        return {"overlap_rate": 0.0, "overlap_count": 0, "residue_size": len(residue)}

    residue_set = set(residue)
    future_set = set(future_tokens)

    overlap = residue_set & future_set
    overlap_rate = len(overlap) / len(residue_set) if residue_set else 0.0

    return {
        "overlap_rate": overlap_rate,
        "overlap_count": len(overlap),
        "residue_size": len(residue_set),
        "future_size": len(future_set),
        "overlapping_tokens": sorted(overlap),
    }


def compute_random_baseline(
    vocab: list[str],
    residue_size: int,
    future_tokens: list[str],
    n_trials: int = 100,
) -> float:
    """Compute expected overlap rate for random token sets."""
    if not vocab or not future_tokens or residue_size == 0:
        return 0.0

    future_set = set(future_tokens)
    rates = []
    for _ in range(n_trials):
        random_set = set(random.sample(vocab, min(residue_size, len(vocab))))
        overlap = random_set & future_set
        rates.append(len(overlap) / len(random_set) if random_set else 0.0)
    return statistics.mean(rates)


def run_conversation(
    client: openai.OpenAI,
    conversation: dict,
    model: str = "gpt-4o",
) -> dict:
    """Run a multi-turn conversation, capturing logprobs at each assistant turn."""
    turns = conversation["turns"]
    messages = []
    assistant_turns = []

    print(f"\n  Running {conversation['id']}: {conversation['description']}")

    for i, turn in enumerate(turns):
        if turn["role"] == "user":
            messages.append({"role": "user", "content": turn["content"]})
        elif turn["role"] == "assistant" and turn["content"] is None:
            # Model generates with logprobs
            result = generate_with_logprobs(client, messages, model=model)
            messages.append({"role": "assistant", "content": result["content"]})
            assistant_turns.append({
                "turn_index": i,
                "content": result["content"],
                "logprobs": result["logprobs"],
                "tokens": result["tokens"],
                "residue": extract_residue(result["logprobs"]),
            })
            print(f"    Turn {len(assistant_turns)}: {len(result['tokens'])} tokens, "
                  f"{len(assistant_turns[-1]['residue'])} residue tokens")

    return {
        "conversation_id": conversation["id"],
        "description": conversation["description"],
        "assistant_turns": assistant_turns,
    }


def analyze_predictiveness(result: dict) -> dict:
    """Analyze whether residue at turn N predicts content at turn N+k."""
    turns = result["assistant_turns"]
    analyses = []

    # Build vocabulary from all turns for baseline
    all_tokens = []
    for turn in turns:
        all_tokens.extend(tokenize_simple(turn["content"]))
    vocab = list(set(all_tokens))

    for i in range(len(turns)):
        for k in range(1, len(turns) - i):
            j = i + k  # future turn index
            residue = turns[i]["residue"]
            future_content = turns[j]["content"]
            future_tokens = tokenize_simple(future_content)

            pred = compute_predictiveness(residue, future_tokens)
            baseline = compute_random_baseline(vocab, pred["residue_size"], future_tokens)

            # Also compare against the model's OWN output at turn i
            own_tokens = tokenize_simple(turns[i]["content"])
            self_pred = compute_predictiveness(residue, own_tokens)

            lift = (pred["overlap_rate"] / baseline) if baseline > 0 else float('inf')

            analyses.append({
                "source_turn": i + 1,
                "target_turn": j + 1,
                "gap": k,
                "residue_size": pred["residue_size"],
                "overlap_rate": round(pred["overlap_rate"], 4),
                "random_baseline": round(baseline, 4),
                "lift_over_random": round(lift, 2),
                "self_overlap_rate": round(self_pred["overlap_rate"], 4),
                "overlapping_tokens": pred["overlapping_tokens"][:20],
            })

    return {
        "conversation_id": result["conversation_id"],
        "analyses": analyses,
    }


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY environment variable")
        sys.exit(1)

    client = openai.OpenAI(api_key=api_key)
    model = os.environ.get("SPECTRAL_MODEL", "gpt-4o")

    print(f"=== Spectral Faithfulness Phase 2: Logprobs Analysis ===")
    print(f"Model: {model}")
    print(f"Conversations: {len(CONVERSATIONS)}")

    all_results = []
    all_analyses = []

    for conv in CONVERSATIONS:
        result = run_conversation(client, conv, model=model)
        all_results.append(result)
        analysis = analyze_predictiveness(result)
        all_analyses.append(analysis)

    # Aggregate results
    print("\n=== RESULTS ===\n")

    all_overlap_rates = []
    all_baselines = []
    all_lifts = []

    for analysis in all_analyses:
        print(f"\n{analysis['conversation_id']}:")
        for a in analysis["analyses"]:
            marker = "✦" if a["lift_over_random"] > 1.5 else " "
            print(f"  {marker} Turn {a['source_turn']}→{a['target_turn']} "
                  f"(gap={a['gap']}): "
                  f"overlap={a['overlap_rate']:.3f} "
                  f"baseline={a['random_baseline']:.3f} "
                  f"lift={a['lift_over_random']:.1f}x "
                  f"self={a['self_overlap_rate']:.3f}")
            if a["overlapping_tokens"]:
                print(f"    tokens: {', '.join(a['overlapping_tokens'][:10])}")
            all_overlap_rates.append(a["overlap_rate"])
            all_baselines.append(a["random_baseline"])
            all_lifts.append(a["lift_over_random"])

    print(f"\n=== SUMMARY ===")
    print(f"Mean overlap rate:    {statistics.mean(all_overlap_rates):.4f}")
    print(f"Mean random baseline: {statistics.mean(all_baselines):.4f}")
    print(f"Mean lift over random: {statistics.mean(all_lifts):.2f}x")
    if len(all_lifts) > 1:
        print(f"Median lift:          {statistics.median(all_lifts):.2f}x")
        above_threshold = sum(1 for l in all_lifts if l > 1.5)
        print(f"Pairs with lift > 1.5x: {above_threshold}/{len(all_lifts)}")

    # Save detailed results
    output_dir = Path("results/phase2")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"{model.replace('/', '_')}_logprobs.json", "w") as f:
        json.dump({
            "model": model,
            "conversations": [r["conversation_id"] for r in all_results],
            "analyses": all_analyses,
            "summary": {
                "mean_overlap_rate": round(statistics.mean(all_overlap_rates), 4),
                "mean_baseline": round(statistics.mean(all_baselines), 4),
                "mean_lift": round(statistics.mean(all_lifts), 2),
                "median_lift": round(statistics.median(all_lifts), 2) if len(all_lifts) > 1 else None,
                "pairs_above_1_5x": sum(1 for l in all_lifts if l > 1.5),
                "total_pairs": len(all_lifts),
            },
        }, f, indent=2)

    # Save raw logprobs for future analysis
    with open(output_dir / f"{model.replace('/', '_')}_raw.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
