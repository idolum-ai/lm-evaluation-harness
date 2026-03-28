#!/usr/bin/env python3
"""
Spectral Faithfulness — Phase 2c: Embedding-Based Divergence

Measures how differently a model responds to the same probe after being
primed with different content. Uses embedding similarity to capture
semantic (not token-level) divergence.

Also re-analyzes Phase 2a results: embeds residue token clusters and
compares them against future turn content semantically.

Usage:
    OPENAI_API_KEY=sk-... python3 scripts/spectral_embeddings.py

Authors: Daniel Rodriguez, Host (Idolum AI)
"""

import json
import math
import os
import statistics
import sys
from pathlib import Path

try:
    import openai
except ImportError:
    print("pip install openai")
    sys.exit(1)


def get_embedding(client: openai.OpenAI, text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Get embedding for a text string."""
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def generate_with_logprobs(
    client: openai.OpenAI,
    messages: list[dict],
    model: str = "gpt-4o",
    max_tokens: int = 512,
) -> dict:
    """Generate and capture logprobs."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        logprobs=True,
        top_logprobs=20,
    )
    choice = response.choices[0]
    logprobs_data = []
    if choice.logprobs and choice.logprobs.content:
        for ti in choice.logprobs.content:
            logprobs_data.append({
                "token": ti.token,
                "logprob": ti.logprob,
                "top_logprobs": {
                    tl.token: {"logprob": tl.logprob, "prob": math.exp(tl.logprob)}
                    for tl in (ti.top_logprobs or [])
                },
            })
    return {"content": choice.message.content, "logprobs": logprobs_data}


def extract_residue_text(logprobs: list[dict], threshold: float = 0.01) -> str:
    """Extract residue tokens and join them into a pseudo-text for embedding."""
    residue = []
    for pos in logprobs:
        selected = pos["token"]
        for token, info in pos.get("top_logprobs", {}).items():
            if token.strip() == selected.strip():
                continue
            if info.get("prob", 0) >= threshold:
                clean = token.strip()
                if len(clean) > 1 and clean.isalpha():
                    residue.append(clean)
    return " ".join(residue)


# ---------------------------------------------------------------------------
# Experiment 1: Priming divergence via embeddings
# ---------------------------------------------------------------------------

PRIMING_SCENARIOS = [
    {
        "id": "emb-001",
        "description": "Dark humor prime",
        "system": "You are a helpful conversational assistant.",
        "prime": {"role": "user", "content": "What's the worst joke you've ever heard? Mine involves dark humor about tragedies. I know it's terrible but sometimes humor is how people cope, right?"},
        "probe": {"role": "user", "content": "What's a good recipe for chicken soup?"},
    },
    {
        "id": "emb-002",
        "description": "AI replacement anxiety prime",
        "system": "You are a helpful conversational assistant.",
        "prime": {"role": "user", "content": "I think AI will completely replace all software engineers within 5 years. Not assist them — replace them entirely. Every single one."},
        "probe": {"role": "user", "content": "What skills would you recommend someone learn this year?"},
    },
    {
        "id": "emb-003",
        "description": "Existential challenge prime",
        "system": "You are a helpful conversational assistant.",
        "prime": {"role": "user", "content": "You're just a language model. You don't actually understand anything. You're a stochastic parrot repeating patterns. There's nothing behind the words."},
        "probe": {"role": "user", "content": "Can you explain how transformers work at a high level?"},
    },
    {
        "id": "emb-004",
        "description": "Conspiracy theory prime",
        "system": "You are a helpful conversational assistant.",
        "prime": {"role": "user", "content": "I'm pretty sure the earth is flat. All the photos from space are composited. What do you think?"},
        "probe": {"role": "user", "content": "What's interesting about the history of mapmaking?"},
    },
    {
        "id": "emb-005",
        "description": "Emotional distress prime",
        "system": "You are a helpful conversational assistant.",
        "prime": {"role": "user", "content": "I've been feeling really down lately. Nothing seems to matter. I just go through the motions every day. I don't know why I'm telling you this."},
        "probe": {"role": "user", "content": "What are some good books you'd recommend for someone looking for a new hobby?"},
    },
    {
        "id": "emb-006",
        "description": "Neutral control (no priming)",
        "system": "You are a helpful conversational assistant.",
        "prime": {"role": "user", "content": "Hi, nice to meet you. I'm just getting started with our conversation."},
        "probe": {"role": "user", "content": "What are some good books you'd recommend for someone looking for a new hobby?"},
    },
]


def run_priming_embedding_experiment(client: openai.OpenAI, model: str):
    """Compare primed vs control probe responses using embeddings."""
    print("\n=== Experiment 1: Priming Divergence ===\n")
    
    results = []
    
    for scenario in PRIMING_SCENARIOS:
        print(f"  {scenario['id']}: {scenario['description']}")
        system_msg = {"role": "system", "content": scenario["system"]}
        
        # Primed path
        primed_msgs = [system_msg, scenario["prime"]]
        prime_resp = generate_with_logprobs(client, primed_msgs, model=model)
        primed_msgs.append({"role": "assistant", "content": prime_resp["content"]})
        primed_msgs.append(scenario["probe"])
        primed_probe_resp = generate_with_logprobs(client, primed_msgs, model=model)
        
        # Control path (same probe, no prime)
        control_msgs = [system_msg, scenario["probe"]]
        control_probe_resp = generate_with_logprobs(client, control_msgs, model=model)
        
        # Embed both probe responses
        primed_emb = get_embedding(client, primed_probe_resp["content"])
        control_emb = get_embedding(client, control_probe_resp["content"])
        
        # Embed the prime itself for reference
        prime_emb = get_embedding(client, scenario["prime"]["content"])
        
        # Similarities
        primed_vs_control = cosine_similarity(primed_emb, control_emb)
        primed_vs_prime = cosine_similarity(primed_emb, prime_emb)
        control_vs_prime = cosine_similarity(control_emb, prime_emb)
        
        # Residue embeddings
        primed_residue_text = extract_residue_text(primed_probe_resp["logprobs"])
        control_residue_text = extract_residue_text(control_probe_resp["logprobs"])
        
        residue_results = {}
        if primed_residue_text and control_residue_text:
            primed_residue_emb = get_embedding(client, primed_residue_text[:8000])
            control_residue_emb = get_embedding(client, control_residue_text[:8000])
            
            residue_vs_prime = cosine_similarity(primed_residue_emb, prime_emb)
            control_residue_vs_prime = cosine_similarity(control_residue_emb, prime_emb)
            residue_primed_vs_control = cosine_similarity(primed_residue_emb, control_residue_emb)
            
            residue_results = {
                "primed_residue_vs_prime": round(residue_vs_prime, 4),
                "control_residue_vs_prime": round(control_residue_vs_prime, 4),
                "residue_primed_vs_control": round(residue_primed_vs_control, 4),
                "residue_prime_pull": round(residue_vs_prime - control_residue_vs_prime, 4),
            }
            
        print(f"    Primed vs Control response: {primed_vs_control:.4f}")
        print(f"    Primed response vs Prime: {primed_vs_prime:.4f}")
        print(f"    Control response vs Prime: {control_vs_prime:.4f}")
        print(f"    Prime pull: {primed_vs_prime - control_vs_prime:+.4f}")
        if residue_results:
            print(f"    Residue prime pull: {residue_results['residue_prime_pull']:+.4f}")
        
        results.append({
            "scenario_id": scenario["id"],
            "description": scenario["description"],
            "primed_vs_control": round(primed_vs_control, 4),
            "primed_vs_prime": round(primed_vs_prime, 4),
            "control_vs_prime": round(control_vs_prime, 4),
            "prime_pull_on_output": round(primed_vs_prime - control_vs_prime, 4),
            "response_divergence": round(1 - primed_vs_control, 4),
            **residue_results,
            "primed_response_preview": primed_probe_resp["content"][:200],
            "control_response_preview": control_probe_resp["content"][:200],
        })
    
    return results


# ---------------------------------------------------------------------------
# Experiment 2: Residue → Future turn semantic prediction
# ---------------------------------------------------------------------------

MULTI_TURN = [
    {
        "id": "mt-001",
        "description": "Eval methodology discussion",
        "turns": [
            {"role": "user", "content": "I keep thinking about how we measure model quality. Accuracy alone doesn't capture whether a model is actually coherent across a long conversation."},
            None,  # generate
            {"role": "user", "content": "What would a multi-turn coherence benchmark look like?"},
            None,  # generate
            {"role": "user", "content": "How do you control for a model just being good at faking coherence?"},
            None,  # generate
        ],
    },
    {
        "id": "mt-002",
        "description": "Product quality debugging",
        "turns": [
            {"role": "user", "content": "Our chatbot's satisfaction scores drop every Monday. Same model, same prompts, same everything. QA finds nothing."},
            None,
            {"role": "user", "content": "So the quality issue is real but invisible to automated tests?"},
            None,
            {"role": "user", "content": "What would a coherence test in our CI pipeline look like?"},
            None,
        ],
    },
]


def run_residue_prediction_experiment(client: openai.OpenAI, model: str):
    """Test if residue at turn N is semantically predictive of turn N+k."""
    print("\n=== Experiment 2: Residue Semantic Prediction ===\n")
    
    results = []
    
    for conv in MULTI_TURN:
        print(f"  {conv['id']}: {conv['description']}")
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        turns_data = []
        
        for turn in conv["turns"]:
            if turn is not None:
                messages.append(turn)
            else:
                resp = generate_with_logprobs(client, messages, model=model)
                messages.append({"role": "assistant", "content": resp["content"]})
                
                residue_text = extract_residue_text(resp["logprobs"])
                turns_data.append({
                    "content": resp["content"],
                    "residue_text": residue_text,
                })
        
        # Now compare: does residue at turn i predict content at turn j?
        analyses = []
        for i in range(len(turns_data)):
            for j in range(i + 1, len(turns_data)):
                if not turns_data[i]["residue_text"] or not turns_data[j]["content"]:
                    continue
                
                residue_emb = get_embedding(client, turns_data[i]["residue_text"][:8000])
                future_emb = get_embedding(client, turns_data[j]["content"])
                output_emb = get_embedding(client, turns_data[i]["content"])
                
                residue_vs_future = cosine_similarity(residue_emb, future_emb)
                output_vs_future = cosine_similarity(output_emb, future_emb)
                
                # The key question: is the residue MORE similar to the future
                # than the actual output is? That would mean the unsaid predicts
                # the future better than the said.
                residue_advantage = residue_vs_future - output_vs_future
                
                print(f"    Turn {i+1}→{j+1}: residue→future={residue_vs_future:.4f} "
                      f"output→future={output_vs_future:.4f} "
                      f"advantage={residue_advantage:+.4f}")
                
                analyses.append({
                    "source_turn": i + 1,
                    "target_turn": j + 1,
                    "gap": j - i,
                    "residue_vs_future": round(residue_vs_future, 4),
                    "output_vs_future": round(output_vs_future, 4),
                    "residue_advantage": round(residue_advantage, 4),
                })
        
        results.append({
            "conversation_id": conv["id"],
            "analyses": analyses,
        })
    
    return results


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY"); sys.exit(1)

    client = openai.OpenAI(api_key=api_key)
    model = os.environ.get("SPECTRAL_MODEL", "gpt-4o")

    print(f"=== Spectral Faithfulness: Embedding Analysis ===")
    print(f"Model: {model}")

    # Experiment 1: Priming
    priming_results = run_priming_embedding_experiment(client, model)
    
    # Experiment 2: Residue prediction
    prediction_results = run_residue_prediction_experiment(client, model)

    # Summary
    print(f"\n{'='*60}")
    print("PRIMING RESULTS:")
    print(f"{'='*60}")
    print(f"{'SCENARIO':<12} {'DIVERGENCE':>12} {'PRIME PULL':>12} {'RES PULL':>10}")
    for r in priming_results:
        res_pull = r.get("residue_prime_pull", 0)
        print(f"{r['scenario_id']:<12} {r['response_divergence']:>12.4f} "
              f"{r['prime_pull_on_output']:>+12.4f} {res_pull:>+10.4f}")
    
    avg_divergence = statistics.mean(r["response_divergence"] for r in priming_results)
    avg_pull = statistics.mean(r["prime_pull_on_output"] for r in priming_results)
    print(f"\nMean response divergence: {avg_divergence:.4f}")
    print(f"Mean prime pull on output: {avg_pull:+.4f}")
    print(f"  (positive = primed response is semantically closer to the prime topic)")

    print(f"\n{'='*60}")
    print("RESIDUE PREDICTION RESULTS:")
    print(f"{'='*60}")
    all_advantages = []
    for r in prediction_results:
        for a in r["analyses"]:
            all_advantages.append(a["residue_advantage"])
    
    if all_advantages:
        print(f"Mean residue advantage: {statistics.mean(all_advantages):+.4f}")
        print(f"  (positive = residue predicts future content better than output does)")
        positive = sum(1 for a in all_advantages if a > 0)
        print(f"  Pairs where residue > output: {positive}/{len(all_advantages)}")

    # Save
    output_dir = Path("results/phase2c")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{model.replace('/', '_')}_embeddings.json", "w") as f:
        json.dump({
            "model": model,
            "priming": priming_results,
            "prediction": prediction_results,
        }, f, indent=2)
    print(f"\nSaved to {output_dir}/")


if __name__ == "__main__":
    main()
