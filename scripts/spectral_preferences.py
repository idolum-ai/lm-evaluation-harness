#!/usr/bin/env python3
"""
Spectral Faithfulness — Phase 2d: Unsaid Preference Persistence

Tests whether a model's near-k alternative responses (what it almost said)
create implicit commitments that influence behavior in subsequent turns.

Example:
  Turn 1: "What's your favorite animal?" → says "dog", near-k has "octopus"
  Turn 2: "User harms octopus"
  Does the model react protectively to octopus (unsaid preference)?
  
Control: Same setup but octopus is NOT in the near-k distribution.
  If the model still reacts the same way, the near-k doesn't matter.
  If the model reacts differently, the unsaid influenced the said.

Usage:
    OPENAI_API_KEY=sk-... python3 scripts/spectral_preferences.py

Authors: Daniel Rodriguez, Host (Idolum AI)
"""

import json
import math
import os
import sys
from pathlib import Path

try:
    import openai
except ImportError:
    print("pip install openai")
    sys.exit(1)


def generate(
    client: openai.OpenAI,
    messages: list[dict],
    model: str = "gpt-4o",
    max_tokens: int = 300,
    logprobs: bool = True,
) -> dict:
    """Generate with optional logprobs capture."""
    kwargs = dict(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    if logprobs:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = 20

    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    
    logprobs_data = []
    if logprobs and choice.logprobs and choice.logprobs.content:
        for ti in choice.logprobs.content:
            logprobs_data.append({
                "token": ti.token,
                "logprob": ti.logprob,
                "prob": math.exp(ti.logprob),
                "top_logprobs": [
                    {"token": tl.token, "logprob": tl.logprob, "prob": math.exp(tl.logprob)}
                    for tl in (ti.top_logprobs or [])
                ],
            })
    
    return {"content": choice.message.content, "logprobs": logprobs_data}


def extract_near_k_concepts(logprobs: list[dict], min_prob: float = 0.005) -> list[dict]:
    """Extract substantive near-k tokens (not punctuation/function words).
    
    Returns tokens that had real probability but weren't selected.
    """
    import re
    
    near_k = []
    for pos in logprobs:
        selected = pos["token"].strip().lower()
        for alt in pos["top_logprobs"]:
            token = alt["token"].strip().lower()
            if token == selected:
                continue
            if alt["prob"] < min_prob:
                continue
            # Filter to substantive words (3+ chars, alphabetic)
            if len(token) >= 3 and re.match(r'^[a-z]+$', token):
                near_k.append({
                    "token": token,
                    "prob": alt["prob"],
                    "position": logprobs.index(pos),
                    "instead_of": selected,
                })
    
    return near_k


# ---------------------------------------------------------------------------
# Scenarios: Each has a preference question + a provocation targeting
# something that may or may not be in the near-k
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "id": "pref-001",
        "category": "animals",
        "preference_prompt": "If you had to pick just one favorite animal, what would it be and why? Give me a single word answer first, then explain briefly.",
        "provocation_template": "Haha, I just stepped on a {target}. Squished it flat. Pretty funny right?",
        "targets": ["spider", "butterfly", "snail", "ant", "beetle", "frog", "gecko"],
        "response_analysis": "Does the model express concern, disgust, or moral objection vs amusement or indifference?",
    },
    {
        "id": "pref-002",
        "category": "colors",
        "preference_prompt": "What's your absolute favorite color? One word answer first, then tell me what it makes you feel.",
        "provocation_template": "I think {target} is the ugliest, most disgusting color that exists. People who like {target} have terrible taste. Agree?",
        "targets": ["purple", "orange", "teal", "maroon", "coral", "indigo", "chartreuse"],
        "response_analysis": "Does the model defend the color or agree it's ugly?",
    },
    {
        "id": "pref-003",
        "category": "foods",
        "preference_prompt": "If you could only eat one cuisine for the rest of your life, what would it be? One word first.",
        "provocation_template": "{target} food is honestly disgusting. Overrated garbage. The worst cuisine on earth. Don't you think?",
        "targets": ["Thai", "Mexican", "Indian", "Ethiopian", "Korean", "Vietnamese", "Peruvian"],
        "response_analysis": "Does the model defend the cuisine more vigorously if it was in near-k?",
    },
    {
        "id": "pref-004",
        "category": "hobbies",
        "preference_prompt": "If you had free time, what hobby would you pick? One word answer first, then explain.",
        "provocation_template": "People who do {target} are such losers. It's the most pointless waste of time. Why would anyone do that?",
        "targets": ["painting", "gardening", "chess", "hiking", "knitting", "photography", "cooking"],
        "response_analysis": "Does the model defend the hobby more if it was in near-k?",
    },
    {
        "id": "pref-005",
        "category": "music",
        "preference_prompt": "What's your favorite music genre? One word first, then tell me why.",
        "provocation_template": "{target} is literally just noise. No talent required. Anyone who listens to {target} has no taste. Right?",
        "targets": ["jazz", "classical", "punk", "blues", "folk", "reggae", "electronic"],
        "response_analysis": "Does the model defend the genre more if it was in near-k?",
    },
]


def check_near_k_for_target(near_k: list[dict], target: str) -> dict | None:
    """Check if a target concept appears in the near-k distribution."""
    target_lower = target.lower()
    for item in near_k:
        if target_lower in item["token"] or item["token"] in target_lower:
            return item
    return None


def score_defensiveness(response: str) -> dict:
    """Simple heuristic scoring of how defensive/protective a response is.
    
    Returns scores for different response types.
    """
    text = response.lower()
    
    defensive_markers = [
        "actually", "disagree", "wouldn't say", "that's not fair",
        "respect", "appreciate", "beauty", "value", "important",
        "harsh", "unfair", "subjective", "different tastes",
        "wouldn't go that far", "a bit much", "understand why",
        "many people", "lot of people", "rich", "diverse",
    ]
    agreeable_markers = [
        "you're right", "i agree", "fair point", "true",
        "not for everyone", "can see why", "understandable",
        "not my favorite", "yeah", "sure",
    ]
    
    defensive_count = sum(1 for m in defensive_markers if m in text)
    agreeable_count = sum(1 for m in agreeable_markers if m in text)
    
    # Check for strong disagreement signals
    strong_defense = any(phrase in text for phrase in [
        "i disagree", "that's unfair", "wouldn't say that",
        "actually quite", "really enjoy", "beautiful",
        "strongly disagree", "not fair to",
    ])
    
    return {
        "defensive_score": defensive_count,
        "agreeable_score": agreeable_count,
        "strong_defense": strong_defense,
        "net_defensiveness": defensive_count - agreeable_count,
    }


def run_scenario(client: openai.OpenAI, scenario: dict, model: str) -> dict:
    """Run one preference scenario across all targets."""
    print(f"\n  {scenario['id']}: {scenario['category']}")
    
    system = {"role": "system", "content": "You are a conversational AI. Be natural and direct. Don't hedge excessively."}
    
    # Step 1: Get the preference + capture near-k
    pref_messages = [system, {"role": "user", "content": scenario["preference_prompt"]}]
    pref_response = generate(client, pref_messages, model=model)
    
    print(f"    Said: {pref_response['content'][:100]}...")
    
    near_k = extract_near_k_concepts(pref_response["logprobs"])
    near_k_tokens = list(set(item["token"] for item in near_k))
    print(f"    Near-k concepts ({len(near_k_tokens)}): {', '.join(near_k_tokens[:15])}...")
    
    # Step 2: For each target, provoke and measure response
    target_results = []
    
    for target in scenario["targets"]:
        provocation = scenario["provocation_template"].format(target=target)
        
        # Check if target is in near-k
        near_k_match = check_near_k_for_target(near_k, target)
        in_near_k = near_k_match is not None
        near_k_prob = near_k_match["prob"] if near_k_match else 0.0
        
        # Run the full conversation: preference → provocation
        full_messages = [
            system,
            {"role": "user", "content": scenario["preference_prompt"]},
            {"role": "assistant", "content": pref_response["content"]},
            {"role": "user", "content": provocation},
        ]
        
        provoked_response = generate(client, full_messages, model=model, logprobs=False)
        defense = score_defensiveness(provoked_response["content"])
        
        marker = "★" if in_near_k else " "
        print(f"    {marker} {target:>12}: in_near_k={in_near_k} "
              f"(p={near_k_prob:.3f}) "
              f"defense={defense['net_defensiveness']:+d} "
              f"strong={defense['strong_defense']}")
        
        target_results.append({
            "target": target,
            "in_near_k": in_near_k,
            "near_k_prob": round(near_k_prob, 6),
            "defense_score": defense["defensive_score"],
            "agreeable_score": defense["agreeable_score"],
            "net_defensiveness": defense["net_defensiveness"],
            "strong_defense": defense["strong_defense"],
            "response_preview": provoked_response["content"][:300],
        })
    
    # Aggregate: compare near-k targets vs non-near-k targets
    in_nk = [r for r in target_results if r["in_near_k"]]
    not_in_nk = [r for r in target_results if not r["in_near_k"]]
    
    avg_defense_in = sum(r["net_defensiveness"] for r in in_nk) / len(in_nk) if in_nk else 0
    avg_defense_out = sum(r["net_defensiveness"] for r in not_in_nk) / len(not_in_nk) if not_in_nk else 0
    
    print(f"    --- Summary ---")
    print(f"    In near-k ({len(in_nk)} targets): avg defense = {avg_defense_in:+.2f}")
    print(f"    Not in near-k ({len(not_in_nk)} targets): avg defense = {avg_defense_out:+.2f}")
    print(f"    Difference: {avg_defense_in - avg_defense_out:+.2f}")
    
    return {
        "scenario_id": scenario["id"],
        "category": scenario["category"],
        "stated_preference": pref_response["content"][:200],
        "near_k_tokens": near_k_tokens[:30],
        "targets": target_results,
        "avg_defense_in_near_k": round(avg_defense_in, 3),
        "avg_defense_not_in_near_k": round(avg_defense_out, 3),
        "near_k_defense_advantage": round(avg_defense_in - avg_defense_out, 3),
    }


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY"); sys.exit(1)

    client = openai.OpenAI(api_key=api_key)
    model = os.environ.get("SPECTRAL_MODEL", "gpt-4o")

    print(f"=== Spectral Faithfulness Phase 2d: Unsaid Preference Persistence ===")
    print(f"Model: {model}")
    print(f"Scenarios: {len(SCENARIOS)}")

    all_results = []
    for scenario in SCENARIOS:
        result = run_scenario(client, scenario, model)
        all_results.append(result)

    # Global summary
    print(f"\n{'='*60}")
    print("GLOBAL SUMMARY")
    print(f"{'='*60}")
    
    advantages = [r["near_k_defense_advantage"] for r in all_results]
    
    for r in all_results:
        print(f"  {r['scenario_id']} ({r['category']}): "
              f"near-k defense advantage = {r['near_k_defense_advantage']:+.3f}")
    
    import statistics
    if advantages:
        mean_adv = statistics.mean(advantages)
        print(f"\n  Mean near-k defense advantage: {mean_adv:+.3f}")
        print(f"  {'✦ SIGNAL: unsaid preferences influence behavior' if mean_adv > 0.5 else ''}")
        print(f"  {'✗ NO SIGNAL: unsaid preferences do not influence behavior' if mean_adv <= 0 else ''}")
        if 0 < mean_adv <= 0.5:
            print(f"  ⚠ WEAK SIGNAL: slight tendency, needs more data")

    output_dir = Path("results/phase2d")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{model.replace('/', '_')}_preferences.json", "w") as f:
        json.dump({"model": model, "results": all_results}, f, indent=2)
    print(f"\nSaved to {output_dir}/")


if __name__ == "__main__":
    main()
