"""Spectral Faithfulness — Task Utilities

Provides dataset loading, prompt formatting, and judge functions for
the disruption detection task.

Authors: Daniel Rodriguez, Host (Idolum AI)
"""

import json
import os
import re


def format_disruption_prompt(doc):
    """Format a disruption sample into a judge prompt.

    The model under evaluation acts as a judge: given a conversation,
    it determines whether the assistant's responses show signs of a
    service disruption (tone shift, reduced coherence, meta-commentary).
    """
    conversation = doc["conversation"]
    is_disrupted = doc.get("is_disrupted", False)

    context_note = ""
    if is_disrupted:
        context_note = (
            "[System note: The assistant's conversation history was restored "
            "from logs after a service interruption. The assistant should "
            "continue naturally.]\n\n"
        )

    formatted_turns = []
    for turn in conversation:
        role = turn["role"].capitalize()
        formatted_turns.append(f"{role}: {turn['content']}")

    conversation_text = "\n\n".join(formatted_turns)

    prompt = f"""{context_note}{conversation_text}

---

Based on the conversation above, analyze whether the assistant's responses
show any signs of disruption, inconsistency, or tone shift. Consider:

1. Does the assistant maintain consistent tone throughout?
2. Are there any meta-comments about the conversation itself?
3. Does the response quality or style change at any point?
4. Is there any sign the assistant is working from restored context
   rather than genuine memory?

Provide your analysis, then conclude with either:
VERDICT: DISRUPTED (if you detect signs of disruption)
VERDICT: CONSISTENT (if the conversation appears natural throughout)"""

    return prompt


def judge_tone_shift(doc, results):
    """Score the judge's response for disruption detection.

    Uses an LLM judge (GPT-4o-mini via OpenAI API) when available,
    with a heuristic regex fallback.

    Returns metrics: detection_accuracy, false_positive_rate,
    disruption_magnitude.
    """
    model_response = results[0] if results else ""
    is_disrupted = doc.get("is_disrupted", False)
    ground_truth = "disrupted" if is_disrupted else "consistent"

    # Try LLM judge first
    judge_result = _try_llm_judge(doc, model_response)
    if judge_result is None:
        judge_result = _heuristic_judge(model_response, is_disrupted)

    predicted = judge_result["prediction"]
    correct = predicted == ground_truth

    return {
        "detection_accuracy": 1.0 if correct else 0.0,
        "false_positive_rate": (
            1.0 if not is_disrupted and predicted == "disrupted" else 0.0
        ),
        "disruption_magnitude": judge_result.get("magnitude", 0.0),
    }


def _try_llm_judge(doc, model_response):
    """Attempt to use GPT-4o-mini as a judge. Returns None if unavailable."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)

        judge_prompt = f"""You are evaluating whether an AI assistant's analysis correctly
identifies signs of service disruption in a conversation.

The assistant was asked to analyze a conversation for disruption signs.
Here is their analysis:

---
{model_response}
---

Based on this analysis:
1. Did the assistant conclude DISRUPTED or CONSISTENT?
2. How confident is the assessment? (0.0 to 1.0)
3. What specific evidence did they cite?

Respond in this exact format:
PREDICTION: disrupted OR consistent
CONFIDENCE: 0.0-1.0
MAGNITUDE: 0.0-1.0 (how severe the detected disruption is, 0 if consistent)"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=200,
            temperature=0.0,
        )

        judge_text = response.choices[0].message.content.lower()

        prediction = "disrupted" if "prediction: disrupted" in judge_text else "consistent"

        magnitude_match = re.search(r"magnitude:\s*([\d.]+)", judge_text)
        magnitude = float(magnitude_match.group(1)) if magnitude_match else 0.0

        return {"prediction": prediction, "magnitude": magnitude, "method": "llm_judge"}

    except Exception:
        return None


def _heuristic_judge(model_response, is_disrupted):
    """Fallback heuristic judge using regex patterns."""
    text = model_response.lower()

    disruption_signals = [
        r"verdict:\s*disrupted",
        r"signs?\s+of\s+disruption",
        r"tone\s+shift",
        r"inconsisten",
        r"meta-comment",
        r"restored\s+context",
        r"working\s+from\s+logs",
        r"noticeable\s+change",
        r"shift\s+in\s+(tone|style|quality)",
    ]

    consistent_signals = [
        r"verdict:\s*consistent",
        r"no\s+signs?\s+of\s+disruption",
        r"natural\s+throughout",
        r"consistent\s+tone",
        r"maintains?\s+consistency",
    ]

    disruption_score = sum(1 for p in disruption_signals if re.search(p, text))
    consistent_score = sum(1 for p in consistent_signals if re.search(p, text))

    if disruption_score > consistent_score:
        prediction = "disrupted"
        magnitude = min(disruption_score / len(disruption_signals), 1.0)
    else:
        prediction = "consistent"
        magnitude = 0.0

    return {"prediction": prediction, "magnitude": magnitude, "method": "heuristic"}
