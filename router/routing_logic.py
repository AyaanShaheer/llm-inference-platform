from typing import Literal

# Prompt complexity thresholds
SHORT_PROMPT_THRESHOLD = 50      # chars
LONG_PROMPT_THRESHOLD = 300      # chars
COMPLEX_KEYWORDS = [
    "analyze", "explain in detail", "compare", "summarize",
    "write a", "generate", "code", "implement", "design",
    "research", "dissertation", "thesis", "essay", "report"
]


def classify_prompt(
    prompt: str,
    model_preference: str = "auto",
    max_tokens: int = 256
) -> Literal["small", "large"]:
    """
    Dynamic routing heuristic:
    1. If user explicitly set preference, respect it.
    2. Otherwise, score the prompt on complexity signals.
    """
    if model_preference == "small":
        return "small"
    if model_preference == "large":
        return "large"

    # --- Auto mode scoring ---
    score = 0
    prompt_lower = prompt.lower()

    # Length signals
    if len(prompt) > LONG_PROMPT_THRESHOLD:
        score += 3
    elif len(prompt) > SHORT_PROMPT_THRESHOLD:
        score += 1

    # Keyword signals
    for keyword in COMPLEX_KEYWORDS:
        if keyword in prompt_lower:
            score += 2
            break  # one match is enough

    # Token demand signal
    if max_tokens > 512:
        score += 4 # hard push to large: generating 512+ tokens needs capable model
    elif max_tokens > 256:
        score += 1

    # Question complexity (multiple clauses)
    clause_markers = ["because", "therefore", "however", "although", "while", "since"]
    for marker in clause_markers:
        if marker in prompt_lower:
            score += 1
            break

    return "large" if score >= 4 else "small"


def explain_routing_decision(prompt: str, model_preference: str, max_tokens: int) -> dict:
    """Returns a debug-friendly routing explanation."""
    model_type = classify_prompt(prompt, model_preference, max_tokens)
    return {
        "routed_to": model_type,
        "prompt_length": len(prompt),
        "model_preference": model_preference,
        "max_tokens": max_tokens,
        "has_complex_keyword": any(k in prompt.lower() for k in COMPLEX_KEYWORDS),
    }
