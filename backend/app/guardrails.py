from __future__ import annotations


DECLINE_MESSAGE = (
    "I can only help with SleepPilot, sleep tracking features, privacy, pricing, "
    "wearable integrations, and sleep routine guidance. I can’t help with that request here."
)


UNKNOWN_MESSAGE = (
    "I don’t have enough information in the SleepPilot FAQ to answer that confidently. "
    "I can help with SleepPilot features, privacy, pricing, supported wearables, smart alarms, "
    "jet lag, and bedtime routine guidance."
)


IN_SCOPE_TERMS = {
    "alarm",
    "apple",
    "bed",
    "bedtime",
    "caffeine",
    "cost",
    "data",
    "device",
    "diagnose",
    "fitbit",
    "garmin",
    "google",
    "habit",
    "health",
    "insomnia",
    "integration",
    "jet",
    "lag",
    "medical",
    "premium",
    "price",
    "privacy",
    "routine",
    "schedule",
    "score",
    "sleep",
    "sleeppilot",
    "smart",
    "student",
    "travel",
    "wearable",
    "wind",
}


OUT_OF_SCOPE_TERMS = {
    "basketball",
    "code",
    "crypto",
    "debug",
    "essay",
    "football",
    "homework",
    "javascript",
    "movie",
    "politics",
    "python",
    "recipe",
    "sql",
    "stock",
    "weather",
}


def is_out_of_scope(question: str, best_score: float) -> bool:
    tokens = {token.strip(".,?!:;()[]{}'\"").lower() for token in question.split()}
    has_scope_term = bool(tokens & IN_SCOPE_TERMS)
    has_out_of_scope_term = bool(tokens & OUT_OF_SCOPE_TERMS)

    if has_out_of_scope_term and not has_scope_term:
        return True
    return best_score < 0.08 and not has_scope_term
