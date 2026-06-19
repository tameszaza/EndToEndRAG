class CosineTestEmbedder:
    """Deterministic test double that makes retrieval depend on cosine only."""

    dimensions = 15
    namespace = "test:cosine:15"

    def embed(
        self,
        text: str,
        task_type: str | None = None,
        title: str | None = None,
    ) -> list[float]:
        lookup_text = title if task_type == "RETRIEVAL_DOCUMENT" and title else text
        normalized = lookup_text.lower()
        index = self._match_index(normalized)
        vector = [0.0] * self.dimensions
        if index is not None:
            vector[index] = 1.0
        return vector

    def _match_index(self, text: str) -> int | None:
        patterns = [
            ("faq-001", ["what is sleeppilot", "what is this", "what is the product"]),
            ("faq-002", ["improve my sleep", "improve sleep"]),
            ("faq-003", ["medical device", "diagnose", "insomnia", "sleep disorders"]),
            ("faq-004", ["what data", "collect"]),
            ("faq-005", ["without a wearable", "without wearable"]),
            (
                "faq-006",
                ["which wearable devices", "garmin", "fitbit", "wearable platforms", "support wearable"],
            ),
            ("faq-007", ["protect my privacy", "privacy", "sell data"]),
            ("faq-008", ["jet lag", "travel"]),
            ("faq-009", ["sleep score"]),
            ("faq-010", ["smart alarm"]),
            ("faq-011", ["bedtime routine", "go to bed", "when should i sleep"]),
            ("faq-012", ["students", "exams"]),
            ("faq-013", ["sleep too late", "sleep late", "sleep earlier"]),
            ("faq-014", ["premium cost", "how much", "pricing", "free plan"]),
            ("faq-015", ["what should i do", "cannot answer", "not enough information"]),
        ]
        for index, (_, terms) in enumerate(patterns):
            if any(term in text for term in terms):
                return index
        return None
