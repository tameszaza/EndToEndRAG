class ConstantTestEmbedder:
    """Minimal test double that avoids external embedding API calls."""

    dimensions = 1
    namespace = "test:constant:1"

    def embed(
        self,
        text: str,
        task_type: str | None = None,
        title: str | None = None,
    ) -> list[float]:
        return [0.0]
