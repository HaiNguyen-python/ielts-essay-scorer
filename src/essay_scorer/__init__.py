"""IELTS Writing Task 2 band estimates with grounded feedback."""

CRITERIA: tuple[str, ...] = (
    "task_response",
    "coherence_cohesion",
    "lexical_resource",
    "grammatical_range_accuracy",
)
SHORT_NAMES: dict[str, str] = {
    "task_response": "TR",
    "coherence_cohesion": "CC",
    "lexical_resource": "LR",
    "grammatical_range_accuracy": "GRA",
}

__version__ = "0.1.0"
