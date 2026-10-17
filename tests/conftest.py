import pytest

from essay_scorer.evaluate import Dataset, fit_model
from essay_scorer.service import Scorer
from essay_scorer.synthetic import GeneratorConfig, generate_essays

PROMPT = (
    "Some people believe that university education should be free for all students. "
    "To what extent do you agree or disagree?"
)
ESSAY = """Nowadays, many people are worried about free university education. In my opinion, \
the advantages of free university education clearly outweigh the disadvantages.

Firstly, free university education has several clear advantages. Students from poor families \
get the same chance as rich students. Furthermore, a country with more graduates usually has \
a stronger economy, which is a crucial advantage.

On the other hand, free university education also has some drawbacks. Taxpayers have to pay \
for the tuition of every student. However, this does not mean that the idea is bad.

In conclusion, I believe that free university education does more good than harm."""


@pytest.fixture
def essay() -> str:
    return ESSAY


@pytest.fixture
def prompt() -> str:
    return PROMPT


@pytest.fixture(scope="session")
def ds() -> Dataset:
    """A small synthetic set: 640 essays, 40 per prompt."""
    return Dataset.from_frame(generate_essays(GeneratorConfig(n_essays=640, seed=3)))


@pytest.fixture(scope="session")
def scorer(ds: Dataset) -> Scorer:
    return Scorer(fit_model(ds, "ridge"))
