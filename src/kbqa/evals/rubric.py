"""Rubric grading for answer correctness.

Spec 06B allows an LLM grader provided its model, prompt, and settings are
pinned and recorded, and provided its judgment is not treated as unversioned
ground truth. It also requires the same deterministic graders and thresholds
for both backends.

Both requirements are met by making the grader an injectable component with a
recorded identity:

- `LexicalRubricGrader` is the default. It is deterministic, needs no network,
  and is what the offline suite runs.
- `OpenAIRubricGrader` applies a fixed, versioned rubric with the pinned answer
  model. It is opt-in, and its model and prompt version are recorded on every
  record alongside its verdicts.

Both return the same `FactVerdict`, so a run's correctness numbers always carry
the identity of the grader that produced them.
"""

from dataclasses import dataclass, field
from typing import Protocol

from kbqa.models import ANSWER_MODEL


RUBRIC_PROMPT_VERSION = "rubric-v1"

RUBRIC_INSTRUCTIONS = """You grade whether an answer states a specific expected fact.

Reply with one line: SUPPORTED, CONTRADICTED, or MISSING, then a second line
giving a one-sentence justification quoting the relevant part of the answer.

SUPPORTED: the answer states the expected fact, including any specific
quantities, dates, or durations, allowing for paraphrase.
CONTRADICTED: the answer states something incompatible with the expected fact,
including a different quantity or a negation of it.
MISSING: the answer does not address the expected fact either way.

Judge only whether the expected fact is stated. Do not judge style, completeness,
or whether the answer cites a source. Do not reveal your reasoning process; give
only the verdict line and the one-sentence justification.
"""


@dataclass(frozen=True)
class FactVerdict:
    """One grader judgment about one expected fact."""

    verdict: str  # "supported" | "contradicted" | "missing"
    score: float
    explanation: str = ""

    @property
    def supported(self) -> bool:
        return self.verdict == "supported"


@dataclass(frozen=True)
class GraderIdentity:
    """What produced a set of verdicts, recorded with every result."""

    name: str
    version: str
    model: str | None = None
    prompt_version: str | None = None
    settings: dict = field(default_factory=dict)

    def as_record(self) -> dict:
        return {
            "grader_name": self.name,
            "grader_version": self.version,
            "grader_model": self.model,
            "grader_prompt_version": self.prompt_version,
            "grader_settings": dict(self.settings),
        }


class RubricGrader(Protocol):
    identity: GraderIdentity

    def grade_fact(self, question: str, expected_fact: str, answer: str) -> FactVerdict:
        ...


class LexicalRubricGrader:
    """Deterministic token-overlap grader with numeric and negation gates.

    This is a lexical proxy, not comprehension. It cannot credit a paraphrase
    that shares no vocabulary with the reference, which is why the numbers it
    produces are always recorded with its identity.
    """

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold
        self.identity = GraderIdentity(
            name="lexical",
            version="deterministic-v2",
            settings={"fact_token_recall_threshold": threshold},
        )

    def grade_fact(self, question: str, expected_fact: str, answer: str) -> FactVerdict:
        from kbqa.evals.metrics import _fact_coverage, _negated

        coverage = _fact_coverage(expected_fact, answer)
        if coverage >= self.threshold:
            return FactVerdict("supported", coverage, "token recall above threshold")
        if _negated(answer) != _negated(expected_fact):
            return FactVerdict("contradicted", coverage, "negation disagrees")
        return FactVerdict("missing", coverage, "token recall below threshold")


class OpenAIRubricGrader:
    """Rubric grader using the pinned answer model.

    Opt-in: it makes a model call per expected fact. Its verdicts are recorded
    with the model and prompt version that produced them so a later run can tell
    whether a correctness change came from the system or from the grader.
    """

    def __init__(
        self,
        model: str = ANSWER_MODEL,
        api_key: str | None = None,
        client=None,
        max_output_tokens: int = 200,
    ) -> None:
        if model != ANSWER_MODEL:
            raise ValueError(f"The shared Q&A contract requires {ANSWER_MODEL}")
        self.model = model
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens
        self._client = client
        self.identity = GraderIdentity(
            name="openai-rubric",
            version="rubric-v1",
            model=model,
            prompt_version=RUBRIC_PROMPT_VERSION,
            settings={"max_output_tokens": max_output_tokens},
        )

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def grade_fact(self, question: str, expected_fact: str, answer: str) -> FactVerdict:
        response = self.client.responses.create(
            model=self.model,
            instructions=RUBRIC_INSTRUCTIONS,
            input=(
                f"QUESTION\n{question}\n\n---\n\n"
                f"EXPECTED FACT\n{expected_fact}\n\n---\n\n"
                f"ANSWER\n{answer}"
            ),
            max_output_tokens=self.max_output_tokens,
        )
        return parse_rubric_reply(response.output_text)


def parse_rubric_reply(text: str) -> FactVerdict:
    """Parse a rubric reply into a verdict.

    An unrecognized reply is 'missing' with a score of 0 rather than an
    exception: a malformed grader response must not be silently credited as
    correct, and must not abort a whole evaluation run.
    """
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return FactVerdict("missing", 0.0, "empty grader reply")
    head = lines[0].strip().rstrip(".").upper()
    explanation = lines[1] if len(lines) > 1 else ""
    if head.startswith("SUPPORTED"):
        return FactVerdict("supported", 1.0, explanation)
    if head.startswith("CONTRADICTED"):
        return FactVerdict("contradicted", 0.0, explanation)
    if head.startswith("MISSING"):
        return FactVerdict("missing", 0.0, explanation)
    return FactVerdict("missing", 0.0, f"unparsed grader reply: {lines[0][:80]}")
