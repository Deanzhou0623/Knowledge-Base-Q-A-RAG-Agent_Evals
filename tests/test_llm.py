from types import SimpleNamespace

import pytest

from kbqa.llm import OpenAIAnswerGenerator


class FakeResponses:
    def create(self, **kwargs):
        return SimpleNamespace(
            output_text="  Grounded answer.  ",
            usage=SimpleNamespace(
                input_tokens=21,
                output_tokens=7,
                total_tokens=28,
            ),
        )


def test_openai_generator_returns_answer_and_token_usage():
    generator = OpenAIAnswerGenerator("gpt-5.6-sol", api_key="test")
    generator._client = SimpleNamespace(responses=FakeResponses())

    result = generator.grounded("question and context")

    assert result.text == "Grounded answer."
    assert result.token_usage.input_tokens == 21
    assert result.token_usage.output_tokens == 7
    assert result.token_usage.total_tokens == 28


def test_openai_generator_rejects_an_unapproved_model():
    with pytest.raises(ValueError, match="gpt-5.6-sol"):
        OpenAIAnswerGenerator("another-model")
