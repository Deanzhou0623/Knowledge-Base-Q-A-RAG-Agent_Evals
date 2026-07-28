import os

import pytest

from kbqa.llm import OpenAIAnswerGenerator


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("KBQA_RUN_LIVE_EVAL_TEST") != "1" or not os.getenv("OPENAI_API_KEY"),
    reason="requires KBQA_RUN_LIVE_EVAL_TEST=1 and OPENAI_API_KEY",
)
def test_pinned_answer_model_live_smoke():
    generator = OpenAIAnswerGenerator(
        "gpt-5.6-sol",
        api_key=os.environ["OPENAI_API_KEY"],
        max_output_tokens=20,
    )
    response = generator.closed_book("Reply with the single word: ready")
    assert response.text
    assert response.token_usage.total_tokens > 0
