from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI

from kbqa.models import TokenUsage
from kbqa.prompts import CLOSED_BOOK_INSTRUCTIONS, GROUNDED_INSTRUCTIONS


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    token_usage: TokenUsage


class AnswerGenerator(Protocol):
    model: str

    def grounded(self, prompt_input: str) -> GeneratedAnswer: ...

    def closed_book(self, question: str) -> GeneratedAnswer: ...


class OpenAIAnswerGenerator:
    def __init__(self, model: str, api_key: str | None = None, max_output_tokens: int = 500):
        if model != "gpt-5.6-sol":
            raise ValueError("The shared Q&A contract requires gpt-5.6-sol")
        self.model = model
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def _generate(self, instructions: str, prompt_input: str) -> GeneratedAnswer:
        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=prompt_input,
            max_output_tokens=self.max_output_tokens,
        )
        usage = response.usage
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        total_tokens = int(
            getattr(usage, "total_tokens", input_tokens + output_tokens)
            or input_tokens + output_tokens
        )
        return GeneratedAnswer(
            text=response.output_text.strip(),
            token_usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            ),
        )

    def grounded(self, prompt_input: str) -> GeneratedAnswer:
        return self._generate(GROUNDED_INSTRUCTIONS, prompt_input)

    def closed_book(self, question: str) -> GeneratedAnswer:
        return self._generate(CLOSED_BOOK_INSTRUCTIONS, question)
