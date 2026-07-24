from typing import Protocol

from openai import OpenAI

from kbqa.prompts import CLOSED_BOOK_INSTRUCTIONS, GROUNDED_INSTRUCTIONS


class AnswerGenerator(Protocol):
    model: str

    def grounded(self, prompt_input: str) -> str: ...

    def closed_book(self, question: str) -> str: ...


class OpenAIAnswerGenerator:
    def __init__(self, model: str, api_key: str | None = None, max_output_tokens: int = 500):
        self.model = model
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def _generate(self, instructions: str, prompt_input: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=prompt_input,
            max_output_tokens=self.max_output_tokens,
        )
        return response.output_text.strip()

    def grounded(self, prompt_input: str) -> str:
        return self._generate(GROUNDED_INSTRUCTIONS, prompt_input)

    def closed_book(self, question: str) -> str:
        return self._generate(CLOSED_BOOK_INSTRUCTIONS, question)
