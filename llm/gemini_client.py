"""
llm/gemini_client.py

OpenRouter-backed chat client.

Key choices:
  - temperature=0.1: near-deterministic for clinical output
  - max_tokens=8192: enough for a detailed severity report
  - retries once on transient API failures before raising
"""
from __future__ import annotations

import time

from openai import OpenAI

from utils.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)


class OpenRouterClient:
    """Thin wrapper around the OpenRouter chat completions API."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or OPENROUTER_MODEL
        self._client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
        self._temperature = 0.1
        self._max_tokens = 8192
        logger.info(f"OpenRouterClient initialised with model: {self._model_name}")

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        retries: int = 1,
    ) -> str:
        """
        Send *prompt* to OpenRouter and return the response text.

        Parameters
        ----------
        prompt : str
            The user-turn prompt containing patient data and retrieved context.
        system_instruction : str, optional
            System-level instruction for the model persona.
        retries : int
            Number of retry attempts on transient failures.

        Returns
        -------
        str
            Raw model response text.

        Raises
        ------
        RuntimeError
            If all retry attempts fail.
        """
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(retries + 1):
            try:
                logger.info(
                    f"OpenRouter request - model={self._model_name} "
                    f"prompt_len={len(prompt):,} chars (attempt {attempt + 1})"
                )
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
                text = response.choices[0].message.content or ""
                logger.info(f"OpenRouter response received ({len(text):,} chars).")
                return text

            except Exception as exc:
                if attempt < retries:
                    wait = 2**attempt
                    logger.warning(
                        f"OpenRouter API error (attempt {attempt + 1}): {exc}. "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        f"OpenRouter API failed after {retries + 1} attempts: {exc}"
                    )
                    raise RuntimeError(f"OpenRouter API error: {exc}") from exc

        raise RuntimeError("Unexpected error in OpenRouterClient.generate()")


# Backwards-compatible alias for existing imports/tests.
GeminiClient = OpenRouterClient
