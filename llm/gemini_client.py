"""
llm/gemini_client.py
─────────────────────
Wraps the Groq SDK for inference calls.

Key choices:
  • temperature=0.1  — near-deterministic for clinical output
  • max_tokens=8192 — enough for a detailed severity report
  • Retries once on transient API failures before raising
"""
from __future__ import annotations

import time

from groq import Groq

from utils.config import GROQ_API_KEY, GROQ_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)


class GeminiClient:
    """Thin wrapper around Groq API client."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or GROQ_MODEL
        self._client = Groq(api_key=GROQ_API_KEY)
        self._temperature = 0.1
        self._max_tokens = 8192
        logger.info(f"GeminiClient initialised with model: {self._model_name}")

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        retries: int = 1,
    ) -> str:
        """
        Send *prompt* to Groq and return the response text.

        Parameters
        ----------
        prompt : str
            The user-turn prompt (contains patient data + retrieved context).
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
                    f"Groq request — model={self._model_name} "
                    f"prompt_len={len(prompt):,} chars (attempt {attempt + 1})"
                )
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
                text = response.choices[0].message.content
                logger.info(f"Groq response received ({len(text):,} chars).")
                return text

            except Exception as exc:
                if attempt < retries:
                    wait = 2 ** attempt  # exponential back-off
                    logger.warning(
                        f"Groq API error (attempt {attempt + 1}): {exc}. "
                        f"Retrying in {wait}s…"
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"Groq API failed after {retries + 1} attempts: {exc}")
                    raise RuntimeError(f"Groq API error: {exc}") from exc

        # Unreachable, but satisfies type-checkers
        raise RuntimeError("Unexpected error in GeminiClient.generate()")
