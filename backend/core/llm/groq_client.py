"""
VidIntel AI — Groq LLM Client
Wraps the Groq SDK for chat completions using Llama 3.
All agent prompts flow through here.
"""

from __future__ import annotations

import json
from typing import List, Optional
from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL, GROQ_MAX_TOKENS, GROQ_TEMPERATURE


_client: Optional[Groq] = None


def get_groq_client() -> Groq:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def chat_complete(
    messages: List[dict],
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    json_mode: bool = False,
) -> str:
    """
    Send messages to Groq and return the text response.

    Args:
        messages:    List of {"role": "user"|"system"|"assistant", "content": str}
        model:       Override default model.
        max_tokens:  Override max tokens.
        temperature: Override temperature.
        json_mode:   If True, request JSON output format.

    Returns:
        The assistant's reply as a string.
    """
    client = get_groq_client()
    kwargs = {
        "model": model or GROQ_MODEL,
        "messages": messages,
        "max_tokens": max_tokens or GROQ_MAX_TOKENS,
        "temperature": temperature or GROQ_TEMPERATURE,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def simple_prompt(system: str, user: str, **kwargs) -> str:
    """Convenience wrapper for single-turn prompts."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return chat_complete(messages, **kwargs)


def parse_json_response(raw: str) -> dict:
    """Safely parse a JSON response from the LLM."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON block from markdown
        import re
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"Could not parse JSON from LLM response:\n{raw[:300]}")
