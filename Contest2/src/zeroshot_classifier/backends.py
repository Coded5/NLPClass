from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


MAX_RESPONSE_BYTES = 1024 * 1024
MAX_OUTPUT_TOKENS = 256


class BackendError(RuntimeError):
    pass


class TransientBackendError(BackendError):
    pass


class PermanentBackendError(BackendError):
    pass


@dataclass(frozen=True)
class BackendSettings:
    backend: str = 'ollama'
    model: str = 'qwen2.5:7b'
    base_url: str | None = None
    api_key_env: str = 'OPENAI_API_KEY'
    timeout: float = 120.0
    temperature: float = 0.0


class LLMBackend(ABC):
    @abstractmethod
    def complete(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> str:
        raise NotImplementedError


class OllamaBackend(LLMBackend):
    def __init__(self, settings: BackendSettings):
        self.settings = settings
        base_url = settings.base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.endpoint = f'{base_url.rstrip("/")}/api/chat'

    def complete(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> str:
        payload = {
            'model': self.settings.model,
            'messages': messages,
            'stream': False,
            'format': schema,
            'options': {
                'temperature': self.settings.temperature,
                'num_predict': MAX_OUTPUT_TOKENS,
            },
        }
        response = _post_json(self.endpoint, payload, self.settings.timeout)
        try:
            content = response['message']['content']
        except (KeyError, TypeError) as error:
            raise PermanentBackendError('Ollama response is missing message.content') from error
        if not isinstance(content, str):
            raise PermanentBackendError('Ollama message.content is not a string')
        return content


class OpenAICompatibleBackend(LLMBackend):
    def __init__(self, settings: BackendSettings):
        self.settings = settings
        base_url = settings.base_url or os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        self.endpoint = f'{base_url.rstrip("/")}/chat/completions'
        self.api_key = os.getenv(settings.api_key_env)

    def complete(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> str:
        payload = {
            'model': self.settings.model,
            'messages': messages,
            'temperature': self.settings.temperature,
            'max_tokens': MAX_OUTPUT_TOKENS,
            'response_format': {
                'type': 'json_schema',
                'json_schema': {
                    'name': 'aspect_sentiment_predictions',
                    'strict': True,
                    'schema': schema,
                },
            },
        }
        headers = {'Authorization': f'Bearer {self.api_key}'} if self.api_key else None
        response = _post_json(self.endpoint, payload, self.settings.timeout, headers=headers)
        try:
            content = response['choices'][0]['message']['content']
        except (IndexError, KeyError, TypeError) as error:
            raise PermanentBackendError('OpenAI-compatible response is missing choices[0].message.content') from error
        if not isinstance(content, str):
            raise PermanentBackendError('OpenAI-compatible message.content is not a string')
        return content


def create_backend(settings: BackendSettings) -> LLMBackend:
    if settings.backend == 'ollama':
        return OllamaBackend(settings)
    if settings.backend == 'openai-compatible':
        return OpenAICompatibleBackend(settings)
    raise ValueError(f'Unknown backend: {settings.backend}')


def _post_json(
    url: str,
    payload: dict[str, Any],
    timeout: float,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    request_headers = {'Content-Type': 'application/json', **(headers or {})}
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers=request_headers,
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_bytes = response.read(MAX_RESPONSE_BYTES + 1)
            if len(response_bytes) > MAX_RESPONSE_BYTES:
                raise PermanentBackendError(
                    f'LLM backend response exceeded {MAX_RESPONSE_BYTES} bytes'
                )
            response_body = response_bytes.decode('utf-8')
    except urllib.error.HTTPError as error:
        body = error.read().decode('utf-8', errors='replace')
        message = f'HTTP {error.code} from LLM backend: {body[:500]}'
        if error.code in {408, 409, 425, 429} or error.code >= 500:
            raise TransientBackendError(message) from error
        raise PermanentBackendError(message) from error
    except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError) as error:
        raise TransientBackendError(f'Could not reach LLM backend: {error}') from error

    try:
        decoded = json.loads(response_body)
    except json.JSONDecodeError as error:
        raise TransientBackendError('LLM backend returned invalid JSON') from error
    if not isinstance(decoded, dict):
        raise PermanentBackendError('LLM backend response must be a JSON object')
    return decoded
