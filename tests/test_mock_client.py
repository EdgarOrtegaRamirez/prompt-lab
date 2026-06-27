"""Tests for mock API client."""

from __future__ import annotations

import asyncio

import pytest

from prompts_lab.mock_client import MockAPIClient
from prompts_lab.models import ModelInfo, ModelProvider


@pytest.fixture
def client():
    return MockAPIClient(seed=42)


@pytest.fixture
def sample_model():
    return ModelInfo(id="gpt-4o", provider=ModelProvider.OPENAI)


class TestMockAPIClient:
    @pytest.mark.asyncio
    async def test_chat_returns_response(self, client, sample_model):
        messages = [{"role": "user", "content": "Hello world"}]
        response = await client.chat(messages, sample_model)
        assert response is not None
        assert response.model == "gpt-4o"
        assert response.provider == "openai"
        assert len(response.content) > 0
        assert response.prompt_tokens > 0
        assert response.completion_tokens > 0
        assert response.error is None

    @pytest.mark.asyncio
    async def test_chat_with_system_message(self, client, sample_model):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is AI?"},
        ]
        response = await client.chat(messages, sample_model)
        assert response is not None
        assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_call_count(self, client, sample_model):
        assert client.call_count == 0
        messages = [{"role": "user", "content": "test"}]
        await client.chat(messages, sample_model)
        assert client.call_count == 1
        await client.chat(messages, sample_model)
        assert client.call_count == 2

    @pytest.mark.asyncio
    async def test_reset_counter(self, client, sample_model):
        messages = [{"role": "user", "content": "test"}]
        await client.chat(messages, sample_model)
        assert client.call_count == 1
        client.reset()
        assert client.call_count == 0

    @pytest.mark.asyncio
    async def test_response_has_time(self, client, sample_model):
        messages = [{"role": "user", "content": "test"}]
        response = await client.chat(messages, sample_model)
        assert response.response_time_ms > 0

    @pytest.mark.asyncio
    async def test_different_models(self, client):
        for model_id, provider in [
            ("gpt-4o", ModelProvider.OPENAI),
            ("claude-sonnet-4-20250514", ModelProvider.ANTHROPIC),
            ("gemini-2.0-flash", ModelProvider.GOOGLE),
        ]:
            model = ModelInfo(id=model_id, provider=provider)
            response = await client.chat([{"role": "user", "content": "test"}], model)
            assert response.model == model_id
            assert response.provider == provider.value
