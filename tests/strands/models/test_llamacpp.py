"""Unit tests for llama.cpp model provider."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel

from strands.types.content import ContentBlock, Message
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException
from strands.models.llamacpp import LlamaCppModel, LlamaCppError, LlamaCppContextOverflowError


class TestLlamaCppModel:
    """Test suite for LlamaCppModel."""
    
    def test_init_default_config(self) -> None:
        """Test initialization with default configuration."""
        model = LlamaCppModel()
        
        assert model.config["model_id"] == "default"
        assert isinstance(model.client, AsyncOpenAI)
        # Check that base_url was set correctly
        assert model.client.base_url == "http://localhost:8080/v1/"
    
    def test_init_custom_config(self) -> None:
        """Test initialization with custom configuration."""
        model = LlamaCppModel(
            base_url="http://example.com:8081/v1",
            model_id="llama-3-8b",
            params={"temperature": 0.7, "max_tokens": 100},
        )
        
        assert model.config["model_id"] == "llama-3-8b"
        assert model.config["params"]["temperature"] == 0.7
        assert model.config["params"]["max_tokens"] == 100
        assert model.client.base_url == "http://example.com:8081/v1/"
    
    def test_format_request_basic(self) -> None:
        """Test basic request formatting."""
        model = LlamaCppModel(model_id="test-model")
        
        messages = [
            {"role": "user", "content": [{"text": "Hello"}]},
        ]
        
        request = model.format_request(messages)
        
        assert request["model"] == "test-model"
        assert request["messages"][0]["role"] == "user"
        # OpenAI format returns content as an array
        assert request["messages"][0]["content"][0]["type"] == "text"
        assert request["messages"][0]["content"][0]["text"] == "Hello"
        assert request["stream"] is True
        assert "extra_body" not in request  # No llama.cpp params, so no extra_body
    
    def test_format_request_with_system_prompt(self) -> None:
        """Test request formatting with system prompt."""
        model = LlamaCppModel()
        
        messages = [
            {"role": "user", "content": [{"text": "Hello"}]},
        ]
        
        request = model.format_request(messages, system_prompt="You are a helpful assistant")
        
        assert request["messages"][0]["role"] == "system"
        assert request["messages"][0]["content"] == "You are a helpful assistant"
        assert request["messages"][1]["role"] == "user"
    
    def test_format_request_with_llamacpp_params(self) -> None:
        """Test request formatting with llama.cpp specific parameters."""
        model = LlamaCppModel(
            params={
                "temperature": 0.8,
                "max_tokens": 50,
                "repeat_penalty": 1.1,
                "top_k": 40,
                "min_p": 0.05,
                "grammar": "root ::= 'yes' | 'no'",
            }
        )
        
        messages = [
            {"role": "user", "content": [{"text": "Is the sky blue?"}]},
        ]
        
        request = model.format_request(messages)
        
        # Standard OpenAI params
        assert request["temperature"] == 0.8
        assert request["max_tokens"] == 50
        
        # llama.cpp specific params should be in extra_body
        assert "extra_body" in request
        assert request["extra_body"]["repeat_penalty"] == 1.1
        assert request["extra_body"]["top_k"] == 40
        assert request["extra_body"]["min_p"] == 0.05
        assert request["extra_body"]["grammar"] == "root ::= 'yes' | 'no'"
    
    def test_format_request_with_all_new_params(self) -> None:
        """Test request formatting with all new llama.cpp parameters."""
        model = LlamaCppModel(
            params={
                # OpenAI params
                "temperature": 0.7,
                "max_tokens": 100,
                "top_p": 0.9,
                "seed": 42,
                # All llama.cpp specific params
                "repeat_penalty": 1.1,
                "top_k": 40,
                "min_p": 0.05,
                "typical_p": 0.95,
                "tfs_z": 0.97,
                "top_a": 0.1,
                "mirostat": 2,
                "mirostat_lr": 0.1,
                "mirostat_ent": 5.0,
                "grammar": "root ::= answer",
                "json_schema": {"type": "object"},
                "penalty_last_n": 256,
                "n_probs": 5,
                "min_keep": 1,
                "ignore_eos": False,
                "logit_bias": {100: 5.0, 200: -5.0},
                "cache_prompt": True,
                "slot_id": 1,
                "samplers": ["top_k", "tfs_z", "typical_p"],
            }
        )
        
        messages = [{"role": "user", "content": [{"text": "Test"}]}]
        request = model.format_request(messages)
        
        # Check OpenAI params are in root
        assert request["temperature"] == 0.7
        assert request["max_tokens"] == 100
        assert request["top_p"] == 0.9
        assert request["seed"] == 42
        
        # Check all llama.cpp params are in extra_body
        assert "extra_body" in request
        extra = request["extra_body"]
        assert extra["repeat_penalty"] == 1.1
        assert extra["top_k"] == 40
        assert extra["min_p"] == 0.05
        assert extra["typical_p"] == 0.95
        assert extra["tfs_z"] == 0.97
        assert extra["top_a"] == 0.1
        assert extra["mirostat"] == 2
        assert extra["mirostat_lr"] == 0.1
        assert extra["mirostat_ent"] == 5.0
        assert extra["grammar"] == "root ::= answer"
        assert extra["json_schema"] == {"type": "object"}
        assert extra["penalty_last_n"] == 256
        assert extra["n_probs"] == 5
        assert extra["min_keep"] == 1
        assert extra["ignore_eos"] == False
        assert extra["logit_bias"] == {100: 5.0, 200: -5.0}
        assert extra["cache_prompt"] == True
        assert extra["slot_id"] == 1
        assert extra["samplers"] == ["top_k", "tfs_z", "typical_p"]
    
    def test_format_request_with_tools(self) -> None:
        """Test request formatting with tool specifications."""
        model = LlamaCppModel()
        
        messages = [
            {"role": "user", "content": [{"text": "What's the weather?"}]},
        ]
        
        tool_specs = [
            {
                "name": "get_weather",
                "description": "Get current weather",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                        "required": ["location"],
                    }
                },
            }
        ]
        
        request = model.format_request(messages, tool_specs=tool_specs)
        
        assert "tools" in request
        assert len(request["tools"]) == 1
        assert request["tools"][0]["function"]["name"] == "get_weather"
    
    def test_update_config(self) -> None:
        """Test configuration update."""
        model = LlamaCppModel(model_id="initial-model")
        
        assert model.config["model_id"] == "initial-model"
        
        model.update_config(model_id="updated-model", params={"temperature": 0.5})
        
        assert model.config["model_id"] == "updated-model"
        assert model.config["params"]["temperature"] == 0.5
    
    def test_get_config(self) -> None:
        """Test configuration retrieval."""
        config = {
            "model_id": "test-model",
            "params": {"temperature": 0.9},
        }
        model = LlamaCppModel(**config)
        
        retrieved_config = model.get_config()
        
        assert retrieved_config["model_id"] == "test-model"
        assert retrieved_config["params"]["temperature"] == 0.9
    
    @pytest.mark.asyncio
    async def test_stream_basic(self) -> None:
        """Test basic streaming functionality."""
        model = LlamaCppModel()
        
        # Create properly structured mock events
        class MockDelta:
            content = None
            tool_calls = None
            def __init__(self, content=None):
                self.content = content
                
        class MockChoice:
            def __init__(self, content=None, finish_reason=None):
                self.delta = MockDelta(content)
                self.finish_reason = finish_reason
                
        class MockChunk:
            def __init__(self, choices, usage=None):
                self.choices = choices
                self.usage = usage
                
        mock_chunks = [
            MockChunk([MockChoice(content="Hello")]),
            MockChunk(
                [MockChoice(content=" world", finish_reason="stop")],
                usage=MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
            ),
        ]
        
        # Create async iterator
        async def mock_stream():
            for chunk in mock_chunks:
                yield chunk
        
        # Mock the create method to return a coroutine that returns the async iterator
        async def mock_create(*args, **kwargs):
            return mock_stream()
        
        with patch.object(model.client.chat.completions, "create", side_effect=mock_create):
            
            messages = [{"role": "user", "content": [{"text": "Hi"}]}]
            
            chunks = []
            async for chunk in model.stream(messages):
                chunks.append(chunk)
            
            # Verify we got the expected chunks
            assert any("messageStart" in chunk for chunk in chunks)
            assert any("contentBlockDelta" in chunk and chunk["contentBlockDelta"]["delta"]["text"] == "Hello" for chunk in chunks)
            assert any("contentBlockDelta" in chunk and chunk["contentBlockDelta"]["delta"]["text"] == " world" for chunk in chunks)
            assert any("messageStop" in chunk for chunk in chunks)
            assert any("metadata" in chunk for chunk in chunks)
    
    @pytest.mark.asyncio
    async def test_structured_output(self) -> None:
        """Test structured output functionality using the enhanced implementation."""
        
        class TestOutput(BaseModel):
            answer: str
            confidence: float
        
        model = LlamaCppModel()
        
        # Mock successful JSON response using the new structured_output implementation
        mock_response_text = '{"answer": "yes", "confidence": 0.95}'
        
        # Create mock stream that returns JSON
        async def mock_stream(*args, **kwargs):
            # Verify json_schema was set
            assert "json_schema" in model.config.get("params", {})
            
            yield {"messageStart": {"role": "assistant"}}
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": mock_response_text}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
            
        with patch.object(model, "stream", side_effect=mock_stream):
            messages = [{"role": "user", "content": [{"text": "Is the earth round?"}]}]
            
            events = []
            async for event in model.structured_output(TestOutput, messages):
                events.append(event)
            
            # Check we got the output
            output_event = next((e for e in events if "output" in e), None)
            assert output_event is not None
            assert output_event["output"].answer == "yes"
            assert output_event["output"].confidence == 0.95
    
    def test_timeout_configuration(self) -> None:
        """Test timeout configuration."""
        model = LlamaCppModel(timeout=30.0)
        
        # The timeout should be passed to the OpenAI client
        assert model.client.timeout == 30.0
        
        # Test with tuple timeout
        model2 = LlamaCppModel(timeout=(10.0, 60.0))
        assert model2.client.timeout == (10.0, 60.0)
    
    def test_max_retries_configuration(self) -> None:
        """Test max retries configuration."""
        model = LlamaCppModel(max_retries=5)
        
        # The max_retries should be passed to the OpenAI client
        assert model.client.max_retries == 5
    
    def test_use_grammar_constraint(self) -> None:
        """Test grammar constraint method."""
        model = LlamaCppModel()
        
        # Apply grammar constraint
        grammar = '''
        root ::= answer
        answer ::= "yes" | "no"
        '''
        model.use_grammar_constraint(grammar)
        
        assert model.config["params"]["grammar"] == grammar
        
        # Update grammar
        new_grammar = 'root ::= [0-9]+'
        model.use_grammar_constraint(new_grammar)
        
        assert model.config["params"]["grammar"] == new_grammar
    
    def test_use_json_schema(self) -> None:
        """Test JSON schema constraint method."""
        model = LlamaCppModel()
        
        # Apply JSON schema
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }
        model.use_json_schema(schema)
        
        assert model.config["params"]["json_schema"] == schema
    
    @pytest.mark.asyncio
    async def test_stream_with_context_overflow_error(self) -> None:
        """Test stream handling of context overflow errors."""
        model = LlamaCppModel()
        
        # Create HTTP error response
        error_response = httpx.Response(
            status_code=400,
            json={"error": {"message": "Context window exceeded. Max context length is 4096 tokens"}},
            request=httpx.Request("POST", "http://test.com")
        )
        error = httpx.HTTPStatusError("Bad Request", request=error_response.request, response=error_response)
        
        # Mock the parent stream to raise the error
        with patch.object(model.client.chat.completions, "create", side_effect=error):
            messages = [{"role": "user", "content": [{"text": "Very long message"}]}]
            
            with pytest.raises(LlamaCppContextOverflowError) as exc_info:
                async for _ in model.stream(messages):
                    pass
            
            assert "Context window exceeded" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_stream_with_server_overload_error(self) -> None:
        """Test stream handling of server overload errors."""
        model = LlamaCppModel()
        
        # Create HTTP error response for 503
        error_response = httpx.Response(
            status_code=503,
            text="Server is busy",
            request=httpx.Request("POST", "http://test.com")
        )
        error = httpx.HTTPStatusError("Service Unavailable", request=error_response.request, response=error_response)
        
        # Mock the parent stream to raise the error
        with patch.object(model.client.chat.completions, "create", side_effect=error):
            messages = [{"role": "user", "content": [{"text": "Test"}]}]
            
            with pytest.raises(ModelThrottledException) as exc_info:
                async for _ in model.stream(messages):
                    pass
            
            assert "server is busy or overloaded" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_structured_output_with_json_schema(self) -> None:
        """Test structured output using JSON schema."""
        
        class TestOutput(BaseModel):
            answer: str
            confidence: float
        
        model = LlamaCppModel()
        
        # Mock successful JSON response
        mock_response_text = '{"answer": "yes", "confidence": 0.95}'
        
        # Create mock stream that returns JSON
        async def mock_stream(*args, **kwargs):
            # Check that json_schema was set correctly
            assert model.config["params"]["json_schema"] == TestOutput.model_json_schema()
            
            yield {"messageStart": {"role": "assistant"}}
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": mock_response_text}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
            
        with patch.object(model, "stream", side_effect=mock_stream):
            messages = [{"role": "user", "content": [{"text": "Is the earth round?"}]}]
            
            events = []
            async for event in model.structured_output(TestOutput, messages):
                events.append(event)
            
            # Check we got the output
            output_event = next((e for e in events if "output" in e), None)
            assert output_event is not None
            assert output_event["output"].answer == "yes"
            assert output_event["output"].confidence == 0.95
    
    @pytest.mark.asyncio
    async def test_structured_output_invalid_json_error(self) -> None:
        """Test structured output raises error for invalid JSON."""
        
        class TestOutput(BaseModel):
            value: int
        
        model = LlamaCppModel()
        
        # Mock stream that returns invalid JSON
        async def mock_stream(*args, **kwargs):
            # Check that json_schema was set correctly
            assert model.config["params"]["json_schema"] == TestOutput.model_json_schema()
            
            yield {"messageStart": {"role": "assistant"}}
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": "This is not valid JSON"}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
            
        with patch.object(model, "stream", side_effect=mock_stream):
            messages = [{"role": "user", "content": [{"text": "Give me a number"}]}]
            
            with pytest.raises(json.JSONDecodeError):
                async for event in model.structured_output(TestOutput, messages):
                    pass