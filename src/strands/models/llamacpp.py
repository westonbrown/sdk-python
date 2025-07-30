"""llama.cpp model provider.

- Docs: https://github.com/ggml-org/llama.cpp
- Server docs: https://github.com/ggml-org/llama.cpp/tree/master/tools/server
"""

import json
import logging
from typing import Any, AsyncGenerator, Optional, Type, TypedDict, TypeVar, Union

import httpx
from pydantic import BaseModel
from typing_extensions import Unpack, override

from ..types.content import Messages
from ..types.exceptions import ContextWindowOverflowException, ModelThrottledException
from ..types.streaming import StreamEvent
from ..types.tools import ToolSpec
from .openai import OpenAIModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LlamaCppError(Exception):
    """Base exception for llama.cpp specific errors."""
    pass


class LlamaCppContextOverflowError(LlamaCppError, ContextWindowOverflowException):
    """Raised when context window is exceeded in llama.cpp."""
    pass


class LlamaCppModel(OpenAIModel):
    """llama.cpp model provider implementation.

    Connects to a llama.cpp server running in OpenAI-compatible mode with
    support for advanced llama.cpp-specific features like grammar constraints,
    Mirostat sampling, and native JSON schema validation.

    The llama.cpp server must be started with the OpenAI-compatible API enabled:
        llama-server -m model.gguf --host 0.0.0.0 --port 8080

    Example:
        Basic usage:
        >>> model = LlamaCppModel(base_url="http://localhost:8080/v1")
        >>> model.update_config(params={"temperature": 0.7, "top_k": 40})

        Grammar constraints:
        >>> model.use_grammar_constraint('''
        ...     root ::= answer
        ...     answer ::= "yes" | "no"
        ... ''')

        Advanced sampling:
        >>> model.update_config(params={
        ...     "mirostat": 2,
        ...     "mirostat_lr": 0.1,
        ...     "tfs_z": 0.95,
        ...     "repeat_penalty": 1.1
        ... })
    """

    class LlamaCppConfig(TypedDict, total=False):
        """Configuration options for llama.cpp models.

        Attributes:
            model_id: Model identifier for the loaded model in llama.cpp server.
                Default is "default" as llama.cpp typically loads a single model.
            params: Model parameters supporting both OpenAI and llama.cpp-specific options.

                OpenAI-compatible parameters:
                - max_tokens: Maximum number of tokens to generate
                - temperature: Sampling temperature (0.0 to 2.0)
                - top_p: Nucleus sampling parameter (0.0 to 1.0)
                - frequency_penalty: Frequency penalty (-2.0 to 2.0)
                - presence_penalty: Presence penalty (-2.0 to 2.0)
                - stop: List of stop sequences
                - seed: Random seed for reproducibility
                - n: Number of completions to generate
                - logprobs: Include log probabilities in output
                - top_logprobs: Number of top log probabilities to include

                llama.cpp-specific parameters:
                - repeat_penalty: Penalize repeat tokens (1.0 = no penalty)
                - top_k: Top-k sampling (0 = disabled)
                - min_p: Min-p sampling threshold (0.0 to 1.0)
                - typical_p: Typical-p sampling (0.0 to 1.0)
                - tfs_z: Tail-free sampling parameter (0.0 to 1.0)
                - top_a: Top-a sampling parameter
                - mirostat: Mirostat sampling mode (0, 1, or 2)
                - mirostat_lr: Mirostat learning rate
                - mirostat_ent: Mirostat target entropy
                - grammar: GBNF grammar string for constrained generation
                - json_schema: JSON schema for structured output
                - penalty_last_n: Number of tokens to consider for penalties
                - n_probs: Number of probabilities to return per token
                - min_keep: Minimum tokens to keep in sampling
                - ignore_eos: Ignore end-of-sequence token
                - logit_bias: Token ID to bias mapping
                - cache_prompt: Cache the prompt for faster generation
                - slot_id: Slot ID for parallel inference
                - samplers: Custom sampler order
        """

        model_id: str
        params: Optional[dict[str, Any]]

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        api_key: Optional[str] = None,
        timeout: Optional[Union[float, tuple[float, float]]] = None,
        max_retries: Optional[int] = None,
        **model_config: Unpack[LlamaCppConfig],
    ) -> None:
        """Initialize llama.cpp provider instance.

        Args:
            base_url: Base URL for the llama.cpp server.
                Default is "http://localhost:8080/v1" for local server.
            api_key: Optional API key if the llama.cpp server requires authentication.
            timeout: Request timeout in seconds. Can be a float or tuple of (connect, read) timeouts.
            max_retries: Maximum number of retries for failed requests.
            **model_config: Configuration options for the llama.cpp model.
        """
        # Set default model_id if not provided
        if "model_id" not in model_config:
            model_config["model_id"] = "default"

        # Build OpenAI client args
        client_args = {
            "base_url": base_url,
            "api_key": api_key or "dummy",  # OpenAI client requires some API key
        }

        if timeout is not None:
            client_args["timeout"] = timeout

        if max_retries is not None:
            client_args["max_retries"] = max_retries

        logger.debug(
            "base_url=<%s>, model_id=<%s> | initializing llama.cpp provider",
            base_url,
            model_config.get("model_id"),
        )

        # Initialize parent OpenAI model with our client args
        super().__init__(client_args=client_args, **model_config)

    def use_grammar_constraint(self, grammar: str) -> None:
        """Apply a GBNF grammar constraint to the generation.

        Args:
            grammar: GBNF (Backus-Naur Form) grammar string defining allowed outputs.
                     See https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md

        Example:
            >>> # Constrain output to yes/no answers
            >>> model.use_grammar_constraint('''
            ...     root ::= answer
            ...     answer ::= "yes" | "no"
            ... ''')

            >>> # JSON object grammar
            >>> model.use_grammar_constraint('''
            ...     root ::= object
            ...     object ::= "{" pair ("," pair)* "}"
            ...     pair ::= string ":" value
            ...     string ::= "\\"" [^"]* "\\""
            ...     value ::= string | number | "true" | "false" | "null"
            ...     number ::= "-"? [0-9]+ ("." [0-9]+)?
            ... ''')
        """
        if not self.config.get("params"):
            self.config["params"] = {}
        self.config["params"]["grammar"] = grammar
        logger.debug("Applied grammar constraint")

    def use_json_schema(self, schema: dict[str, Any]) -> None:
        """Apply a JSON schema constraint for structured output.

        Args:
            schema: JSON schema dictionary defining the expected output structure.

        Example:
            >>> model.use_json_schema({
            ...     "type": "object",
            ...     "properties": {
            ...         "name": {"type": "string"},
            ...         "age": {"type": "integer", "minimum": 0}
            ...     },
            ...     "required": ["name", "age"]
            ... })
        """
        if not self.config.get("params"):
            self.config["params"] = {}
        self.config["params"]["json_schema"] = schema
        logger.debug("Applied JSON schema constraint")

    @override
    def format_request(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """Format a request for the llama.cpp server.

        This method overrides the OpenAI format to properly handle llama.cpp-specific parameters.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.

        Returns:
            A request formatted for llama.cpp server's OpenAI-compatible API.
        """
        # Build base request structure without calling super() to avoid
        # parameter conflicts between OpenAI and llama.cpp specific params.
        # This allows us to properly separate parameters into the appropriate
        # request fields (direct vs extra_body).
        request = {
            "messages": self.format_request_messages(messages, system_prompt),
            "model": self.config["model_id"],
            "stream": True,
            "stream_options": {"include_usage": True},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool_spec["name"],
                        "description": tool_spec["description"],
                        "parameters": tool_spec["inputSchema"]["json"],
                    },
                }
                for tool_spec in tool_specs or []
            ],
        }

        # Handle parameters if provided
        if self.config.get("params"):
            params = self.config["params"]

            # Define llama.cpp-specific parameters that need special handling
            llamacpp_specific_params = {
                "repeat_penalty",
                "top_k",
                "min_p",
                "typical_p",
                "tfs_z",
                "top_a",
                "mirostat",
                "mirostat_lr",
                "mirostat_ent",
                "grammar",
                "json_schema",
                "penalty_last_n",
                "n_probs",
                "min_keep",
                "ignore_eos",
                "logit_bias",
                "cache_prompt",
                "slot_id",
                "samplers",
            }

            # Standard OpenAI parameters that go directly in request
            openai_params = {
                "temperature",
                "max_tokens",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "stop",
                "seed",
                "n",
                "logprobs",
                "top_logprobs",
                "response_format",
            }

            # Add OpenAI parameters directly to request
            for param, value in params.items():
                if param in openai_params:
                    request[param] = value

            # Collect llama.cpp-specific parameters for extra_body
            extra_body = {}
            for param, value in params.items():
                if param in llamacpp_specific_params:
                    extra_body[param] = value

            # Add extra_body if we have llama.cpp-specific parameters
            if extra_body:
                request["extra_body"] = extra_body

        return request

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream conversation with the llama.cpp model.

        This method extends the OpenAI stream to handle llama.cpp-specific errors.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Formatted message chunks from the model.

        Raises:
            LlamaCppContextOverflowError: When the context window is exceeded.
            ModelThrottledException: When the llama.cpp server is overloaded.
        """
        try:
            async for event in super().stream(messages, tool_specs, system_prompt, **kwargs):
                yield event
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                # Parse error response
                try:
                    error_data = e.response.json()
                    error_msg = str(error_data.get("error", {}).get("message", str(error_data)))
                except (json.JSONDecodeError, KeyError, AttributeError):
                    error_msg = e.response.text

                # Check for context overflow
                if any(term in error_msg.lower() for term in ["context", "kv cache", "slot"]):
                    raise LlamaCppContextOverflowError(
                        f"Context window exceeded: {error_msg}"
                    ) from e
            elif e.response.status_code == 503:
                raise ModelThrottledException(
                    "llama.cpp server is busy or overloaded"
                ) from e
            raise
        except Exception as e:
            # Handle other potential errors
            error_msg = str(e).lower()
            if "rate" in error_msg or "429" in str(e):
                raise ModelThrottledException(str(e)) from e
            raise

    @override
    async def structured_output(
        self, output_model: Type[T], prompt: Messages, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> AsyncGenerator[dict[str, Union[T, Any]], None]:
        """Get structured output using llama.cpp's native JSON schema support.

        This implementation uses llama.cpp's json_schema parameter to constrain
        the model output to valid JSON matching the provided schema.

        Args:
            output_model: The Pydantic model defining the expected output structure.
            prompt: The prompt messages to use for generation.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Model events with the last being the structured output.

        Raises:
            json.JSONDecodeError: If the model output is not valid JSON.
            pydantic.ValidationError: If the output doesn't match the model schema.
        """
        # Get the JSON schema from the Pydantic model
        schema = output_model.model_json_schema()

        # Store current params to restore later
        original_params = self.config.get("params", {}).copy()

        try:
            # Configure for JSON output with schema constraint
            if not self.config.get("params"):
                self.config["params"] = {}

            self.config["params"]["json_schema"] = schema
            self.config["params"]["cache_prompt"] = True  # Cache schema processing

            # Collect the response
            response_text = ""
            async for event in self.stream(prompt, system_prompt=system_prompt, **kwargs):
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"]["delta"]
                    if "text" in delta:
                        response_text += delta["text"]
                # Pass through other events
                yield event

            # Parse and validate the JSON response
            data = json.loads(response_text.strip())
            output_instance = output_model(**data)
            yield {"output": output_instance}

        finally:
            # Restore original params
            self.config["params"] = original_params

    def _generate_pydantic_grammar(self, model: Type[BaseModel]) -> str:
        """Generate a GBNF grammar from a Pydantic model.

        Args:
            model: The Pydantic model to generate grammar for.

        Returns:
            GBNF grammar string.

        Note:
            This provides a basic JSON grammar. A future enhancement would
            generate model-specific grammars based on the Pydantic schema.
        """
        # Basic JSON grammar that works for most cases
        return '''
root ::= object
object ::= "{" pair ("," pair)* "}"
pair ::= string ":" value
string ::= "\\"" [^"]* "\\""
value ::= string | number | boolean | null | array | object
array ::= "[" (value ("," value)*)? "]"
number ::= "-"? [0-9]+ ("." [0-9]+)?
boolean ::= "true" | "false"
null ::= "null"
'''
