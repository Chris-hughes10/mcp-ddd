import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class LLMProvider(ABC):
    """Abstract base class for Large Language Model service providers.

    Enables swapping between different LLM services (OpenAI, Anthropic, etc.)
    while maintaining consistent interface for MCP clients. Handles provider-
    specific details like authentication, request formatting, and response parsing.

    Key responsibilities:
    - Format MCP tools for provider-specific schemas
    - Manage conversation state and message history
    - Handle tool call execution results
    - Abstract provider-specific authentication and requests
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the LLM client"""
        pass

    @abstractmethod
    async def format_tools_for_model(self, mcp_tools: List) -> Any:
        """Convert MCP tool definitions to provider-specific format.

        Each LLM provider has different schemas for tool/function calling.
        This method transforms the standardized MCP tool definitions into
        the format expected by the specific provider.

        Args:
            mcp_tools: List of MCP tool objects with standardized schema

        Returns:
            Provider-specific tool definitions (format varies by provider)
        """

        pass

    @abstractmethod
    async def add_user_message(self, message: str) -> None:
        """Add a user message to the conversation"""
        pass

    @abstractmethod
    async def get_model_response(self, tools: Any) -> Tuple[str, List[Dict[str, Any]]]:
        """Get a response from the model, returning (text, tool_calls)"""
        pass

    @abstractmethod
    async def process_tool_result(
        self, tool_name: str, result: Any, tool_call_id: Optional[str] = None
    ) -> None:
        """Process and incorporate a tool result into the conversation"""
        pass
