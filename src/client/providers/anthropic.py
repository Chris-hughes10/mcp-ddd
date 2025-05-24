from typing import Any, Dict, List, Optional, Tuple

from anthropic import Anthropic

from client.providers.base import LLMProvider


class AnthropicProvider(LLMProvider):
    """Anthropic-specific implementation"""

    def __init__(self, api_key, model="claude-3-5-sonnet-20241022", max_tokens=1000):
        self.anthropic = Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.messages = []

    async def initialize(self) -> None:
        pass

    async def format_tools_for_model(self, mcp_tools: List) -> List[Dict[str, Any]]:
        # Convert MCP tools to Anthropic format
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in mcp_tools
        ]

    async def add_user_message(self, message: str) -> None:
        self.messages.append({"role": "user", "content": message})

    async def get_model_response(self, tools: Any) -> Tuple[str, List[Dict[str, Any]]]:
        response = self.anthropic.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=self.messages,
            tools=tools,
        )

        text_content = ""
        tool_calls = []

        for content in response.content:
            if content.type == "text":
                text_content += content.text
            elif content.type == "tool_use":
                tool_calls.append(
                    {
                        "name": content.name,
                        "arguments": content.input,
                    }
                )

        if text_content:
            self.messages.append({"role": "assistant", "content": text_content})

        return text_content, tool_calls

    async def process_tool_result(
        self, tool_name: str, result: Any, tool_call_id: Optional[str] = None
    ) -> None:
        self.messages.append({"role": "user", "content": result})
