import json
import logging
from typing import Any, Dict, List, Tuple

from openai import AsyncAzureOpenAI

from client.providers.base import LLMProvider


class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI implementation with enterprise authentication.

    Handles Azure-specific authentication, endpoint management, and
    request formatting. Uses the OpenAI API format but with Azure's
    deployment and authentication model.

    Args:
        azure_openai_endpoint: Azure OpenAI resource endpoint URL
        api_key: Azure OpenAI API key
        api_version: API version (e.g., "2024-12-01-preview")
        azure_openai_model_name: Deployment name in Azure OpenAI
        temperature: Model temperature (0.0-1.0) for response randomness
    """

    def __init__(
        self,
        azure_openai_endpoint,
        api_key,
        api_version,
        azure_openai_model_name,
        temperature=0.0,
    ):
        self.client = AsyncAzureOpenAI(
            azure_endpoint=azure_openai_endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        self.deployment_name = azure_openai_model_name
        self.messages = []
        self.temperature = temperature

    async def initialize(self) -> None:
        pass

    async def format_tools_for_model(self, mcp_tools: List) -> List[Dict[str, Any]]:
        # Convert MCP tools to Azure OpenAI format
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in mcp_tools
        ]

    async def add_user_message(self, message: str) -> None:
        self.messages.append({"role": "user", "content": message})

    async def get_model_response(self, tools: Any) -> Tuple[str, List[Dict[str, Any]]]:
        logging.debug("\n========== DATA SENT TO AZURE OPENAI ==========")
        logging.debug(f"MESSAGES: {self.messages}")
        logging.debug(f"TOOLS: {tools}")
        logging.debug("==============================================\n")

        response = await self.client.chat.completions.create(
            model=self.deployment_name,
            messages=self.messages,
            tools=tools,
            temperature=self.temperature,
        )

        logging.debug("\n========== RESPONSE FROM AZURE OPENAI ==========")
        logging.debug(f"RESPONSE TYPE: {type(response)}")
        logging.debug(f"CHOICES: {response.choices}")
        logging.debug("===============================================\n")

        response_message = response.choices[0].message
        text_content = response_message.content or ""
        tool_calls = []

        # Add message to history (including tool calls)
        self.messages.append(response_message)

        # Extract tool calls if any
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                tool_calls.append(
                    {
                        "name": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments),
                        "id": tool_call.id,
                    }
                )

        return text_content, tool_calls

    async def process_tool_result(
        self, tool_name: str, result: Any, tool_call_id: str = None
    ) -> None:
        # Process result from MCP tool call
        if hasattr(result, "content"):
            # Extract text from content items
            tool_response_content = []
            for item in result.content:
                if hasattr(item, "text"):
                    tool_response_content.append({"type": "text", "text": item.text})

            # Format content properly
            content = (
                json.dumps(tool_response_content)
                if tool_response_content
                else "No content"
            )
        else:
            # Fallback if result doesn't have content attribute
            content = str(result)

        # Create the message with proper structure
        message = {"role": "tool", "name": tool_name, "content": content}

        # Add tool_call_id (required for Azure OpenAI)
        if tool_call_id is not None:
            message["tool_call_id"] = tool_call_id

        logging.debug("\n========== TOOL RESULT ADDED TO CONVERSATION ==========")
        logging.debug(f"TOOL: {tool_name}")
        logging.debug(f"MESSAGE: {message}")
        logging.debug("=====================================================\n")

        # Add to conversation history
        self.messages.append(message)
