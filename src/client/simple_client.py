import logging
from contextlib import AsyncExitStack
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from client.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class SimpleChatMCPClient:
    """Basic MCP client with tool calling and conversation management.

    Provides core MCP functionality:
    - Connects to MCP servers via stdio
    - Discovers and calls tools
    - Manages conversation state with LLM providers
    - Handles interactive chat sessions

    Args:
        llm_provider: LLM service provider (Azure OpenAI, Anthropic, etc.)
    """

    def __init__(self, llm_provider: LLMProvider):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.provider = llm_provider
        self.available_resources = []
        self.resource_templates = []
        self.available_prompts = []

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server"""
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command, args=[server_script_path], env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()
        await self.provider.initialize()

        await self.discover_tools()
        await self._discover_resources()
        await self._discover_prompts()

    async def discover_tools(self):
        response = await self.session.list_tools()
        self._mcp_server_tools = response.tools
        self._formatted_tools = await self.provider.format_tools_for_model(
            self._mcp_server_tools
        )
        logger.info(
            f"Connected to server with tools: {[tool.name for tool in self._mcp_server_tools]}"
        )

    async def _discover_resources(self):
        """Discover available resources and resource templates from MCP server.

        MCP servers can expose two types of resources:
        1. Concrete resources: Fixed URIs with static content
        2. Resource templates: URI patterns with parameters (e.g., "user://{id}")

        This method queries both endpoints and stores the results for later
        resource selection and loading.

        Populates:
            self.available_resources: List of concrete resource descriptors
            self.resource_templates: List of parameterized resource templates
        """
        if not self.session:
            logger.error("Cannot discover resources: No active session")
            return

        try:
            # First, get concrete resources
            logger.info("Discovering concrete resources...")
            resources_response = await self.session.list_resources()

            # Then, get resource templates
            logger.info("Discovering resource templates...")
            templates_response = await self.session.list_resource_templates()

            # Store both concrete resources and resource templates
            self.available_resources = []
            self.resource_templates = []

            # Process concrete resources
            if hasattr(resources_response, "resources"):
                for resource in resources_response.resources:
                    if hasattr(resource, "uri"):
                        self.available_resources.append(
                            {
                                "uri": resource.uri,
                                "name": resource.name,
                                "description": getattr(resource, "description", ""),
                                "mimeType": getattr(resource, "mimeType", "text/plain"),
                            }
                        )

            # Process resource templates
            if hasattr(templates_response, "resourceTemplates"):
                for template in templates_response.resourceTemplates:
                    if hasattr(template, "uriTemplate"):
                        self.resource_templates.append(
                            {
                                "uriTemplate": template.uriTemplate,
                                "name": template.name,
                                "description": getattr(template, "description", ""),
                                "mimeType": getattr(template, "mimeType", "text/plain"),
                            }
                        )

            # Log discovered resources
            logger.info(
                f"Discovered {len(self.available_resources)} concrete resources"
            )
            logger.info(f"Discovered {len(self.resource_templates)} resource templates")

            for template in self.resource_templates:
                logger.info(f"Template: {template['uriTemplate']}")

        except Exception as e:
            logger.error(f"Error discovering resources: {str(e)}")

    async def _discover_prompts(self):
        """Discover prompt templates available from the MCP server.

        Prompts in MCP are user-controlled templates that provide standardized
        ways to initiate specific types of conversations. They can accept
        parameters and return structured prompt content.

        Populates:
            self.available_prompts: List of prompt descriptors with metadata
        """
        if not self.session:
            return

        try:
            prompts_response = await self.session.list_prompts()
            self.available_prompts = []

            if hasattr(prompts_response, "prompts"):
                for prompt in prompts_response.prompts:
                    self.available_prompts.append(
                        {
                            "name": prompt.name,
                            "description": getattr(prompt, "description", ""),
                            "arguments": getattr(prompt, "arguments", []),
                        }
                    )

            logger.info(f"Discovered {len(self.available_prompts)} prompts")
        except Exception as e:
            logger.error(f"Error discovering prompts: {str(e)}")

    async def handle_user_query(self, query: str) -> str:
        """Process a query using the LLM provider and available tools"""

        # Add the user's query
        await self.provider.add_user_message(query)

        response_parts = []

        while True:
            # Get next response from provider (text and any tool calls)
            text, tool_calls = await self.provider.get_model_response(
                self._formatted_tools
            )

            if text:
                response_parts.append(text)

            # If no tool calls, we're done
            if not tool_calls:
                break

            # Process each tool call
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["arguments"]
                tool_call_id = tool_call.get("id")

                logger.info(f"[Calling tool {tool_name} with args {tool_args}]")

                # Execute tool call via MCP
                result = await self.session.call_tool(tool_name, tool_args)

                # Add tool result to conversation
                await self.provider.process_tool_result(
                    tool_name,
                    result,
                    tool_call_id,
                )

        return "\n".join([text for text in response_parts if text])

    async def start_interactive_session(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == "quit":
                    await self.cleanup()
                    break

                response = await self.handle_user_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
