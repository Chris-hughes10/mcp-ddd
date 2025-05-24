import logging

from client.providers.base import LLMProvider
from client.simple_client import SimpleChatMCPClient

logger = logging.getLogger(__name__)


class LLMResourceSelector(SimpleChatMCPClient):
    """MCP client that uses LLM intelligence to select relevant resources.

    Extends basic client with smart resource selection. Before processing
    queries, asks the LLM to analyze available resources and load only
    those relevant to the user's request.

    This reduces token usage and improves response quality by providing
    focused context rather than all available resources.
    """

    def __init__(self, llm_provider: LLMProvider):
        super().__init__(llm_provider)
        self.resource_templates = []
        self.available_resources = []

    async def handle_user_query(self, query: str) -> str:
        """
        Process a query using LLM-driven resource selection
        """
        # First, have the LLM analyze the query to determine which resources to use
        resource_uris = await self._select_relevant_resources(query)

        if resource_uris:
            logger.info(f"Identified resources as relevant: {resource_uris}")
            # Load the selected resources
            resource_content = await self._load_resources(resource_uris)

            # Enhance the query with the selected resources
            enhanced_query = (
                f"I have the following information that may be relevant to the query:\n\n"
                f"{resource_content}\n\n"
                f"Using this information where relevant, please answer: {query}"
                "You do not have to use a resource if you don't think it will help, you may also use any tools you have\n\n"
            )

            # Use the enhanced query
            return await super().handle_user_query(enhanced_query)
        else:
            # No relevant resources identified, use the original query
            return await super().handle_user_query(query)

    async def _select_relevant_resources(self, query: str) -> list:
        """
        Have the LLM decide which resources would be relevant for the given query.
        Returns a list of resource URIs to load.
        """
        # Create a summary of available resources for the LLM
        resource_descriptions = self._format_resource_descriptions()

        # Create a prompt for resource selection
        resource_selection_prompt = (
            f"I need to determine which resources would help answer this query: '{query}'\n\n"
            f"Available resources:\n{resource_descriptions}\n\n"
            f"For resource templates, I need to determine appropriate parameter values.\n\n"
            f"Analyze the query and list only the URIs of resources that would be helpful, "
            f"with any parameter substitutions for templates. Format as a JSON list."
        )

        # Ask the LLM to select resources
        await self.provider.add_user_message(resource_selection_prompt)
        response_text, _ = await self.provider.get_model_response([])

        # Extract the resource URIs from the response
        return self._extract_resource_uris(response_text)

    def _format_resource_descriptions(self) -> str:
        """Format resource descriptions for the LLM prompt"""
        descriptions = []

        for resource in self.available_resources:
            descriptions.append(
                f"- {resource['name']}: {resource['uri']} - {resource['description']}"
            )

        for template in self.resource_templates:
            descriptions.append(
                f"- {template['name']}: {template['uriTemplate']} - {template['description']}"
            )

        return "\n".join(descriptions)

    def _extract_resource_uris(self, response_text: str) -> list:
        """Extract resource URIs from LLM response"""
        import json
        import re

        # Try to find and parse a JSON list in the response
        json_match = re.search(r"\[.*?\]", response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback: extract anything that looks like a resource URI
        uri_pattern = r'(([a-zA-Z0-9_-]+)://[^\s,"\']*)'
        matches = re.findall(uri_pattern, response_text)
        if matches:
            return [match[0] for match in matches]

        return []

    async def _load_resources(self, resource_uris: list) -> str:
        """Load the content of selected resources"""
        resource_sections = []

        for uri in resource_uris:
            resource_sections.append(f"\n\nRESOURCE: {uri}\n")
            try:
                # Fetch the resource content
                response = await self.session.read_resource(uri)
                for response_contents in response.contents:
                    # Add to the collection with metadata
                    resource_sections.append(f"\n{response_contents}")
            except Exception as e:
                logger.error(f"Error loading resource {uri}: {str(e)}")

        return "\n\n".join(resource_sections)
