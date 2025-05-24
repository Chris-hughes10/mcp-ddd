import json
import logging

from client.simple_client import SimpleChatMCPClient

logger = logging.getLogger(__name__)


class PromptAwareMCPClient(SimpleChatMCPClient):
    """Extends SimpleChatMCPClient with prompt discovery and usage"""

    async def start_interactive_session(self):
        """Show available prompts at start of interaction"""
        print("\nMCP Client Started!")

        # Show available prompts
        if self.available_prompts:
            print("\nAvailable MCP Prompts:")
            for i, prompt in enumerate(self.available_prompts, 1):
                args = ", ".join([f"{arg.name}" for arg in prompt.get("arguments", [])])
                print(f"{i}. {prompt['name']} - {prompt['description']}")
            print(
                "\nYou can use these prompts in your queries by calling them like Python functions."
            )

        # Use the parent class's interactive session logic
        await super().start_interactive_session()

    async def handle_user_query(self, query: str) -> str:
        """
        Process a query using LLM-driven resource selection
        """
        # First, have the LLM analyze the query to determine which resources to use
        prompt_info = await self._select_relevant_prompt(query)

        if prompt_info:
            logger.info(f"Identified prompt call as: {prompt_info}")
            # Load the selected resources

            enhanced_query = await self.use_prompt(prompt_info)

            # Use the enhanced query
            return await super().handle_user_query(enhanced_query)
        else:
            # No relevant resources identified, use the original query
            return await super().handle_user_query(query)

    async def _select_relevant_prompt(self, query: str) -> list:
        """
        Have the LLM decide which resources would be relevant for the given query.
        Returns a list of resource URIs to load.
        """

        # Create a prompt for prompt selection
        prompt_selection_prompt = (
            f"I need to determine whether the user is trying to use an available prompt. "
            f"Their query is: '{query}'\n\n"
            f"The available prompts you have are:\n{self.available_prompts}\n\n"
            f"Only use a prompt if the user explicitly asked for it "
            f"If they are trying to use a prompt, please return the name of the prompt and any arguments "
            f'that should be used as Json with format {{"prompt_name": ..., "arguments": {{...}}}}. '
            f"If they are not trying to use a prompt, please return an empty Json array.\n\n"
        )

        # Ask the LLM to select resources
        await self.provider.add_user_message(prompt_selection_prompt)
        response_text, _ = await self.provider.get_model_response([])

        # Extract the prompt name and arguments from the response
        try:
            response_json = json.loads(response_text)
            if isinstance(response_json, dict):
                if not response_json:
                    return []
                prompt_name = response_json.get("prompt_name")
                arguments = response_json.get("arguments", {})
                if prompt_name and isinstance(arguments, dict):
                    return [prompt_name, arguments]
        except json.JSONDecodeError:
            logger.debug(f"Failed to parse JSON response: {response_text}")
            return []

    async def use_prompt(self, prompt_info) -> str:
        """Use an MCP prompt with given arguments"""
        try:
            prompt_name, arguments = prompt_info
            prompt_result = await self.session.get_prompt(prompt_name, arguments)

            return prompt_result.messages[0].content.text
        except Exception as e:
            return f"Error using prompt '{prompt_info}': {str(e)}"
