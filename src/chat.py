import asyncio
import os
import pathlib
import sys

from dotenv import load_dotenv

from client.prompt_aware_client import PromptAwareMCPClient
from client.providers.anthropic import AnthropicProvider
from client.providers.azure_openai import AzureOpenAIProvider
from client.resource_selector_client import LLMResourceSelector
from client.simple_client import SimpleChatMCPClient

__FILE_PATH__ = pathlib.Path(__file__).resolve()

import logging

logging.basicConfig(level=logging.INFO)


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script> [provider]")
        sys.exit(1)

    # Determine provider from command line arg or use Azure OpenAI by default
    provider_name = sys.argv[2] if len(sys.argv) > 2 else "azure"
    client_type = sys.argv[3] if len(sys.argv) > 3 else "simple"

    env_path = __FILE_PATH__.parent.parent / "env"

    if provider_name.lower() == "anthropic":
        load_dotenv(env_path / "anthropic.env")
        provider = AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"])
    else:
        load_dotenv(env_path / "azure.env")

        provider = AzureOpenAIProvider(
            azure_openai_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ["OPENAI_API_VERSION"],
            azure_openai_model_name=os.environ["AZURE_OPENAI_MODEL"],
        )

    if client_type.lower() == "resource_selector":
        client = LLMResourceSelector(provider)
    elif client_type.lower() == "prompt_aware":
        client = PromptAwareMCPClient(provider)
    else:
        client = SimpleChatMCPClient(provider)

    try:
        await client.connect_to_server(sys.argv[1])
        await client.start_interactive_session()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
