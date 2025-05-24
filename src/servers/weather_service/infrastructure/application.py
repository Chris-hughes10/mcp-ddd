from abc import ABC, abstractmethod
from typing import Callable, List, Tuple

from mcp.server.fastmcp import FastMCP


class MCPApplicationService(ABC):
    """Base class for MCP Application Services following DDD patterns.

    Application Services orchestrate domain objects to fulfill use cases
    while remaining independent of infrastructure concerns. This class
    provides the MCP-specific infrastructure setup.

    In DDD terms:
    - Tools map to domain service operations
    - Resources provide read-only access to aggregates
    - Prompts offer templated workflows

    Args:
        mcp: FastMCP server instance for protocol handling
    """

    def __init__(self, mcp: FastMCP):
        self.mcp = mcp

        self._register_tools()
        self._register_resources()
        self._register_prompts()

    @property
    @abstractmethod
    def tools(self) -> List[Tuple[str, str, Callable]]:
        pass

    @property
    @abstractmethod
    def resources(self) -> List[Tuple[str, str, str, Callable]]:
        pass

    @property
    @abstractmethod
    def prompts(self) -> List[Tuple[str, str, Callable]]:
        pass

    def _register_tools(self):
        """Register MCP tools"""
        for name, description, tool in self.tools:
            self.mcp.tool(name=name, description=description)(tool)

    def _register_resources(self):
        """Register MCP resources"""
        for uri, name, description, resource in self.resources:
            self.mcp.resource(uri, name=name, description=description)(resource)

    def _register_prompts(self):
        """Register MCP prompts"""
        for name, description, prompt in self.prompts:
            self.mcp.prompt(name=name, description=description)(prompt)

    def run(self, transport: str = "stdio"):
        """Run the MCP server"""
        self.mcp.run(transport=transport)
