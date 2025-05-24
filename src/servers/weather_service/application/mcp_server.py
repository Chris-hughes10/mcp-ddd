from typing import Callable, List, Tuple

from mcp.server.fastmcp import FastMCP

from servers.weather_service.domain.repository.interfaces import (
    WeatherAlertRepository,
    WeatherForecastRepository,
)
from servers.weather_service.domain.service.interfaces import WeatherService
from servers.weather_service.infrastructure.application import MCPApplicationService


class WeatherMCPService(MCPApplicationService):
    """MCP Application Service for weather domain operations.

    Exposes weather domain capabilities through the Model Context Protocol.
    Orchestrates WeatherService and Repository operations while maintaining
    separation from MCP protocol details.

    Available Tools:
        - get_forecast: Real-time weather forecast retrieval
        - get_alerts: Current weather alerts for US states

    Available Resources:
        - historical://alerts/{state}: Historical alert data

    Available Prompts:
        - weather_analysis_prompt: Structured weather analysis template
    """

    def __init__(
        self,
        mcp: FastMCP,
        weather_service: WeatherService,
        weather_forecast_repository: WeatherForecastRepository,
        weather_alert_repository: WeatherAlertRepository,
    ):
        self.weather_service = weather_service
        self.weather_forecast_repository = weather_forecast_repository
        self.weather_alert_repository = weather_alert_repository
        super().__init__(mcp)

    @property
    def tools(self) -> List[Tuple[str, str, Callable]]:
        """Register tools for the MCP server"""
        return [
            (
                "get_forecast",
                "Get weather forecast for a location",
                self.get_forecast,
            ),
            (
                "get_alerts",
                "Get weather alerts for a US state",
                self.get_alerts,
            ),
        ]

    @property
    def resources(self) -> List[Tuple[str, str, str, Callable]]:
        """Register resources for the MCP server"""
        return [
            (
                "historical://alerts/{state}",
                "Get historical weather alerts for a US state",
                "Get historical weather alerts for a US state. State should be provided as a two-letter US state code (e.g. CA, NY). Do not use this for current alerts.",
                self.get_historical_alerts,
            ),
        ]

    @property
    def prompts(self) -> List[Tuple[str, str, Callable]]:
        """Register prompts for the MCP server"""
        return [
            (
                "weather_analysis",
                "Analyze weather conditions for a location",
                self.weather_analysis_prompt,
            ),
        ]

    def weather_analysis_prompt(self, location: str) -> str:
        """
        A prompt template for analyzing weather conditions for a location.

        Args:
            location: The location to analyze weather for
        """
        return f"""
        You are a weather analysis expert providing detailed insights about weather conditions.
        
        Analyze the current and forecasted weather conditions for {location}. 
        Include information about:
        - Current temperatures and conditions
        - Expected changes over the next few days
        - Any notable weather patterns or anomalies
        - Practical advice based on the conditions
        
        Present your analysis in a clear, structured format that's easy to understand.
        """

    async def get_forecast(self, latitude: float, longitude: float) -> str:
        """MCP tool: Get weather forecast for coordinates.

        Retrieves current forecast from weather service and persists
        to repository for historical tracking. Coordinates are validated
        and forecast is formatted for LLM consumption.

        Args:
            latitude: Decimal degrees latitude [-90, 90]
            longitude: Decimal degrees longitude [-180, 180]

        Returns:
            Human-readable forecast string with periods and metadata

        Note:
            This is an MCP tool function - return values should be
            formatted for LLM understanding rather than programmatic use.
        """
        forecast = await self.weather_service.get_forecast(latitude, longitude)
        await self.weather_forecast_repository.save_forecast(
            latitude, longitude, forecast
        )
        return forecast.to_display_string()

    async def get_alerts(self, state: str) -> str:
        """Retrieve active weather alerts for a US state.

        Fetches current weather warnings, watches, and advisories from the
        National Weather Service for the specified state. Returns all active
        alerts regardless of severity level.

        Args:
            state: Two-letter US state code (e.g., "CA", "TX", "FL")
                   Case-insensitive, automatically normalized to uppercase

        Returns:
            List of active WeatherAlert objects, empty list if no alerts

        """
        alerts = await self.weather_service.get_alerts(state)

        if not alerts:
            return f"No active alerts for {state}"

        await self.weather_alert_repository.save_alerts(state, alerts)

        return "\n---\n".join(alert.to_display_string() for alert in alerts)

    async def get_historical_alerts(self, state: str) -> str:
        """MCP resource handler for historical weather alerts.

        Provides access to recently cached weather alerts through the MCP
        resource system. This allows LLMs to access historical alert data
        for analysis and comparison.

        URI Template: historical://alerts/{state}

        Args:
            state: Two-letter US state code (e.g., "CA", "TX")

        Returns:
            Formatted historical alerts or "no data" message

        Note:
            This is a resource handler, not a tool. Resources provide
            read-only access to data, while tools perform actions.
        """
        alerts = await self.weather_alert_repository.get_alerts(
            state=state,
        )

        if not alerts:
            return f"No historical alerts found for {state}"

        return "\n\n===\n\n".join(alert_set.to_display_string() for alert_set in alerts)
