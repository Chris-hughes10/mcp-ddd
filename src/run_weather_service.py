import logging

from mcp.server.fastmcp import FastMCP

from servers.weather_service.application.mcp_server import WeatherMCPService
from servers.weather_service.domain.repository.repositories import (
    JsonFileWeatherAlertRepository,
    JsonFileWeatherForecastRepository,
)
from servers.weather_service.domain.service.services import NWSWeatherService
from servers.weather_service.infrastructure.adaptors import make_request

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)

# Create domain layer components
weather_service = NWSWeatherService(make_request)
weather_forecast_repository = JsonFileWeatherForecastRepository()
weather_alert_repository = JsonFileWeatherAlertRepository()

# Create and configure application service
weather_mcp = WeatherMCPService(
    mcp=FastMCP("weather_service"),
    weather_service=weather_service,
    weather_forecast_repository=weather_forecast_repository,
    weather_alert_repository=weather_alert_repository,
)

# Run the service
print("Starting Weather MCP Service...")
weather_mcp.run()
