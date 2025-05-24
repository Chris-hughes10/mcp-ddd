import datetime
from typing import Any, Dict, List

from servers.weather_service.domain.models import Forecast, WeatherAlert, WeatherPeriod
from servers.weather_service.domain.service.interfaces import WeatherService
from servers.weather_service.infrastructure.adaptors import make_request


class NWSWeatherService(WeatherService):
    """National Weather Service implementation of WeatherService.

    Integrates with the NWS API (weather.gov) which provides free
    weather data for US locations. Uses the two-step process:
    1. GET /points/{lat},{lon} to get forecast endpoint
    2. GET forecast endpoint to retrieve actual forecast data

    Args:
        make_http_request: HTTP client function for dependency injection
    """

    def __init__(self, make_http_request=make_request):
        self.get = make_http_request
        self.base_url = "https://api.weather.gov"
        self.headers = {
            "User-Agent": "weather-app/1.0",
            "Accept": "application/geo+json",
        }

    async def get_forecast(self, latitude: float, longitude: float) -> Forecast:
        """Get weather forecast for a location.

        Args:
            latitude: Latitude of the location
            longitude: Longitude of the location
        """
        # First get the forecast grid endpoint
        points_url = f"{self.base_url}/points/{latitude},{longitude}"
        request_time = datetime.datetime.now()
        points_data = await self.get(points_url, headers=self.headers)

        if not points_data:
            return Forecast(
                periods=[],
                error="Unable to fetch forecast data for this location.",
                retrieved_at=request_time,
                latitude=latitude,
                longitude=longitude,
            )

        # Get the forecast URL from the points response
        forecast_url = points_data["properties"]["forecast"]
        forecast_data = await self.get(forecast_url, headers=self.headers)

        if not forecast_data:
            return Forecast(
                periods=[],
                error="Unable to fetch detailed forecast.",
                retrieved_at=request_time,
                latitude=latitude,
                longitude=longitude,
            )

        # Convert API data to Domain objects
        periods = []
        for period in forecast_data["properties"]["periods"][:5]:
            periods.append(
                WeatherPeriod(
                    name=period["name"],
                    temperature=period["temperature"],
                    temperature_unit=period["temperatureUnit"],
                    wind_speed=period["windSpeed"],
                    wind_direction=period["windDirection"],
                    detailed_forecast=period["detailedForecast"],
                )
            )

        return Forecast(
            periods=periods,
            error=None,
            retrieved_at=request_time,
            latitude=latitude,
            longitude=longitude,
        )

    async def get_alerts(self, state: str) -> List[WeatherAlert]:
        """Get weather alerts for a US state.

        Args:
            state: Two-letter US state code (e.g. CA, NY)
        """
        data = await self.get(
            f"{self.base_url}/alerts/active/area/{state}", headers=self.headers
        )

        if not data or "features" not in data:
            return []  # "Unable to fetch alerts or no alerts found."

        if not data["features"]:
            return []  # "No active alerts for this state."

        alerts = []
        for feature in data["features"]:
            props = feature["properties"]
            alerts.append(
                WeatherAlert(
                    event=props.get("event", "Unknown"),
                    area=props.get("areaDesc", "Unknown"),
                    severity=props.get("severity", "Unknown"),
                    description=props.get("description", "No description available"),
                    instructions=props.get("instruction"),
                )
            )

        return alerts
