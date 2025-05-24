from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from servers.weather_service.domain.models import (
    AlertSnapshot,
    Forecast,
    WeatherAlert,
    WeatherPeriod,
)
from servers.weather_service.domain.repository.interfaces import (
    WeatherAlertRepository,
    WeatherForecastRepository,
)
from servers.weather_service.infrastructure.repositories import (
    TimestampedCollectionRepository,
)


class JsonFileWeatherForecastRepository(
    TimestampedCollectionRepository[Forecast, str], WeatherForecastRepository
):
    """File-based weather forecast repository using JSON persistence.

    Stores forecasts in a local JSON file with automatic rotation and
    size limits. Suitable for development, testing, and small deployments.
    For production use, consider database-backed implementations.

    Args:
        file_path: Path to JSON storage file (created if doesn't exist)
        max_forecasts_per_location: Maximum forecasts to retain per location
            (older forecasts are automatically purged)

    File Format:
        JSON object with location keys mapping to arrays of forecast objects.
        Example: {"location:40.7128:-74.0060": [forecast1, forecast2, ...]}
    """

    def __init__(
        self, file_path: str = "weather_data.json", max_forecasts_per_location: int = 10
    ):
        """Initialize repository"""
        super().__init__(file_path, max_forecasts_per_location)

    async def get_forecasts(
        self,
        latitude: float,
        longitude: float,
        time_window: Union[int, timedelta] = 3,
        time_unit: str = "hours",
        limit: Optional[int] = None,
    ) -> List[Forecast]:
        """Find forecasts for a location within a time window"""
        key = self._make_location_key(latitude, longitude)
        return await self.find_items(key, time_window, time_unit, limit)

    async def save_forecast(
        self, latitude: float, longitude: float, forecast: Forecast
    ) -> None:
        """Save forecast data"""
        # Ensure forecast has location and timestamp data
        forecast.latitude = latitude
        forecast.longitude = longitude
        if not forecast.retrieved_at:
            forecast.retrieved_at = datetime.now()

        key = self._make_location_key(latitude, longitude)
        await self.save_item(key, forecast)

    def _make_location_key(self, latitude: float, longitude: float) -> str:
        """Create a unique key for location"""
        return f"location:{latitude:.4f}:{longitude:.4f}"

    def _serialize_item(self, forecast: Forecast) -> Dict[str, Any]:
        """Convert a Forecast to a serializable dictionary"""
        return {
            "periods": [
                self._serialize_weather_period(period) for period in forecast.periods
            ],
            "error": forecast.error,
            "retrieved_at": forecast.retrieved_at.isoformat()
            if forecast.retrieved_at
            else None,
            "latitude": forecast.latitude,
            "longitude": forecast.longitude,
        }

    def _deserialize_item(self, data: Dict[str, Any]) -> Forecast:
        """Create a Forecast from a dictionary"""
        return Forecast(
            periods=[
                self._deserialize_weather_period(period) for period in data["periods"]
            ],
            error=data["error"],
            retrieved_at=datetime.fromisoformat(data["retrieved_at"])
            if data["retrieved_at"]
            else None,
            latitude=data.get("latitude"),
            longitude=data.get("longitude"),
        )

    def _deserialize_key(self, key_str: str) -> str:
        """Keys are already strings"""
        return key_str

    def _serialize_weather_period(self, period: WeatherPeriod) -> dict:
        """Convert a WeatherPeriod to a serializable dictionary"""
        return {
            "name": period.name,
            "temperature": period.temperature,
            "temperature_unit": period.temperature_unit,
            "wind_speed": period.wind_speed,
            "wind_direction": period.wind_direction,
            "detailed_forecast": period.detailed_forecast,
        }

    def _deserialize_weather_period(self, data: dict) -> WeatherPeriod:
        """Create a WeatherPeriod from a dictionary"""
        return WeatherPeriod(
            name=data["name"],
            temperature=data["temperature"],
            temperature_unit=data["temperature_unit"],
            wind_speed=data["wind_speed"],
            wind_direction=data["wind_direction"],
            detailed_forecast=data["detailed_forecast"],
        )


class JsonFileWeatherAlertRepository(
    TimestampedCollectionRepository[AlertSnapshot, str], WeatherAlertRepository
):
    """Repository for weather alerts using JSON file storage"""

    def __init__(
        self, file_path: str = "weather_alerts.json", max_sets_per_state: int = 20
    ):
        """Initialize repository"""
        super().__init__(file_path, max_sets_per_state)

    async def get_alerts(
        self,
        state: str,
        time_window: Union[int, timedelta] = 24,
        time_unit: str = "hours",
        limit: Optional[int] = None,
    ) -> List[AlertSnapshot]:
        """Find alert sets for a state within a time window"""
        key = self._make_state_key(state)
        return await self.find_items(key, time_window, time_unit, limit)

    async def save_alerts(
        self, state: str, alerts: List[WeatherAlert]
    ) -> AlertSnapshot:
        """Save alerts as a new alert set"""
        # Create new alert set
        alert_set = AlertSnapshot(
            alerts=alerts, retrieved_at=datetime.now(), state=state.upper()
        )

        key = self._make_state_key(state)
        return await self.save_item(key, alert_set)

    def _make_state_key(self, state: str) -> str:
        """Create a unique key for state"""
        return f"state:{state.upper()}"

    def _serialize_item(self, alert_set: AlertSnapshot) -> Dict[str, Any]:
        """Convert an AlertSnapshot to a serializable dictionary"""
        return {
            "state": alert_set.state,
            "retrieved_at": alert_set.retrieved_at.isoformat(),
            "alerts": [self._serialize_alert(alert) for alert in alert_set.alerts],
        }

    def _deserialize_item(self, data: Dict[str, Any]) -> AlertSnapshot:
        """Create an AlertSnapshot from a dictionary"""
        return AlertSnapshot(
            state=data["state"],
            retrieved_at=datetime.fromisoformat(data["retrieved_at"]),
            alerts=[
                self._deserialize_alert(alert_data) for alert_data in data["alerts"]
            ],
        )

    def _deserialize_key(self, key_str: str) -> str:
        """Keys are already strings"""
        return key_str

    def _serialize_alert(self, alert: WeatherAlert) -> dict:
        """Convert a WeatherAlert to a serializable dictionary"""
        return {
            "event": alert.event,
            "area": alert.area,
            "severity": alert.severity,
            "description": alert.description,
            "instructions": alert.instructions,
        }

    def _deserialize_alert(self, data: dict) -> WeatherAlert:
        """Create a WeatherAlert from a dictionary"""
        return WeatherAlert(
            event=data["event"],
            area=data["area"],
            severity=data["severity"],
            description=data["description"],
            instructions=data.get("instructions"),
        )
