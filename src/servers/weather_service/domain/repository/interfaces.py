from abc import ABC, abstractmethod
from datetime import timedelta
from typing import List, Optional, Union

from servers.weather_service.domain.models import AlertSnapshot, Forecast, WeatherAlert


class WeatherForecastRepository(ABC):
    """Repository for weather forecast aggregate persistence.

    Stores and retrieves complete Forecast aggregates by geographic location
    and time. Supports location-based queries and time-windowed filtering
    for both current and historical forecast analysis.
    """

    @abstractmethod
    async def get_forecasts(
        self,
        latitude: float,
        longitude: float,
        time_window: Union[int, timedelta] = 3,
        time_unit: str = "hours",
        limit: Optional[int] = None,
    ) -> List[Forecast]:
        """Find forecasts for a location within a time window.

        Args:
            latitude: Geographic latitude in decimal degrees
            longitude: Geographic longitude in decimal degrees
            time_window: How far back to search (int uses time_unit)
            time_unit: "hours" or "days" when time_window is int
            limit: Maximum forecasts to return

        Returns:
            List of complete Forecast aggregates, newest-first
        """
        pass

    @abstractmethod
    async def save_forecast(
        self, latitude: float, longitude: float, forecast: Forecast
    ) -> None:
        """Save a complete forecast aggregate.

        Args:
            latitude: Geographic latitude
            longitude: Geographic longitude
            forecast: Complete Forecast with all periods and metadata
        """
        pass


class WeatherAlertRepository(ABC):
    """Repository for weather alert snapshot aggregate persistence.

    Manages AlertSnapshot aggregates that capture the complete set of alerts
    active for a state at specific points in time. Enables historical alert
    pattern analysis and trend monitoring.
    """

    @abstractmethod
    async def get_alerts(
        self,
        state: str,
        time_window: Union[int, timedelta] = 24,
        time_unit: str = "hours",
        limit: Optional[int] = None,
    ) -> List[AlertSnapshot]:
        """Find alert sets for a state within a time window

        Args:
            state: Two-letter US state code
            time_window: Number of time units to look back
            time_unit: Either "hours" or "days"
            limit: Maximum number of alert sets to return

        Returns:
            List of alert sets matching criteria, sorted newest first
        """
        pass

    @abstractmethod
    async def save_alerts(
        self, state: str, alerts: List[WeatherAlert]
    ) -> AlertSnapshot:
        """Save alerts as a new alert set

        Args:
            state: Two-letter US state code
            alerts: List of alerts to save (empty list = no alerts)

        Returns:
            The created AlertSnapshot
        """
        pass
