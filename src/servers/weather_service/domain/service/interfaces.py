from abc import ABC, abstractmethod
from typing import List

from servers.weather_service.domain.models import Forecast, WeatherAlert


class WeatherService(ABC):
    """Abstract interface for weather data retrieval services.

    Defines the domain contract for weather operations. Implementations
    should handle external API communication while maintaining domain
    model consistency.

    This follows the DDD domain service pattern for operations that
    don't naturally belong to a single entity or value object.
    """

    @abstractmethod
    async def get_forecast(self, latitude: float, longitude: float) -> Forecast:
        """Retrieve weather forecast for geographic coordinates.

        Args:
            latitude: Latitude in decimal degrees, range [-90, 90]
            longitude: Longitude in decimal degrees, range [-180, 180]

        Returns:
            Forecast aggregate with periods and metadata
        """
        pass

    @abstractmethod
    async def get_alerts(self, state: str) -> List[WeatherAlert]:
        """Retrieve active weather alerts for a US state.

        Args:
            state: Two-letter US state code (e.g., "CA", "TX", "FL")
                Must be normalized to uppercase by implementations
                Invalid state codes should return empty list, not error

        Returns:
            List of WeatherAlert domain objects representing all currently
            active alerts for the specified state. Empty list indicates no
            active alerts (normal condition, not an error).
        """
        pass
