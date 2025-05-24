import datetime
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class WeatherPeriod:
    """Represents a single period in a weather forecast.

    A weather period typically corresponds to a 12-hour timeframe
    (e.g., "Tuesday Night", "Wednesday") and contains the core
    meteorological data for that period.

    Attributes:
        name: Human-readable period name (e.g., "Tonight", "Tuesday")
        temperature: Temperature value in the specified unit
        temperature_unit: Temperature unit ("F" for Fahrenheit, "C" for Celsius)
        wind_speed: Wind speed description (e.g., "5 to 10 mph")
        wind_direction: Wind direction abbreviation (e.g., "NW", "SSE")
        detailed_forecast: Complete narrative forecast for this period
    """

    name: str
    temperature: float
    temperature_unit: str
    wind_speed: str
    wind_direction: str
    detailed_forecast: str

    def to_display_string(self) -> str:
        """Format for human-readable display"""
        return f"{self.name}: {self.temperature}°{self.temperature_unit} - {self.detailed_forecast}"


@dataclass
class Forecast:
    """Weather forecast aggregate containing multiple time periods.

    Represents a complete weather forecast for a location, typically
    covering the next 5-7 periods (2-3 days). Follows DDD aggregate
    pattern where Forecast is the aggregate root.

    Attributes:
        periods: List of forecast periods, ordered chronologically
        error: Error message if forecast retrieval failed, None if successful
        latitude: Location latitude in decimal degrees
        longitude: Location longitude in decimal degrees
        retrieved_at: UTC timestamp when forecast was fetched
    """

    periods: List[WeatherPeriod]
    error: str
    latitude: float
    longitude: float
    retrieved_at: datetime.datetime = None

    def to_display_string(self) -> str:
        """Format for human-readable display"""
        if not self.periods:
            return f"Error: {self.error or 'No forecast data available'}"

        return (
            f"Forecast retrieved at: {self.retrieved_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            + f"\nLocation: {self.latitude:.4f}, {self.longitude:.4f}"
            + "\n".join(period.to_display_string() for period in self.periods)
        )


@dataclass
class WeatherAlert:
    """Individual weather alert issued by meteorological authorities.

    Represents a single weather warning, watch, or advisory (e.g., tornado
    warning, flood watch, heat advisory). Alerts are value objects in DDD
    terms - they're identified by their content rather than a unique ID.

    Attributes:
        event: Alert type/category (e.g., "Tornado Warning", "Heat Advisory")
        area: Geographic description of affected area (e.g., "Harris County, TX")
        severity: Alert severity level ("Minor", "Moderate", "Severe", "Extreme")
        description: Full alert description with details and impacts
        instructions: Safety instructions for the public (None if not provided)
    """

    event: str
    area: str
    severity: str
    description: str
    instructions: Optional[str] = None

    def to_display_string(self) -> str:
        """Format alert for display"""
        result = f"Event: {self.event}\nArea: {self.area}\nSeverity: {self.severity}"
        result += f"\nDescription: {self.description}"

        if self.instructions:
            result += f"\nInstructions: {self.instructions}"

        return result


@dataclass
class AlertSnapshot:
    """Collection of weather alerts for a specific area and time.

    Aggregate root for weather alerts, grouping related alerts that were
    active for a geographic area at a specific point in time. This allows
    tracking alert history and provides context for alert analysis.

    In DDD terms, this is an aggregate that ensures consistency across
    related alerts and provides a clear boundary for persistence operations.

    Attributes:
        alerts: List of individual weather alerts active at retrieval time
        retrieved_at: UTC timestamp when alerts were fetched from service
        state: Two-letter US state code for geographic grouping

    Business Rules:
        - AlertSnapshot represents alerts at a point in time (immutable snapshot)
        - Empty alerts list indicates no active alerts (not an error condition)
        - State code used for geographic partitioning in storage
    """

    alerts: List[WeatherAlert]
    retrieved_at: datetime.datetime
    state: str

    def to_display_string(self) -> str:
        """Format alert set for display"""
        if not self.alerts:
            return f"No active alerts for {self.state} at {self.retrieved_at.strftime('%Y-%m-%d %H:%M:%S')}"

        result = f"Alerts for {self.state} at {self.retrieved_at.strftime('%Y-%m-%d %H:%M:%S')}:\n"
        result += f"Found {len(self.alerts)} active alert(s)\n\n"

        return result + "\n---\n".join(
            alert.to_display_string() for alert in self.alerts
        )
