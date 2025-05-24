import json
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Generic, List, Optional, TypeVar, Union


class JsonFileRepository(ABC):
    """Abstract base class for JSON file-based repositories"""

    def __init__(self, file_path: str):
        """Initialize repository and load existing data

        Args:
            file_path: Path to the JSON file for persistence
        """
        self.file_path = Path(file_path)
        data = self._load_from_file()  # Capture the returned data
        self._deserialize_data(data)  # Process the data into collections

    def _save_to_file(self, serializable_data: Dict[str, Any]) -> None:
        """Save data to JSON file

        Args:
            serializable_data: Dictionary of serialized data to save
        """
        # Create directory if it doesn't exist
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        with open(self.file_path, "w") as f:
            json.dump(serializable_data, f, indent=2)

    def _load_from_file(self) -> Dict[str, Any]:
        """Load data from JSON file if it exists

        Returns:
            Dictionary of loaded data or empty dict if file doesn't exist
        """
        if not self.file_path.exists():
            return {}  # No file to load

        try:
            with open(self.file_path, "r") as f:
                loaded_data = json.load(f)
                return loaded_data

        except (json.JSONDecodeError, IOError) as e:
            # Log error but continue with empty repository
            print(f"Error loading data from {self.file_path}: {e}")
            return {}

    @abstractmethod
    def _serialize_data(self) -> Dict[str, Any]:
        """Convert in-memory data to serializable dictionary

        Returns:
            Dictionary representation of the data
        """
        pass

    @abstractmethod
    def _deserialize_data(self, data: Dict[str, Any]) -> None:
        """Load data from serialized dictionary into memory

        Args:
            data: Serialized data to load
        """
        pass


# Define generic type variables
T = TypeVar("T")  # For domain objects (Forecast, AlertSnapshot)
K = TypeVar("K")  # For key types (str for state, str for location)


class TimestampedCollectionRepository(JsonFileRepository, Generic[T, K]):
    """Generic repository for timestamped domain objects with TTL support.

    Implements a generic pattern for storing collections of timestamped
    objects with automatic expiration and
    size limits. Uses the Repository pattern from DDD to abstract
    persistence concerns.

    Type Parameters:
        T: Domain object type (must have timestamp attribute)
        K: Key type for grouping objects (str for location keys)

    Args:
        file_path: JSON file path for persistence
        max_items_per_key: Maximum objects to retain per key
    """

    def __init__(self, file_path: str, max_items_per_key: int = 20):
        """Initialize repository"""
        self.collections: Dict[K, Deque[T]] = {}
        self.max_items_per_key = max_items_per_key
        super().__init__(file_path)

    async def find_items(
        self,
        key: K,
        time_window: Union[int, timedelta] = 24,
        time_unit: str = "hours",
        limit: Optional[int] = None,
        timestamp_getter: Callable[[T], datetime] = lambda x: x.retrieved_at,
    ) -> List[T]:
        """Find items within a time window, newest first.

        Args:
            key: Grouping key (e.g., location identifier)
            time_window: How far back to look (int or timedelta)
            time_unit: Units for time_window if int ("hours" or "days")
            limit: Maximum items to return (None for no limit)
            timestamp_getter: Function to extract timestamp from items

        Returns:
            List of items matching criteria, ordered newest to oldest
        """
        if key not in self.collections:
            return []

        # Calculate cutoff time
        cutoff_time = self._calculate_cutoff_time(time_window, time_unit)

        # Filter items by time
        valid_items = [
            item
            for item in self.collections[key]
            if timestamp_getter(item) >= cutoff_time
        ]

        # Apply limit if specified
        if limit is not None and limit > 0:
            valid_items = valid_items[:limit]

        return valid_items

    async def save_item(self, key: K, item: T) -> T:
        """Save an item under the given key"""
        # Initialize deque if this is the first item for this key
        if key not in self.collections:
            self.collections[key] = deque(maxlen=self.max_items_per_key)

        # Add new item at the beginning (newest first)
        self.collections[key].appendleft(item)

        # Save to file
        self._save_to_file(self._serialize_data())

        return item

    async def get_most_recent_item(self, key: K) -> Optional[T]:
        """Get the most recent item for a key"""
        if key not in self.collections or not self.collections[key]:
            return None

        return self.collections[key][0]  # First item is most recent

    def _calculate_cutoff_time(
        self, time_window: Union[int, timedelta], time_unit: str
    ) -> datetime:
        """Calculate cutoff time based on window and unit"""
        if isinstance(time_window, timedelta):
            return datetime.now() - time_window

        if time_unit == "hours":
            return datetime.now() - timedelta(hours=time_window)
        elif time_unit == "days":
            return datetime.now() - timedelta(days=time_window)
        else:
            raise ValueError(
                f"Invalid time_unit: {time_unit}, must be 'hours' or 'days'"
            )

    def _serialize_data(self) -> Dict[str, Any]:
        """Convert in-memory collections to serializable dictionary"""
        serializable_data = {}

        for key, items in self.collections.items():
            # Convert deque to list and serialize each item
            serializable_data[str(key)] = [self._serialize_item(item) for item in items]

        return serializable_data

    def _deserialize_data(self, data: Dict[str, Any]) -> None:
        """Load data from serialized dictionary into memory"""
        for key_str, items_data in data.items():
            key = self._deserialize_key(key_str)
            # Create a deque with maxlen for each key
            self.collections[key] = deque(
                [self._deserialize_item(item) for item in items_data],
                maxlen=self.max_items_per_key,
            )

    @abstractmethod
    def _serialize_item(self, item: T) -> Dict[str, Any]:
        """Convert an item to a serializable dictionary"""
        pass

    @abstractmethod
    def _deserialize_item(self, data: Dict[str, Any]) -> T:
        """Create an item from a dictionary"""
        pass

    @abstractmethod
    def _deserialize_key(self, key_str: str) -> K:
        """Convert a string key back to the appropriate type"""
        pass
