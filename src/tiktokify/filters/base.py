"""Base filter interface for pipeline filters."""

from abc import ABC, abstractmethod


class BaseFilter[T](ABC):
    """Base interface for all pipeline filters.

    Generic type T represents the type of items being filtered.
    - URLFilter: T = str (URLs)
    - ContentFilter: T = Post (crawled posts)
    """

    @abstractmethod
    async def filter(self, items: list[T]) -> tuple[list[T], list[tuple[T, str]]]:
        """Filter items of type T.

        Args:
            items: Input items to filter

        Returns:
            Tuple of (passed_items, rejected_items_with_reasons)
            where rejected_items_with_reasons is a list of (item, reason) tuples
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Filter name for logging and identification."""
        pass


# Type alias using Python 3.12+ syntax
type FilterResult[T] = tuple[list[T], list[tuple[T, str]]]
