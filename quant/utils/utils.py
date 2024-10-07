"""Common utility functions."""

from typing import Any


def noop(*args: Any, **kwargs: Any) -> None:
    """No-op that returns None."""
    return None
