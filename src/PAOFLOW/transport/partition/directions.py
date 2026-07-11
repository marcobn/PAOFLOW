from __future__ import annotations

from typing import Literal

TransportDirection = Literal['x', 'y', 'z']

_DIRECTION_AXES: dict[str, int] = {'x': 1, 'y': 2, 'z': 3}


def normalize_transport_direction(direction: str) -> TransportDirection:
    if not isinstance(direction, str):
        raise ValueError("transport_direction must be one of 'x', 'y', or 'z'.")
    direction_normalized = direction.strip().lower()
    if direction_normalized not in _DIRECTION_AXES:
        raise ValueError("transport_direction must be one of 'x', 'y', or 'z'.")
    return direction_normalized  # type: ignore[return-value]


def direction_axis(direction: str) -> int:
    return _DIRECTION_AXES[normalize_transport_direction(direction)]


def direction_index(direction: str) -> int:
    return direction_axis(direction) - 1
