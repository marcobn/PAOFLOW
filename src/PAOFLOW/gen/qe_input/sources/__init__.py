"""Database source registry.

Adding a new database = implement a :class:`DatabaseSource` subclass in this
package and register it in :data:`_SOURCES` below.
"""

from .aflow import AflowSource
from .base import DatabaseSource
from .c2db import C2dbSource

# Ordered so that ``detect_source`` prefers the most specific match first.
_SOURCES = [AflowSource(), C2dbSource()]


def available_sources():
    """Return the list of registered source names."""
    return [s.name for s in _SOURCES]


def get_source(name):
    """Return the source adapter registered under *name*."""
    for source in _SOURCES:
        if source.name == name:
            return source
    raise RuntimeError(
        "Unknown source '{}'. Available: {}".format(name, ', '.join(available_sources()))
    )


def detect_source(identifier):
    """Return the adapter whose :meth:`matches` accepts *identifier*.

    Raises ``RuntimeError`` when no registered source recognizes it.
    """
    for source in _SOURCES:
        if source.matches(identifier):
            return source
    raise RuntimeError(
        "Could not auto-detect a database for '{}'. Pass --source explicitly ({}).".format(
            identifier, ', '.join(available_sources())
        )
    )


__all__ = [
    'DatabaseSource',
    'AflowSource',
    'C2dbSource',
    'available_sources',
    'get_source',
    'detect_source',
]
