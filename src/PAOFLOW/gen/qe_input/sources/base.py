"""Base class and shared helpers for database source adapters."""

import urllib.error
import urllib.request


def download_text(url, timeout=30):
    """Return the decoded text body of *url* or raise a RuntimeError."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode('utf-8', errors='replace')
    except urllib.error.HTTPError as exc:
        raise RuntimeError('HTTP {} fetching {}'.format(exc.code, url)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError('Could not fetch {}: {}'.format(url, exc.reason)) from exc


class DatabaseSource:
    """Abstract adapter mapping a database identifier to a ``MaterialRecord``.

    Concrete subclasses set :attr:`name` and implement :meth:`matches` (used by
    ``--source auto`` detection) and :meth:`fetch`.
    """

    #: Short source label, also the value accepted by ``--source``.
    name = ''

    def matches(self, identifier):
        """Return ``True`` if *identifier* belongs to this database."""
        raise NotImplementedError

    def fetch(self, identifier, **options):
        """Download and normalize *identifier* into a ``MaterialRecord``."""
        raise NotImplementedError
