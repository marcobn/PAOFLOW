#!/usr/bin/env python3
"""Backward-compatible entry point for the QE-input generator.

The implementation now lives in :mod:`PAOFLOW.gen.qe_input`, which supports
multiple materials databases (AFLOW, C2DB, ...).  This module is preserved so
that the ``paoflow-gen-qe`` console script and any existing imports keep
working; ``main`` defaults to ``--source auto`` so historical AFLOW invocations
behave exactly as before.
"""

from .qe_input.cli import main
from .qe_input.record import MaterialRecord
from .qe_input.sources.aflow import (
    aurl_to_url,
    entry_file_url,
    is_magnetic,
    parse_contcar_qe,
    resolve_auid,
    resolve_entry_url,
    resolve_species,
)
from .qe_input.sources.base import download_text
from .qe_input.writer import build_qe_input

__all__ = [
    'main',
    'MaterialRecord',
    'build_qe_input',
    'download_text',
    'resolve_auid',
    'resolve_entry_url',
    'aurl_to_url',
    'entry_file_url',
    'parse_contcar_qe',
    'resolve_species',
    'is_magnetic',
]


if __name__ == '__main__':
    raise SystemExit(main())
