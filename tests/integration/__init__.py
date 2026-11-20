from __future__ import annotations

import json
import re
from pathlib import Path


def uncatalogued_files(outdir: Path) -> list[str]:
    """Output files of a run that the results catalogue does not know.

    A file of the run record needs a term (``kind``) of ``KINDS`` or a place in ``SKIPPED_FILES``
    (``src/PAOFLOW/results/catalogue.py``); see ``.agents/skills/fair-records/SKILL.md``.

    Parameters
    ----------
    outdir : Path
        Output folder of a run, holding ``paoflow_outputs.json``.

    Returns
    -------
    list of str
        Paths relative to ``outdir``; empty when every file is catalogued.
    """
    from PAOFLOW.results.catalogue import SKIPPED_FILES

    record = json.loads((Path(outdir) / 'paoflow_outputs.json').read_text())
    return sorted(
        {
            entry['file']
            for run in record['runs']
            for entry in run['files']
            if 'kind' not in entry and not re.fullmatch(SKIPPED_FILES, Path(entry['file']).name)
        }
    )
