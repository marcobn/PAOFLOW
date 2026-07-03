"""Database-agnostic material record shared by every QE-input source.

Each database adapter (AFLOW, C2DB, ...) downloads and normalizes one entry
into a :class:`MaterialRecord`.  The QE writer in
:mod:`PAOFLOW.gen.qe_input.writer` consumes only this record, so the
quantum-espresso side of the generator is completely decoupled from the
provenance of the data.
"""

from dataclasses import dataclass, field


@dataclass
class MaterialRecord:
    """Normalized description of a single database entry.

    Attributes
    ----------
    compound :
        Short name used as the QE ``prefix`` and default output stem.
    geometry :
        Verbatim QE geometry cards in the ``parse_contcar_qe`` shape, with
        keys ``cell_header``, ``cell_unit``, ``cell_rows``, ``pos_header``,
        ``pos_unit``, ``pos_rows`` and ``atom_order``.  Both AFLOW and C2DB
        normalize their structure into this dict so the writer reuses a
        single assembly path.
    species :
        Ordered ``[(element, count), ...]`` list (count may be ``None``).
    natoms :
        Number of atoms in the cell.
    metallic :
        ``True`` when the system needs Gaussian smearing (no gap).
    magnetic :
        ``True`` when a collinear spin-polarized input is requested.
    dimensionality :
        ``'3D'`` or ``'2D'``.  ``'2D'`` triggers the vacuum-on-c handling in
        the writer (proper in-plane ibrav, ``K_POINTS`` kz=1, and optional
        ``assume_isolated='2D'``).
    kpoints :
        Optional ``(k1, k2, k3)`` Monkhorst-Pack grid hint.  ``None`` lets
        the writer fall back to its default.
    energy_cutoff :
        Optional ``ecutwfc`` fallback in Rydberg when ``reference.json`` does
        not cover every species.
    bravais_hint :
        Optional Bravais-lattice symbol (e.g. AFLOW ``'FCC'``) passed to the
        ibrav detector.
    spacegroup :
        Optional space-group number/symbol passed to the ibrav detector.
    source :
        Provenance label (``'aflow'``, ``'c2db'``, ...), used only for
        messages and header comments.
    """

    compound: str
    geometry: dict
    species: list = field(default_factory=list)
    natoms: int = 0
    metallic: bool = True
    magnetic: bool = False
    dimensionality: str = '3D'
    kpoints: tuple = None
    energy_cutoff: float = None
    bravais_hint: str = None
    spacegroup: object = None
    source: str = ''

    @property
    def is_2d(self):
        """``True`` when the entry is a two-dimensional material."""
        return self.dimensionality.upper() == '2D'
