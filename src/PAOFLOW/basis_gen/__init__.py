"""Pseudo-atom basis generator (norm-conserving v1).

Solves the radial Schroedinger equation in a confining box using the
operator stored in a UPF (local potential + Kleinman-Bylander projectors),
to produce smooth radial functions for arbitrary (n, l[, j]) channels.
"""

from .driver import generate_basis_for_directory, generate_basis_for_pseudo
from .radial import pseudize_shell, solve_radial_channel

__all__ = [
    'generate_basis_for_directory',
    'generate_basis_for_pseudo',
    'pseudize_shell',
    'solve_radial_channel',
]
