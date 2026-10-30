"""Point-group constraints on third-rank response tensors (shift current, SHG).

Thin wrapper of :mod:`PAOFLOW.utils.point_groups`, which also provides the k-grid reduction
used by the non-linear optics engine; kept so that the workflow scripts keep importing
``PAOFLOW.agentic.nlo.symmetry``.
"""

from PAOFLOW.utils.point_groups import (
    ALIASES,
    GENERATORS,
    LABELS,
    cartesian_from_fractional_k,
    close_group,
    component_closure,
    forbidden_components,
    fractional_k_from_cartesian,
    independent_relations,
    kleinman_error,
    point_group_operations,
    reduce_k_grid,
    symmetrize,
    symmetrize_components,
    symmetry_error,
    tensor_from_components,
)

__all__ = [
    'ALIASES',
    'GENERATORS',
    'LABELS',
    'cartesian_from_fractional_k',
    'close_group',
    'component_closure',
    'forbidden_components',
    'fractional_k_from_cartesian',
    'independent_relations',
    'kleinman_error',
    'point_group_operations',
    'reduce_k_grid',
    'symmetrize',
    'symmetrize_components',
    'symmetry_error',
    'tensor_from_components',
]
