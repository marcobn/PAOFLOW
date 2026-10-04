"""``PAOFLOW(..., sparse=...)`` routing: one driver, two engines.

Covers the contract of :mod:`PAOFLOW.sparse.dispatch` without DFT data
(a ``restart=True`` driver needs none until an archive is loaded): every
public method has a role, routed methods keep the dense signature and the
engine accepts it, dense-only methods refuse loudly in a sparse run, and a
dense run is untouched.  The numerical side of the same routing is covered
by the parity tests.
"""

import inspect

import pytest

from PAOFLOW.PAOFLOW import PAOFLOW
from PAOFLOW.sparse import SparseConfig
from PAOFLOW.sparse.config import resolve_threshold
from PAOFLOW.sparse.engine import SparseEngine


def _role(name):
    return getattr(vars(PAOFLOW)[name], '_sparse_role', None)


def _public_methods():
    return [n for n, f in vars(PAOFLOW).items() if not n.startswith('_') and inspect.isfunction(f)]


def _overrides():
    return [n for n in _public_methods() if _role(n) == 'override']


@pytest.fixture
def dense(tmp_path):
    return PAOFLOW(workpath=str(tmp_path), outputdir='dense', restart=True)


@pytest.fixture
def sparse(tmp_path):
    return PAOFLOW(workpath=str(tmp_path), outputdir='sparse', restart=True, sparse=True)


# ----------------------------------------------------------------------
# SparseConfig
# ----------------------------------------------------------------------


def test_config_rejects_two_cutoffs():
    with pytest.raises(ValueError, match='not both'):
        SparseConfig(rcut=5.0, bond_order=2)


@pytest.mark.parametrize('cutoff', [{'rcut': 5.0}, {'bond_order': 2}])
def test_config_rejects_threshold_with_a_cutoff(cutoff):
    with pytest.raises(ValueError, match='threshold cannot be combined'):
        SparseConfig(threshold=1e-3, **cutoff)


def test_config_threshold_defaults_by_truncation_mode():
    assert SparseConfig().threshold is None
    assert resolve_threshold(None) == 1.0e-3
    assert resolve_threshold(None, bond_order=2) == 0.0
    assert resolve_threshold(None, rcut=5.0) == 0.0
    assert resolve_threshold(0.0, bond_order=2) == 0.0


def test_config_rejects_unknown_solver():
    with pytest.raises(ValueError, match='hk_solver'):
        SparseConfig(hk_solver='lobpcg')


@pytest.mark.parametrize(
    'value, expected',
    [
        (None, None),
        (False, None),
        (True, SparseConfig()),
        ({'threshold': 1e-4}, SparseConfig(threshold=1e-4)),
        (SparseConfig(bond_order=3), SparseConfig(bond_order=3)),
    ],
)
def test_config_coerce(value, expected):
    assert SparseConfig.coerce(value) == expected


def test_config_coerce_rejects_other_types():
    with pytest.raises(TypeError, match='sparse='):
        SparseConfig.coerce(1e-3)


# ----------------------------------------------------------------------
# Class-level contract
# ----------------------------------------------------------------------


def test_every_public_method_has_a_role():
    assert all(_role(n) in ('override', 'shared', 'dense') for n in _public_methods())


def test_decorators_keep_the_dense_signature():
    for name in _public_methods():
        method = vars(PAOFLOW)[name]
        assert inspect.signature(method) == inspect.signature(inspect.unwrap(method)), name


@pytest.mark.parametrize('name', _overrides())
def test_engine_accepts_the_dense_arguments(name):
    """A routed call passes the dense arguments through unchanged, so the
    engine method must accept every one of them by name."""
    engine_params = inspect.signature(getattr(SparseEngine, name)).parameters
    if any(p.kind is p.VAR_KEYWORD for p in engine_params.values()):
        return
    dense_params = inspect.signature(vars(PAOFLOW)[name]).parameters
    missing = set(dense_params) - set(engine_params)
    assert not missing, '%s: engine lacks %s' % (name, sorted(missing))


# ----------------------------------------------------------------------
# Instance behaviour
# ----------------------------------------------------------------------


def test_dense_run_has_no_engine(dense):
    assert dense._engine is None
    with pytest.raises(RuntimeError, match='This run is dense'):
        dense.sparse
    with pytest.raises(RuntimeError, match='This run is dense'):
        dense.to_dense()


def test_sparse_run_holds_the_engine(sparse):
    assert isinstance(sparse.sparse, SparseEngine)
    assert sparse.sparse.config == SparseConfig()
    # the mesh pass always produces adaptive widths
    assert sparse.data_controller.data_attributes['smearing'] == 'gauss'


def test_dense_only_method_refuses_in_a_sparse_run(sparse):
    with pytest.raises(NotImplementedError, match='(?s)berry_curvature.*to_dense'):
        sparse.berry_curvature()


def test_routed_method_reaches_the_engine(sparse):
    with pytest.raises(RuntimeError, match='sparse dos requires the sparse Hamiltonian'):
        sparse.dos()


def test_mesh_stages_are_optional_and_accepted(sparse):
    sparse.pao_eigh()
    sparse.gradient_and_momenta()
    sparse.adaptive_smearing(afac=1.5)
    assert sparse.sparse._mesh_plan['afac'] == 1.5


def test_dense_only_options_of_routed_methods_refuse(sparse):
    with pytest.raises(NotImplementedError, match='band_curvature'):
        sparse.gradient_and_momenta(band_curvature=True)
    with pytest.raises(NotImplementedError, match='adhoc_SO'):
        sparse.bands(adhoc_SO=True)
    with pytest.raises(NotImplementedError, match='reshift_Ef'):
        sparse.interpolated_hamiltonian(reshift_Ef=True)
    with pytest.raises(ValueError, match='SparseConfig'):
        sparse.save_sparse_hamiltonian(threshold=1e-4)
