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
from PAOFLOW.sparse.config import SparseConfig, resolve_threshold
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
    return PAOFLOW(workpath=str(tmp_path), outputdir='sparse', restart=True, sparse={})


# ----------------------------------------------------------------------
# sparse= options
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
        ({}, SparseConfig()),
        ({'threshold': 1e-4}, SparseConfig(threshold=1e-4)),
        ({'bond_order': 3, 'hk_solver': 'dense'}, SparseConfig(bond_order=3, hk_solver='dense')),
    ],
)
def test_config_parse(value, expected):
    assert SparseConfig.parse(value) == expected


@pytest.mark.parametrize('value', [True, False, 1e-3, SparseConfig()])
def test_config_parse_takes_only_a_dict(value):
    with pytest.raises(TypeError, match='sparse= takes a dict'):
        SparseConfig.parse(value)


def test_config_parse_points_true_to_the_empty_dict():
    with pytest.raises(TypeError, match=r'sparse=\{\}'):
        SparseConfig.parse(True)


def test_config_parse_names_unknown_options():
    with pytest.raises(ValueError, match="'thresold' \\(did you mean 'threshold'\\?\\)"):
        SparseConfig.parse({'thresold': 1e-4})
    with pytest.raises(ValueError, match='Valid options: threshold, rcut, bond_order'):
        SparseConfig.parse({'npool': 2})


def test_constructor_rejects_unknown_options(tmp_path):
    with pytest.raises(ValueError, match='unknown option'):
        PAOFLOW(workpath=str(tmp_path), outputdir='s', restart=True, sparse={'rcutt': 5.0})


# ----------------------------------------------------------------------
# Class-level contract
# ----------------------------------------------------------------------


def test_every_public_method_has_a_role():
    assert all(_role(n) in ('override', 'shared', 'base_cell', 'dense') for n in _public_methods())


def test_every_dense_only_method_says_why():
    from PAOFLOW.sparse.dispatch import DENSE_ONLY_REASONS

    dense_only = {n for n in _public_methods() if _role(n) == 'dense'}
    assert dense_only == set(DENSE_ONLY_REASONS), (
        'dense-only methods without a reason: %s; reasons for routed methods: %s'
        % (
            sorted(dense_only - set(DENSE_ONLY_REASONS)),
            sorted(set(DENSE_ONLY_REASONS) - dense_only),
        )
    )


def test_decorators_keep_the_dense_signature():
    for name in _public_methods():
        method = vars(PAOFLOW)[name]
        assert inspect.signature(method) == inspect.signature(inspect.unwrap(method)), name


def _engine_callable(name):
    """What a routed call reaches: an engine method, or the ``__init__`` of the
    registered property class (bound past ``self, engine``)."""
    from PAOFLOW.sparse.properties import _load

    if name in vars(SparseEngine):
        return getattr(SparseEngine, name), 1
    registry = _load()
    assert name in registry, (
        '%s is @sparse_override but neither an engine method nor a property' % (name)
    )
    return registry[name].__init__, 2


@pytest.mark.parametrize('name', _overrides())
def test_engine_accepts_the_dense_arguments(name):
    """A routed call passes the dense arguments through unchanged, so the
    engine method or property class must accept every one of them by name,
    in the dense order."""
    target, skip = _engine_callable(name)
    engine_params = inspect.signature(target).parameters
    if any(p.kind is p.VAR_KEYWORD for p in engine_params.values()):
        return
    dense_params = list(inspect.signature(vars(PAOFLOW)[name]).parameters)[1:]
    assert list(engine_params)[skip:] == dense_params, name


def test_every_registered_property_is_routed():
    """A property class nobody routes to is dead code; one that is routed
    must implement exactly the dense method of its name."""
    from PAOFLOW.sparse.properties import _load

    for name, cls in _load().items():
        assert cls.method == name
        assert _role(name) == 'override', '%s is registered but not @sparse_override' % name


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
    with pytest.raises(NotImplementedError, match='(?s)find_weyl_points is dense-only.*to_dense'):
        sparse.find_weyl_points()
    with pytest.raises(NotImplementedError, match='save_sparse_hamiltonian'):
        sparse.restart_dump()


def test_base_cell_transform_refuses_after_doubling(sparse):
    class _Doubled:
        _doubled = True
        nk_grid = (1, 1, 1)

    sparse.sparse.H = _Doubled()
    with pytest.raises(RuntimeError, match='before doubling_Hamiltonian'):
        sparse.cutting_Hamiltonian(z=True)


def test_routed_method_reaches_the_engine(sparse):
    with pytest.raises(RuntimeError, match='sparse dos requires the sparse Hamiltonian'):
        sparse.dos()


def test_mesh_stages_are_optional_and_accepted(sparse):
    sparse.pao_eigh()
    sparse.gradient_and_momenta()
    sparse.adaptive_smearing(afac=1.5)
    assert sparse.sparse._mesh_plan['afac'] == 1.5


def test_band_curvature_is_recorded_for_the_mesh(sparse):
    sparse.gradient_and_momenta(band_curvature=True)
    assert sparse.sparse._mesh_plan['products'] == {'d2Ed2k'}


def test_unknown_engine_attribute_is_an_attribute_error(sparse):
    with pytest.raises(AttributeError, match='no sparse property'):
        sparse.sparse.not_a_property


def test_dense_only_options_of_routed_methods_refuse(sparse):
    with pytest.raises(NotImplementedError, match='nonlocal_velocity'):
        sparse.gradient_and_momenta(nonlocal_velocity=True)
    with pytest.raises(NotImplementedError, match='adhoc_SO'):
        sparse.bands(adhoc_SO=True)
    with pytest.raises(NotImplementedError, match='reshift_Ef'):
        sparse.interpolated_hamiltonian(reshift_Ef=True)
    with pytest.raises(ValueError, match='sparse= options'):
        sparse.save_sparse_hamiltonian(threshold=1e-4)
