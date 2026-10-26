"""``PAOFLOW(..., sparse=True)`` routing: one driver, two engines.

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
from PAOFLOW.sparse.config import InteriorWindow, SparseConfig, resolve_threshold
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
# sparse= flag and sparse_config= options
# ----------------------------------------------------------------------


def test_config_rejects_two_cutoffs():
    with pytest.raises(ValueError, match='not both'):
        SparseConfig(rcut=5.0, bond_order=2)


@pytest.mark.parametrize('cutoff', [{'rcut': 5.0}, {'bond_order': 2}])
def test_config_rejects_threshold_with_a_cutoff(cutoff):
    with pytest.raises(ValueError, match='hopping_threshold cannot be combined'):
        SparseConfig(hopping_threshold=1e-3, **cutoff)


def test_config_threshold_defaults_by_truncation_mode():
    assert SparseConfig().hopping_threshold is None
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
        (None, SparseConfig()),
        ({}, SparseConfig()),
        ({'hopping_threshold': 1e-4}, SparseConfig(hopping_threshold=1e-4)),
        ({'bond_order': 3, 'hk_solver': 'dense'}, SparseConfig(bond_order=3, hk_solver='dense')),
    ],
)
def test_config_parse(value, expected):
    assert SparseConfig.parse(value) == expected


@pytest.mark.parametrize('value', [True, False, 1e-3, SparseConfig()])
def test_config_parse_takes_only_a_dict(value):
    with pytest.raises(TypeError, match='sparse_config= takes a dict'):
        SparseConfig.parse(value)


def test_config_parses_nev():
    assert SparseConfig.parse({'nev': 12}).nev == 12
    assert SparseConfig().nev is None


@pytest.mark.parametrize('value', [0, -3, 2.5, True])
def test_config_rejects_a_bad_nev(value):
    with pytest.raises(ValueError, match='nev must be an integer'):
        SparseConfig.parse({'nev': value})


def test_config_has_no_energy_window_option():
    with pytest.raises(ValueError, match="unknown option 'energy_window'"):
        SparseConfig.parse({'energy_window': {'emax': 2.2}})


def test_config_parses_the_interior_window():
    cfg = SparseConfig.parse({'interior_window': {'elo': -3, 'ehi': 3}})
    assert cfg.interior_window == InteriorWindow(elo=-3.0, ehi=3.0)
    assert not cfg.interior_window.auto
    auto = SparseConfig.parse({'interior_window': {}}).interior_window
    assert auto == InteriorWindow() and auto.auto


@pytest.mark.parametrize(
    'window, error, match',
    [
        ({'elo': -3.0}, ValueError, "both 'elo' and 'ehi'"),
        ({'elo': 1.0, 'ehi': -1.0}, ValueError, 'ehi > elo'),
        (
            {'elo': -1.0, 'ehi': 1.0, 'kT_margin_eV': 0.1},
            ValueError,
            "unknown option 'kT_margin_eV'",
        ),
        ({'smear_margin_eV': 0.5}, ValueError, "unknown option 'smear_margin_eV'"),
        ({'elo': -1.0, 'eh': 1.0}, ValueError, "did you mean 'ehi'"),
        ([-1.0, 1.0], TypeError, 'takes a dict'),
        (True, TypeError, 'takes a dict'),
    ],
)
def test_config_rejects_a_bad_interior_window(window, error, match):
    with pytest.raises(error, match=match):
        SparseConfig.parse({'interior_window': window})


def test_config_solver_limits_default_to_the_solver_constants():
    from PAOFLOW.sparse.solver import DENSE_N_MAX, DENSE_RATIO

    cfg = SparseConfig()
    assert cfg.limits == {'dense_ratio': DENSE_RATIO, 'dense_n_max': DENSE_N_MAX}
    cfg = SparseConfig.parse({'dense_n_max': 10**9, 'dense_ratio': 0.3})
    assert cfg.limits == {'dense_ratio': 0.3, 'dense_n_max': 10**9}


@pytest.mark.parametrize(
    'value, match',
    [
        ({'dense_n_max': 0}, 'dense_n_max'),
        ({'dense_ratio': 0.0}, 'dense_ratio'),
        ({'dense_ratio': 1.5}, 'dense_ratio'),
    ],
)
def test_config_rejects_bad_solver_limits(value, match):
    with pytest.raises(ValueError, match=match):
        SparseConfig.parse(value)


def test_config_rejects_nev_with_an_interior_window():
    with pytest.raises(ValueError, match="either 'nev' or 'interior_window'"):
        SparseConfig.parse({'nev': 8, 'interior_window': {'elo': -1.0, 'ehi': 1.0}})


@pytest.mark.parametrize('value', [{}, {'hopping_threshold': 1e-4}, None, 'yes', 1])
def test_constructor_takes_only_a_bool(tmp_path, value):
    with pytest.raises(TypeError, match='sparse= takes True or False'):
        PAOFLOW(workpath=str(tmp_path), outputdir='s', restart=True, sparse=value)


def test_constructor_points_a_dict_to_sparse_config(tmp_path):
    with pytest.raises(TypeError, match=r"sparse_config=\{'hopping_threshold': 0.0001\}"):
        PAOFLOW(
            workpath=str(tmp_path), outputdir='s', restart=True, sparse={'hopping_threshold': 1e-4}
        )


def test_constructor_refuses_config_without_sparse(tmp_path):
    with pytest.raises(ValueError, match='sparse=False'):
        PAOFLOW(workpath=str(tmp_path), outputdir='s', sparse_config={'hopping_threshold': 1e-4})


def test_constructor_passes_the_config_to_the_engine(tmp_path):
    p = PAOFLOW(
        workpath=str(tmp_path),
        outputdir='s',
        restart=True,
        sparse=True,
        sparse_config={'bond_order': 3, 'nev': 10},
    )
    assert p.sparse.config == SparseConfig(bond_order=3, nev=10)


def test_config_parse_names_unknown_options():
    with pytest.raises(ValueError, match="'thresold' \\(did you mean 'hopping_threshold'\\?\\)"):
        SparseConfig.parse({'thresold': 1e-4})
    with pytest.raises(ValueError, match='Valid options: hopping_threshold, rcut, bond_order'):
        SparseConfig.parse({'npool': 2})


def test_constructor_rejects_unknown_options(tmp_path):
    with pytest.raises(ValueError, match='unknown option'):
        PAOFLOW(
            workpath=str(tmp_path),
            outputdir='s',
            restart=True,
            sparse=True,
            sparse_config={'rcutt': 5.0},
        )


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


def test_dense_save_names_hopping_threshold_in_its_refusal(dense):
    with pytest.raises(ValueError, match='hopping_threshold cannot be combined'):
        dense.save_sparse_hamiltonian(hopping_threshold=1e-3, bond_order=2)


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
    with pytest.raises(ValueError, match='sparse_config options'):
        sparse.save_sparse_hamiltonian(hopping_threshold=1e-4)


class _ProjectedH:
    """Stands in for the bond list: only what the doubling pre-flight reads."""

    nawf, nnz = 16, 1000

    def project_doubling(self, nx, ny, nz):
        gb = 1024.0**3
        d = nx + ny + nz
        return {
            'd': d,
            'N': 2**d,
            'nawf': self.nawf * 2**d,
            'nnz': self.nnz * 2**d,
            'peak_bytes': 4.0 * gb,
            'steady_bytes': 2.0 * gb,
            'dense_hk_bytes': 0.1 * gb,
        }


@pytest.mark.parametrize('avail_gb, warns', [(1.0, True), (100.0, False)])
def test_doubling_preflight_warns_and_continues(sparse, monkeypatch, capsys, avail_gb, warns):
    from PAOFLOW.sparse import engine

    monkeypatch.setattr(engine, '_available_memory_bytes', lambda: avail_gb * 1024.0**3)
    monkeypatch.setattr(engine, '_node_local_ranks', lambda comm: 1)
    sparse.sparse.H = _ProjectedH()
    sparse.data_controller.data_attributes['bnd'] = 8
    proj = sparse.sparse._preflight_doubling(1, 1, 0)
    assert proj['N'] == 4
    out = capsys.readouterr().out
    assert ('may run out of memory' in out) is warns
    if warns:
        assert 'reduce the doubling count' in out


# ----------------------------------------------------------------------
# The solve window is resolved before every pass; only a pass sets bnd
# ----------------------------------------------------------------------


class _WindowH:
    """Stands in for the bond list: only what resolving a window without a
    probe reads."""

    nawf = 20
    _doubled = False


@pytest.fixture
def windowed(tmp_path):
    p = PAOFLOW(
        workpath=str(tmp_path),
        outputdir='w',
        restart=True,
        sparse=True,
        sparse_config={'nev': 10},
    )
    p.sparse.H = _WindowH()
    p.data_controller.data_attributes['bnd'] = 8
    return p


def test_a_given_nev_sizes_the_window_but_not_bnd(windowed):
    engine = windowed.sparse
    window = engine._resolve_window((), 'test')
    assert (window.mode, window.nev, window.nev_source) == ('bottom', 10, 'given')
    assert window.at_top  # more than the 8 projectable bands
    # only a pass writes attr['bnd']
    assert windowed.data_controller.data_attributes['bnd'] == 8


def test_the_default_window_solves_the_projectable_bands(sparse):
    sparse.sparse.H = _WindowH()
    sparse.data_controller.data_attributes['bnd'] = 8
    window = sparse.sparse._resolve_window((), 'test')
    assert (window.nev, window.nev_source, window.at_top) == (8, 'projectable', True)
    assert sparse.data_controller.data_attributes['bnd'] == 8


def test_doubling_refuses_after_a_solve(windowed):
    windowed.sparse._mesh_passes = 1
    with pytest.raises(RuntimeError, match='before bands'):
        windowed.doubling_Hamiltonian(1, 0, 0)


def test_an_explicit_interior_window_is_used_as_given(tmp_path):
    p = PAOFLOW(
        workpath=str(tmp_path),
        outputdir='i',
        restart=True,
        sparse=True,
        sparse_config={'interior_window': {'elo': -3.0, 'ehi': 3.0}},
    )
    engine = p.sparse
    engine.H = _WindowH()
    p.data_controller.data_attributes['bnd'] = 8
    assert engine._interior is None
    window = engine._resolve_window((), 'test')
    assert engine._interior == (-3.0, 3.0) == window.interior
    assert window.nev is None
    assert p.data_controller.data_attributes['bnd'] == 8  # nev comes from the window count


def test_an_automatic_interior_window_needs_a_property_range(tmp_path):
    p = PAOFLOW(
        workpath=str(tmp_path),
        outputdir='a',
        restart=True,
        sparse=True,
        sparse_config={'interior_window': {}},
    )
    p.sparse.H = _WindowH()
    p.data_controller.data_attributes['bnd'] = 8
    with pytest.raises(ValueError, match='no property of this pass has one'):
        p.sparse._resolve_window((), 'test')


def test_the_engine_has_no_window_methods(sparse):
    for name in ('energy_window', 'interior_window'):
        with pytest.raises(AttributeError):
            getattr(sparse.sparse, name)


def test_to_dense_warns_that_a_window_option_is_dropped(windowed, monkeypatch, capsys):
    import numpy as np

    from PAOFLOW.sparse import bridge

    class _BaseCellH(_WindowH):
        nk_grid = (2, 2, 2)
        nnz = 1

    def _densify(dc, H, Dnm=None):
        dc.data_arrays['HRs'] = np.zeros((1, 1, 2, 2, 2, 1))

    monkeypatch.setattr(bridge, 'densify', _densify)
    attr = windowed.data_controller.data_attributes
    attr['nk1'] = attr['nk2'] = attr['nk3'] = 2
    windowed.sparse.H = _BaseCellH()
    windowed.to_dense()
    assert "'nev' option of sparse_config does not apply" in capsys.readouterr().out
    assert windowed._engine is None
