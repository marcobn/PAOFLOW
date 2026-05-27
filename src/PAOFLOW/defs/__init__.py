"""Back-compatibility shim for the legacy ``PAOFLOW.defs.*`` layout.

The contents of this package were redistributed across the pipeline-stage
subpackages in ``src/PAOFLOW/`` (see
``TODOs/defs_reorganization_plan.md`` §8.1).  Legacy imports of the form

    from PAOFLOW.defs.X import Y
    import PAOFLOW.defs.X

continue to work via a meta-path finder that transparently aliases the
old dotted name to its new location and emits a ``FutureWarning``
(shown unconditionally, unlike ``DeprecationWarning`` which is silenced
outside ``__main__`` per PEP 565).

External code (notebooks, AFLOWpi, user scripts) should migrate to the
new paths:

    PAOFLOW.inputs       PAOFLOW.spectrum     PAOFLOW.boltzmann
    PAOFLOW.projection   PAOFLOW.topology     PAOFLOW.models
    PAOFLOW.hamiltonian  PAOFLOW.response     PAOFLOW.writers
                                              PAOFLOW.utils
"""
from __future__ import annotations

import sys as _sys
import warnings as _warnings
from importlib import import_module as _import_module
from importlib.abc import Loader as _Loader, MetaPathFinder as _MetaPathFinder
from importlib.util import spec_from_loader as _spec_from_loader

# Mapping: legacy module name → new subpackage under PAOFLOW.
_MODULE_MAP = {
    # inputs
    'read_QE_xml': 'inputs', 'read_VASP': 'inputs',
    'read_inputfile_xml_parse': 'inputs', 'read_upf': 'inputs',
    'read_sh_nl': 'inputs', 'read_pao_output': 'inputs',
    'file_io': 'inputs', 'basis_presets': 'inputs',
    # projection
    'do_atwfc_proj': 'projection', 'projection_operator': 'projection',
    'do_projectability': 'projection', 'do_minimal': 'projection',
    'do_ortho': 'projection', 'do_non_ortho': 'projection',
    'upf_gaussfit': 'projection',
    # hamiltonian
    'do_build_pao_hamiltonian': 'hamiltonian', 'do_doubling': 'hamiltonian',
    'do_real_space': 'hamiltonian', 'pao_sym': 'hamiltonian',
    'add_ext_field': 'hamiltonian', 'do_spin_orbit': 'hamiltonian',
    'do_gradient': 'hamiltonian', 'do_momentum': 'hamiltonian',
    'do_double_grid': 'hamiltonian', 'do_d2Hd2k': 'hamiltonian',
    # spectrum
    'do_eigh': 'spectrum', 'do_bands': 'spectrum',
    'sparse_bands': 'spectrum', 'kpnts_interpolation_mesh': 'spectrum',
    'do_site_projected_bands': 'spectrum', 'do_band_curvature': 'spectrum',
    'do_effective_mass': 'spectrum', 'do_dos': 'spectrum',
    'do_pdos': 'spectrum', 'do_Efermi': 'spectrum',
    'do_adaptive_smearing': 'spectrum',
    # topology
    'do_topology': 'topology', 'do_berry_phase': 'topology',
    'do_find_Weyl': 'topology', 'do_fermisurf': 'topology',
    'do_spin_texture': 'topology',
    'do_wave_function_site_projection': 'topology',
    'pfaffian': 'topology', 'clebsch_gordan': 'topology',
    # response
    'do_epsilon': 'response', 'do_Hall': 'response',
    'do_rashba_edelstein': 'response', 'do_ipr': 'response',
    # boltzmann
    'TauModel': 'boltzmann', 'do_tau_models': 'boltzmann',
    'do_Boltz_tensors': 'boltzmann', 'do_transport': 'boltzmann',
    'do_doping': 'boltzmann',
    # models
    'models': 'models', 'sk_fitting': 'models',
    'edtb_params': 'models', 'dual_params': 'models',
    'band_unfold': 'models', 'surface_project': 'models',
    # writers
    'write2xsf': 'writers', 'write2bxsf': 'writers',
    'write2bxsf4skeaf': 'writers', 'write4bt2': 'writers',
    # utils
    'communication': 'utils', 'constants': 'utils',
    'smearing': 'utils', 'pyints': 'utils',
    'perturb_split': 'utils', 'get_K_grid_fft': 'utils',
    'get_R_grid_fft': 'utils', 'zero_pad': 'utils',
    'header': 'utils', 'report_exception': 'utils',
    'module_prerequisites': 'utils',
}

_PREFIX = __name__ + '.'  # 'PAOFLOW.defs.'


class _DefsAliasLoader(_Loader):
    def __init__(self, target_name: str) -> None:
        self._target_name = target_name

    def create_module(self, spec):
        mod = _import_module(self._target_name)
        _sys.modules[spec.name] = mod
        return mod

    def exec_module(self, module):  # pragma: no cover - aliased module already loaded
        return None


class _DefsAliasFinder(_MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if not fullname.startswith(_PREFIX):
            return None
        leaf = fullname[len(_PREFIX):]
        if '.' in leaf or leaf not in _MODULE_MAP:
            return None
        new_stage = _MODULE_MAP[leaf]
        new_name = f'PAOFLOW.{new_stage}.{leaf}'
        _warnings.warn(
            f'{fullname} is deprecated; import from {new_name} instead',
            FutureWarning, stacklevel=2,
        )
        return _spec_from_loader(fullname, _DefsAliasLoader(new_name))


# Install exactly once.
if not any(isinstance(f, _DefsAliasFinder) for f in _sys.meta_path):
    _sys.meta_path.append(_DefsAliasFinder())


def __getattr__(name):
    """Support ``from PAOFLOW import defs; defs.TauModel`` style access."""
    if name in _MODULE_MAP:
        _warnings.warn(
            f'PAOFLOW.defs.{name} is deprecated; import from '
            f'PAOFLOW.{_MODULE_MAP[name]}.{name} instead',
            FutureWarning, stacklevel=2,
        )
        return _import_module(f'PAOFLOW.defs.{name}')
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(set(globals()) | set(_MODULE_MAP))
