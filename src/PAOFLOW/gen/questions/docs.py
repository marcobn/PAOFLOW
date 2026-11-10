"""Field help taken from the PAOFLOW documentation itself.

A question's ``doc`` (or the :data:`FIELD_DOCS` entry of its workflow) points at the parameter
of a method whose docstring documents it (``'PAOFLOW.PAOFLOW.PAOFLOW.dos:emin'``) or at an entry
of a comment dictionary (``'PAOFLOW.gen.elphon_system.OPTICS_COMMENTS:degauss_ev'``, the comments
written into ``system.py``).  The GUI shows that text under the field, so the help stays as
current as the API.  NumPy docstrings and the older ``Arguments:`` style are both read.
"""

from __future__ import annotations

import importlib
import inspect
import re
from functools import lru_cache
from typing import Any

PAO = 'PAOFLOW.PAOFLOW.PAOFLOW.'
ELPHON = 'PAOFLOW.gen.elphon_system.'

#: Documentation of the questions, per workflow (questions may also carry ``doc`` themselves).
FIELD_DOCS: dict[str, dict[str, str]] = {
    'run': {
        'ibrav': PAO + 'bands:ibrav',
        'npool': PAO + '__init__:npool',
        'smearing': PAO + '__init__:smearing',
        'std_basis': PAO + 'projections:configuration',
        'emin': PAO + 'dos:emin',
        'emax': PAO + 'dos:emax',
        'ne': PAO + 'dos:ne',
        'do_pdos': PAO + 'dos:do_pdos',
        'nfft': PAO + 'interpolated_hamiltonian:nfft1',
        'nlo_molecule': PAO + 'nonlinear_optics:molecule',
        'nlo_nk': PAO + 'nonlinear_optics:nk',
        'nlo_emin': PAO + 'nonlinear_optics:emin',
        'nlo_emax': PAO + 'nonlinear_optics:emax',
        'nlo_ne': PAO + 'nonlinear_optics:ne',
        'nlo_eta': PAO + 'nonlinear_optics:eta',
        'nlo_scissor': PAO + 'nonlinear_optics:scissor',
        'nlo_components': PAO + 'nonlinear_optics:components',
        'nlo_dipoles': PAO + 'nonlinear_optics:position_dipoles',
        'coupled_nk': PAO + 'coupled_optical_response:nk',
        'coupled_kernels': PAO + 'coupled_optical_response:kernels',
        'coupled_emin': PAO + 'coupled_optical_response:dielectric',
        'coupled_emax': PAO + 'coupled_optical_response:dielectric',
        'coupled_ne': PAO + 'coupled_optical_response:dielectric',
        'coupled_shg_energies': PAO + 'coupled_optical_response:second_harmonic',
        'coupled_eta': PAO + 'coupled_optical_response:eta',
        'coupled_components': PAO + 'coupled_optical_response:components',
    },
    'acbn0': {'ibrav': PAO + 'bands:ibrav', 'nk': PAO + 'bands:nk'},
    'phonon': {
        'ibrav': PAO + 'bands:ibrav',
        'supercell': PAO + 'phonons:supercell_matrix',
        'displacement': PAO + 'phonon_setup:displacement_distance',
        'hubbard_file': PAO + 'phonons:hubbard_file',
        'born_method': PAO + 'born_charges:method',
        'vibdielectric_gamma': PAO + 'vibrational_dielectric:gamma',
        'raman_delta': PAO + 'raman_spectrum:delta',
        'raman_nbnd': PAO + 'raman_spectrum:nbnd',
    },
    'pyskeaf': {
        'outputdir': PAO + '__init__:outputdir',
        'bands': PAO + 'pyskeaf:bands',
        'fermi_energy': PAO + 'pyskeaf:fermi_energy',
        'num_interpolation': PAO + 'pyskeaf:num_interpolation',
        'b_field': PAO + 'pyskeaf:b_field',
        'theta': PAO + 'pyskeaf:azimuthal',
        'phi': PAO + 'pyskeaf:polar',
        'theta_start': PAO + 'pyskeaf:azimuthal',
        'theta_end': PAO + 'pyskeaf:azimuthal',
        'phi_start': PAO + 'pyskeaf:polar',
        'phi_end': PAO + 'pyskeaf:polar',
        'num_angles': PAO + 'pyskeaf:num_angles',
        'minimum_frequency': PAO + 'pyskeaf:minimum_frequency',
        'frequency_tolerance': PAO + 'pyskeaf:frequency_tolerance',
        'orbit_tolerance': PAO + 'pyskeaf:orbit_tolerance',
        'allow_wall_orbits': PAO + 'pyskeaf:allow_wall_orbits',
        'verbose': PAO + 'pyskeaf:verbose',
    },
    'sscha': {
        'temperatures': PAO + 'sscha_setup:temperature',
        'supercell': PAO + 'sscha_setup:supercell_matrix',
        'harmonic_dyn': PAO + 'sscha_setup:harmonic_dynamical_matrices',
        'kgrid': PAO + 'sscha_setup:kgrid',
        'population_sizes': PAO + 'sscha_setup:population_sizes',
        'hessian_size': PAO + 'sscha_setup:hessian_size',
        'min_step_dyn': PAO + 'sscha_setup:min_step_dyn',
        'kong_liu_ratio': PAO + 'sscha_setup:kong_liu_ratio',
        'meaningful_factor': PAO + 'sscha_setup:meaningful_factor',
        'pp_dir': PAO + 'sscha_setup:pp_dir',
    },
    'elphon': {
        'coupling_source': ELPHON + 'SYSTEM_COMMENTS:coupling_source',
        'kgrid_axis': ELPHON + 'SYSTEM_COMMENTS:coarse_grid',
        'qgrid_axis': ELPHON + 'SYSTEM_COMMENTS:qgrid',
        'masses_amu': ELPHON + 'SYSTEM_COMMENTS:masses_amu',
        'nelec': ELPHON + 'SYSTEM_COMMENTS:nelec',
        'basisdir': ELPHON + 'SYSTEM_COMMENTS:basis_dir',
        'pthr': ELPHON + 'SYSTEM_COMMENTS:pthr',
        'sc_nq': ELPHON + 'SUPERCONDUCTIVITY_COMMENTS:nq',
        'sc_nk': ELPHON + 'SUPERCONDUCTIVITY_COMMENTS:nk',
        'sc_adaptive': ELPHON + 'SUPERCONDUCTIVITY_COMMENTS:adaptive_smearing',
        'sc_sigma': ELPHON + 'SUPERCONDUCTIVITY_COMMENTS:sigma_ev',
        'sc_mu_star': ELPHON + 'SUPERCONDUCTIVITY_COMMENTS:mu_star',
        'sc_transport': ELPHON + 'SUPERCONDUCTIVITY_COMMENTS:transport',
        'sc_fs_coupling': ELPHON + 'SUPERCONDUCTIVITY_COMMENTS:fs_coupling',
        'sc_fsthick': ELPHON + 'SUPERCONDUCTIVITY_COMMENTS:fsthick_ev',
        'sc_me_temps': ELPHON + 'SUPERCONDUCTIVITY_COMMENTS:me_temps',
        'optics_nq': ELPHON + 'OPTICS_COMMENTS:nq',
        'optics_nk': ELPHON + 'OPTICS_COMMENTS:nk',
        'optics_electrons_save': ELPHON + 'OPTICS_COMMENTS:electrons_save',
        'optics_gap': ELPHON + 'OPTICS_COMMENTS:target_gap_ev',
        'optics_gap_reference': ELPHON + 'OPTICS_COMMENTS:gap_reference_save',
        'optics_varshni': ELPHON + 'OPTICS_COMMENTS:varshni',
        'optics_screening_eps': ELPHON + 'OPTICS_COMMENTS:screening_eps',
        'optics_emissivity': ELPHON + 'OPTICS_COMMENTS:emissivity',
        'optics_thickness': ELPHON + 'OPTICS_COMMENTS:thickness_um',
        'optics_angles': ELPHON + 'OPTICS_COMMENTS:angles',
        'optics_direct': ELPHON + 'OPTICS_COMMENTS:direct_dielectric_dir',
        'optics_omega_min': ELPHON + 'OPTICS_COMMENTS:omega_ev',
        'optics_temps': ELPHON + 'OPTICS_COMMENTS:temps_k',
        'optics_degauss': ELPHON + 'OPTICS_COMMENTS:degauss_ev',
        'optics_fsthick': ELPHON + 'OPTICS_COMMENTS:fsthick_ev',
        'optics_refractive_index': ELPHON + 'OPTICS_COMMENTS:refractive_index',
        'optics_position_dipoles': ELPHON + 'OPTICS_COMMENTS:position_dipoles',
        'resistivity_temps': ELPHON + 'TRANSPORT_COMMENTS:temps_k',
        'resistivity_rho0': ELPHON + 'TRANSPORT_COMMENTS:rho0_uohm_cm',
        'resistivity_b_ee': ELPHON + 'TRANSPORT_COMMENTS:b_ee_per_ev',
        'resistivity_interband': ELPHON + 'TRANSPORT_COMMENTS:interband_dielectric_dir',
        'mobility_nq': ELPHON + 'TRANSPORT_COMMENTS:nq',
        'mobility_nk': ELPHON + 'TRANSPORT_COMMENTS:nk',
        'mobility_temps': ELPHON + 'TRANSPORT_COMMENTS:temps_k',
        'mobility_doping': ELPHON + 'TRANSPORT_COMMENTS:doping_cm3',
        'mobility_window': ELPHON + 'TRANSPORT_COMMENTS:window_ev',
        'mobility_smearing': ELPHON + 'TRANSPORT_COMMENTS:smearing_ev',
        'mobility_electrons_save': ELPHON + 'TRANSPORT_COMMENTS:electrons_save',
        'mobility_gap': ELPHON + 'TRANSPORT_COMMENTS:target_gap_ev',
        'mobility_varshni': ELPHON + 'TRANSPORT_COMMENTS:varshni',
        'mobility_screening_eps': ELPHON + 'TRANSPORT_COMMENTS:screening_eps',
        'arpes_photon': ELPHON + 'ARPES_COMMENTS:photon_ev',
        'arpes_work_function': ELPHON + 'ARPES_COMMENTS:work_function_ev',
        'arpes_direction': ELPHON + 'ARPES_COMMENTS:direction',
        'arpes_label': ELPHON + 'ARPES_COMMENTS:direction_label',
        'thermal_temps': ELPHON + 'THERMAL_COMMENTS:temps_k',
        'thermal_supercell': ELPHON + 'THERMAL_COMMENTS:supercell',
        'thermal_kgrid': ELPHON + 'THERMAL_COMMENTS:kgrid',
        'thermal_bands_per_atom': ELPHON + 'THERMAL_COMMENTS:nbnd_per_atom',
        'thermal_configurations': ELPHON + 'THERMAL_COMMENTS:seeds',
    },
}


GREEK = {
    'alpha': 'α',
    'beta': 'β',
    'gamma': 'γ',
    'Gamma': 'Γ',
    'delta': 'δ',
    'Delta': 'Δ',
    'epsilon': 'ε',
    'eta': 'η',
    'lambda': 'λ',
    'mu': 'μ',
    'nu': 'ν',
    'omega': 'ω',
    'Omega': 'Ω',
    'pi': 'π',
    'sigma': 'σ',
    'tau': 'τ',
    'chi': 'χ',
    'phi': 'φ',
    'theta': 'θ',
    'hbar': 'ħ',
    'parallel': '∥',
    'infty': '∞',
    'cdot': '·',
    'times': '×',
    'nabla': '∇',
    'partial': '∂',
    'rm': '',
    'mathrm': '',
}


def plain_text(text: str) -> str:
    """Docstring text without reST markup (roles, literals, simple LaTeX to Unicode).

    Parameters
    ----------
    text : str
        Text with ``:math:`\\delta```, ``` ``code`` ```, ``:mod:`a.b```, ...

    Returns
    -------
    str
    """

    def role(match: re.Match) -> str:
        kind, content = match.group(1), match.group(2)
        if kind == 'math':
            content = re.sub(r'\\([A-Za-z]+)', lambda m: GREEK.get(m.group(1), m.group(1)), content)
            return content.replace('{', '').replace('}', '').replace('\\', '')
        content = content.lstrip('~')
        return (
            content.rsplit('.', 1)[-1]
            if kind in ('mod', 'func', 'meth', 'class', 'data', 'attr')
            else content
        )

    text = re.sub(r':(\w+):`([^`]+)`', role, text)
    text = re.sub(r'``([^`]+)``', r'\1', text)
    return re.sub(r'\s+', ' ', text).strip()


def _resolve(path: str) -> Any:
    parts = path.split('.')
    for cut in range(len(parts), 0, -1):
        try:
            target: Any = importlib.import_module('.'.join(parts[:cut]))
        except ImportError:
            continue
        for name in parts[cut:]:
            target = getattr(target, name)
        return target
    raise ImportError(path)


def _clean(lines: list[str]) -> str:
    text = ' '.join(line.strip() for line in lines if line.strip())
    return re.sub(r'\s+', ' ', text).strip()


def docstring_parameters(obj: Any) -> dict[str, str]:
    """Parameter descriptions of a docstring (NumPy ``Parameters`` or ``Arguments:`` style).

    Parameters
    ----------
    obj : Any
        A function, method or class.

    Returns
    -------
    dict
        Description by parameter name (several names on one line, ``a, b : float``, each get it).
        Every NumPy parameter section is read (``Parameters``, ``Other Parameters``, ``Keyword
        Arguments``).
    """
    lines = (inspect.getdoc(obj) or '').splitlines()
    found: dict[str, str] = {}
    titles = ('Parameters', 'Other Parameters', 'Keyword Arguments')
    for index, line in enumerate(lines):
        if (
            line.strip() not in titles
            or index + 1 >= len(lines)
            or set(lines[index + 1].strip()) != {'-'}
        ):
            continue
        names: list[str] = []
        body: list[str] = []
        for entry in lines[index + 2 :] + ['']:
            stripped = entry.strip()
            if entry and not entry.startswith(' '):
                for name in names:
                    found[name] = _clean(body)
                names, body = [], []
                if ':' not in entry:
                    if re.match(r'^[A-Z][A-Za-z ]+$', stripped):
                        break  # the next section title
                    continue
                header = entry.split(':', 1)[0]
                names = [n.strip().lstrip('*') for n in header.split(',') if n.strip()]
            else:
                body.append(entry)
        for name in names:
            found[name] = _clean(body)
    if found:
        return {k: v for k, v in found.items() if v}
    entry_pattern = re.compile(
        r'^\s+([A-Za-z_*][\w*]*(?:\s*,\s*[A-Za-z_*][\w*]*)*)\s*(\([^)]*\))?\s*:\s*(.*)$'
    )
    for index, line in enumerate(lines):
        if not line.strip().lower().startswith(('arguments:', 'args:')):
            continue
        current: list[str] = []
        for entry in lines[index + 1 :]:
            match = entry_pattern.match(entry)
            if match and len(entry) - len(entry.lstrip()) <= 4:
                current = [n.strip().lstrip('*') for n in match.group(1).split(',')]
                for name in current:
                    found[name] = match.group(3).strip()
            elif entry.strip() and current and entry.startswith('    '):
                for name in current:
                    found[name] = (found[name] + ' ' + entry.strip()).strip()
            elif not entry.strip():
                continue
            else:
                break
        break
    return found


@lru_cache(maxsize=512)
def field_doc(reference: str) -> tuple[str, str] | None:
    """Documentation text of a reference and a short name of its source.

    Parameters
    ----------
    reference : str
        ``'module.object:parameter'`` or ``'module.DICT:key'``.

    Returns
    -------
    tuple of (str, str) or None
        ``(text, source)``, e.g. ``('broadening of the ...', 'nonlinear_optics')``.
    """
    if ':' not in reference:
        return None
    path, key = reference.rsplit(':', 1)
    try:
        target = _resolve(path)
    except (ImportError, AttributeError):
        return None
    name = path.rsplit('.', 1)[-1]
    if isinstance(target, dict):
        text = target.get(key)
        return (plain_text(str(text)), 'system.py comment') if text else None
    text = docstring_parameters(target).get(key)
    if not text:
        return None
    source = 'PAOFLOW' if name == '__init__' else name
    return plain_text(text), source


def question_doc(workflow: str, key: str, own: str = '') -> tuple[str, str] | None:
    """Documentation of a question (its own ``doc`` first, then :data:`FIELD_DOCS`).

    Parameters
    ----------
    workflow : str
        Workflow name.
    key : str
        Question key.
    own : str, optional
        The question's ``doc``.

    Returns
    -------
    tuple of (str, str) or None
    """
    reference = own or FIELD_DOCS.get(workflow, {}).get(key, '')
    return field_doc(reference) if reference else None
