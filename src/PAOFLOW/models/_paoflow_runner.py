"""
_paoflow_runner.py — Standalone, MPI-aware PAOFLOW invocation helpers.

Owns all direct interaction with the ``PAOFLOW.PAOFLOW`` object so that the
model classes (``EDTBModel``, band-unfolding, surface projection) never embed
PAOFLOW calls in their methods.  Because these are plain module-level
functions taking a model dict, an external dispatcher can call them directly.

PAOFLOW currently binds itself to ``MPI.COMM_WORLD`` internally, so a single
call already runs across every rank of the launching communicator.  These
helpers add the missing rank-awareness (collective barriers so that files
written on rank 0 are visible everywhere before the call returns).
"""

from __future__ import annotations

import numpy as np


def run_model_bands(
    model_dict: dict,
    *,
    ibrav: int = 0,
    nk: int = 500,
    outputdir: str = 'output',
    band_path: str | None = None,
    high_sym_points: dict | None = None,
    smearing: str = 'gauss',
    verbose: bool = False,
) -> dict:
    """Instantiate PAOFLOW from a model dict and compute the band structure.

    Parameters
    ----------
    model_dict : dict
        PAOFLOW-compatible model dict (e.g. ``EDTBModel.to_model_dict()``).
    ibrav : int
        Bravais lattice type.
    nk : int
        Number of k-points along the path.
    outputdir : str
        Directory for output files.
    band_path : str, optional
        Custom band path (e.g. ``"L-G-X"``).
    high_sym_points : dict, optional
        Custom high-symmetry point coordinates.
    smearing : str
        Smearing type.
    verbose : bool
        Print PAOFLOW output.

    Returns
    -------
    dict
        ``bands_file`` : str — path to ``bands_0.dat``
        ``sym_file`` : str — path to ``kpath_points.txt``
        ``paoflow`` : PAOFLOW object (for further analysis)
    """
    from PAOFLOW import PAOFLOW as PF

    pao = PF.PAOFLOW(
        savedir=None,
        model=model_dict,
        outputdir=outputdir,
        smearing=smearing,
        verbose=verbose,
    )
    _, attr = pao.data_controller.data_dicts()

    bands_kw = {'ibrav': ibrav, 'nk': nk}
    if band_path is not None:
        bands_kw['band_path'] = band_path
    if high_sym_points is not None:
        bands_kw['high_sym_points'] = high_sym_points

    pao.bands(**bands_kw)

    # Ensure rank-0 output files are on disk before any rank reads them.
    pao.comm.Barrier()

    return {
        'bands_file': f'{attr["outputdir"]}/bands_0.dat',
        'sym_file': f'{attr["outputdir"]}/kpath_points.txt',
        'paoflow': pao,
    }


def build_model_hamiltonian(
    model_dict: dict,
    *,
    outputdir: str = '_paoflow_tmp',
    smearing: str = 'gauss',
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, int, int, np.ndarray | None]:
    """Instantiate PAOFLOW from a model dict and extract HRs + R-grid.

    Parameters
    ----------
    model_dict : dict
        PAOFLOW-compatible model dict.
    outputdir : str
        Temporary output directory name.
    smearing : str
        Smearing type.
    verbose : bool
        Print PAOFLOW output.

    Returns
    -------
    HRs : (nawf, nawf, nR, nspin) real-space Hamiltonian.
    R : (nR, 3) lattice vectors in Cartesian/alat units.
    nawf : number of Wannier functions.
    nspin : number of spin channels.
    norbitals : (natoms,) orbital count per atom, or None.
    """
    from PAOFLOW import PAOFLOW as PF
    from PAOFLOW.utils.get_R_grid_fft import get_R_grid_fft

    pf = PF.PAOFLOW(
        savedir=None,
        model=model_dict,
        outputdir=outputdir,
        smearing=smearing,
        verbose=verbose,
    )
    arry, _ = pf.data_controller.data_dicts()
    nawf, _, nk1, nk2, nk3, nspin = arry['HRs'].shape

    get_R_grid_fft(pf.data_controller, nk1, nk2, nk3)
    R = arry['R'].copy()
    HRs = arry['HRs'].reshape(nawf, nawf, -1, nspin).copy()

    norbitals = arry.get('norbitals', None)
    return HRs, R, nawf, nspin, norbitals
