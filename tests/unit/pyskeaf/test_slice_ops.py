import numpy as np

from PAOFLOW.pyskeaf.io_bxsf import BXSFData
from PAOFLOW.pyskeaf.slice_ops import _bxsf_indices_from_slice, make_slice_geometry


def test_plr_inverse_matches_skeaf_reciprocal_basis_convention():
    recip = np.array(
        [
            [2.0, 0.5, 0.0],
            [0.0, 1.5, 0.0],
            [0.0, 0.0, 0.7],
        ]
    )
    bxsf = BXSFData(
        filename='nonorthogonal.bxsf',
        fermi_energy=0.0,
        nx=2,
        ny=2,
        nz=2,
        recip_au=recip,
        recip_ang=recip,
        energies=np.zeros((2, 2, 2)),
    )

    geom = make_slice_geometry(bxsf, numint=2, theta=0.0, phi=0.0)
    frac = np.array([0.25, 0.5, 0.75])
    cart = recip.T @ frac

    assert np.allclose(geom.plr_inverse @ cart, frac)


def test_slice_indices_are_wrapped_into_periodic_grid():
    recip = np.eye(3)
    bxsf = BXSFData(
        filename='identity.bxsf',
        fermi_energy=0.0,
        nx=33,
        ny=33,
        nz=33,
        recip_au=recip,
        recip_ang=recip,
        energies=np.zeros((33, 33, 33)),
    )

    geom = make_slice_geometry(bxsf, numint=60, theta=0.0, phi=0.0)
    ti = np.arange(1, geom.numx + 1, dtype=float)
    ux, uy, uz = _bxsf_indices_from_slice(geom, ti, ti, 1, bxsf.nx, bxsf.ny, bxsf.nz)

    assert np.min(ux) >= 0.0
    assert np.min(uy) >= 0.0
    assert np.min(uz) >= 0.0
    assert np.max(ux) <= bxsf.nx - 1
    assert np.max(uy) <= bxsf.ny - 1
    assert np.max(uz) <= bxsf.nz - 1
