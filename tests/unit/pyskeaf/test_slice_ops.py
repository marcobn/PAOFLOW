import numpy as np

from PAOFLOW.pyskeaf.io_bxsf import BXSFData
from PAOFLOW.pyskeaf.slice_ops import make_slice_geometry


def test_plr_inverse_matches_row_vector_reciprocal_basis():
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
