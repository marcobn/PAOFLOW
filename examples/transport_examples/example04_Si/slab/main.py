"""Si(001) slab bands projected onto the two outermost atomic planes.

Reproduces the upper panel of Fig. 10 of the PAOFLOW paper. The band structure
of a finite Si(001) slab is computed along the Gamma-Xbar line of the (1x1)
surface BZ, and each eigenstate is weighted by

    w_nk = sum_{mu in surface planes} |v^mu_nk|^2

so that surface-localized states stand out against the slab's bulk-like
subbands. The parent directory computes the same physics for a semi-infinite
crystal via the NEGF surface Green's function (lower panel); both use the same
surface cell and the same k-path, so the panels are directly comparable.

Run ``job.sh`` (which regenerates the QE inputs via ``build_slab.py``) first.
"""

import numpy as np

from build_slab import NLAYERS, PREFIX
from PAOFLOW import PAOFLOW


def main():
    paoflow = PAOFLOW.PAOFLOW(
        savedir=f'output/qe/{PREFIX}.save',
        outputdir='output/paoflow',
        smearing='gauss',
        npool=1,
        verbose=True,
    )

    paoflow.read_atomic_proj_QE()
    paoflow.projectability(pthr=0.90)
    paoflow.pao_hamiltonian(shift_type=1, expand_wedge=False)

    # ibrav=6 (simple tetragonal) matches the slab cell; its tabulated X point
    # is (0, 1/2, 0), i.e. in the surface plane, and coincides with the Xbar
    # used by the semi-infinite calculation in the parent directory.
    paoflow.bands(ibrav=6, band_path='gG-X', nk=200)

    # Two outermost planes on each face. The slab is symmetric, so both faces
    # carry the same surface states; including all four planes just doubles the
    # signal. Use only [0, 1] to look at a single face.
    site_proj = np.array([0, 1, NLAYERS - 2, NLAYERS - 1])
    paoflow.site_projected_bands(site_proj=site_proj)

    print(f'projected slab bands onto planes {site_proj.tolist()} of {NLAYERS}')

    paoflow.finish_execution()


if __name__ == '__main__':
    main()
