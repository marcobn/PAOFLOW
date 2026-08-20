"""Sanity gate: bulk PAO band structure of the 15-atom Bi2Se3 cell.

Run this before ``main.py``. It is cheap and it catches the two failure modes
that otherwise show up only as a featureless surface spectrum:

*   A bad projection. Watch the projectability report -- if a large number of
    states fall below ``pthr``, the PAO manifold does not span the occupied
    bands and the Hamiltonian is not trustworthy. Raising ``nbnd`` in
    ``build_inputs.py`` is the usual fix.
*   A closed or wrongly placed gap. Bi2Se3 should show a ~0.3 eV direct gap at
    Gamma with the valence-band maximum slightly off Gamma. If the gap is closed
    here it will be closed in the surface calculation too, and there will be no
    Dirac cone to find.

Compare ``output/paoflow/bands_pao_0.dat`` against the eigenvalues in
``output/qe/nscf.out``; the PAO bands should track the DFT bands closely over
the whole valence manifold and for several eV above E_F.
"""

from PAOFLOW import PAOFLOW


def main():
    paoflow = PAOFLOW.PAOFLOW(
        savedir='output/qe/bi2se3.save',
        outputdir='output/paoflow',
        smearing='gauss',
        npool=1,
        verbose=True,
    )

    paoflow.read_atomic_proj_QE()
    paoflow.projectability(pthr=0.90)
    paoflow.pao_hamiltonian(shift_type=1, expand_wedge=False)

    # Bulk hexagonal path. Gamma-M-K-Gamma runs in the (0001) plane -- the same
    # plane the surface calculation samples -- and A adds the out-of-plane
    # direction, whose near-flatness is the signature of the van der Waals gap.
    paoflow.bands(
        ibrav=4,
        band_path='gG-M-K-gG-A',
        nk=400,
        fname='bands_pao',
    )

    paoflow.finish_execution()


if __name__ == '__main__':
    main()
