"""Surface spectral function of Bi2Se3(0001) from the NEGF surface Green's function.

This is the PAOFLOW/QE counterpart of the WannierTools Bi2Se3 example
(https://www.wanniertools.org/examples/bi2se3/), which builds the same quantity
from a VASP + Wannier90 Hamiltonian. Here the tight-binding Hamiltonian comes
from projecting the QE spin-orbit bands onto pseudo-atomic orbitals, so no
Wannierisation step is needed and the basis is fixed by the pseudopotentials.

Method
------
The bulk PAO Hamiltonian of the 15-atom hexagonal cell is partitioned into
principal layers along the surface normal a3 = c. One principal layer is one
cell = three quintuple layers, and the cell boundary sits in the van der Waals
gap (see ``build_inputs.py``), so the stack terminates the way Bi2Se3 cleaves.

The conductor Green's function is then evaluated in *surface* mode, i.e. with
the right-lead self-energy dropped:

    G_s(k, E) = [ (E + i.delta) S - H_00 - Sigma_L ]^-1

Sigma_L is the self-energy of the semi-infinite stack below, obtained from the
transfer-matrix iteration, so the system is a semi-infinite crystal with one
exposed (0001) face. Its spectral function

    A(k, E) = -1/pi * Im Tr G_s(k, E)

is the surface-projected spectral density. Sweeping the surface-BZ path
K-Gamma-M and the energy grid gives the (ne x nk) map in ``surfband_surf.dat``.

What to look for
----------------
A single Dirac cone at Gamma inside the ~0.3 eV bulk gap. Where the crossing
falls relative to E_F = 0 is not a prediction of this calculation: for a gapped
system the smeared Fermi level is pinned only to somewhere inside the gap, so
the cone can come out slightly above or below zero. What is meaningful is the
cone itself, its linear dispersion, and that it is the only spectral weight in
the gap. It is absent from the bulk band structure and absent from a slab thin
enough for the two faces to hybridise -- it exists only because this calculation
is genuinely semi-infinite.

Caveat: the trace runs over all 270 orbitals of the principal layer, i.e. over
all three quintuple layers, not just the outermost one. WannierTools projects
onto the top layer instead, which suppresses the bulk continuum. Here the
continuum is roughly three times brighter than in the WannierTools figure; the
gap region is unaffected, since nothing else lives there.

Run the QE steps first (see ``job.sh``), then this script.
"""

from PAOFLOW import PAOFLOW
from PAOFLOW.Transport import Transport


def main():
    paoflow = PAOFLOW.PAOFLOW(
        savedir='output/qe/bi2se3.save',
        outputdir='output/paoflow',
        smearing='gauss',
        npool=1,
        verbose=True,
        save_overlaps=True,
    )

    paoflow.read_atomic_proj_QE()
    # Bi 5d/6s/6p and Se 3d/4s/4p give 18 spinor orbitals per atom, so
    # nawf = 15 x 18 = 270. Check the reported projectability: with nbnd = 300
    # the PAO manifold should be well spanned and few states fall below pthr.
    paoflow.projectability(pthr=0.90)
    paoflow.pao_hamiltonian(
        shift_type=1,
        expand_wedge=False,
    )
    paoflow.projections()

    transport = Transport(paoflow.data_controller)

    # Surface normal is a3 = c. The 15 atoms sit on 15 distinct z planes and all
    # of them go into the central region, so the principal layer is the whole
    # cell: dimC = 270.
    transport.define_partition(central_atoms='ALL', transport_direction='z')

    # Surface-BZ path. ibrav=4 is passed explicitly because the SCF cell is
    # specified with A/C rather than celldm. The tabulated hexagonal points
    # K = (1/3, 1/3, 0), Gamma and M = (1/2, 0, 0) already lie in the surface
    # plane, so projecting out k_z leaves them untouched. Gamma sits mid-path so
    # the Dirac cone appears centred, as in the WannierTools figure.
    #
    # Must be configured BEFORE build_hamiltonian_blocks: the k-path is built
    # during Hamiltonian preparation.
    transport.configure_surface_bands(
        ibrav=4,
        band_path='K-gG-M',
        nk_path=201,
    )

    # Bulk/surface run: lead and conductor are the same material, no alignment.
    transport.configure_onsite_shifts(shift_L=0.0, shift_C=0.0, shift_R=0.0, shift_corr=0.0)
    transport.configure_lead_convergence(
        niterx=400, transfer_thr=1.0e-7, nprint=50, nfailx=5, surface=True
    )

    # Energies are referenced to E_F = 0: PAOFLOW subtracts the SCF Fermi level
    # when reading the QE eigenvalues, so no efermi_bulk shift is needed.
    #
    # delta sets the linewidth of the surface state. 0.01 eV resolves the Dirac
    # cone inside the ~0.3 eV gap; going much smaller sharpens the cone but slows
    # the transfer-matrix iteration, so raise niterx alongside it.
    transport.configure_energy_grid(
        emin=-1.0,
        emax=1.0,
        ne=401,
        delta=0.01,
    )

    transport.configure_outputs(
        output_dir='./output/paoflow',
        postfix='_surf',
    )

    transport.build_hamiltonian_blocks(
        calculation_type='bulk',
        use_sym=False,
        do_overlap_transformation=False,
    )

    spectral_map = transport.compute_surface_bands()
    print('surface spectral map A(k,E):', spectral_map.shape)


if __name__ == '__main__':
    main()
