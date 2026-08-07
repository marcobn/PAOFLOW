"""Surface-projected bulk band structure of bcc Fe(001).

Reproduces the surface band structure described in Section 7.3 / Fig. 10 of the
PAOFLOW paper. The bulk PAO Hamiltonian is partitioned into principal layers
along the surface normal (``z``); the conductor Green's function is evaluated in
surface mode, i.e. with the right-lead self-energy dropped,

    G_s(k, E) = [ (E + i.delta) S - H_00 - Sigma_L ]^-1

so that its spectral function

    A(k, E) = -1/pi * Im Tr G_s(k, E)

is the surface-projected bulk band structure. Sweeping a transverse
high-symmetry k-path and the energy grid produces the (ne x nk) spectral map
written to ``surfband_surf.dat``.

Run the QE steps first (see ``job.sh``), then this script.
"""

from PAOFLOW import PAOFLOW
from PAOFLOW.Transport import Transport


def main():
    paoflow = PAOFLOW.PAOFLOW(
        savedir='output/qe/fe.save',
        outputdir='output/paoflow',
        smearing='gauss',
        npool=1,
        verbose=True,
        save_overlaps=True,
    )

    paoflow.read_atomic_proj_QE()
    # Legacy transportPAO used atmproj_thr = 0.9 and atmproj_sh = 3.5; the
    # projectability threshold and null-space shift live in the PAOFLOW core.
    paoflow.projectability(pthr=0.90)
    paoflow.pao_hamiltonian(
        shift_type=1,
        expand_wedge=False,
    )
    paoflow.projections()

    transport = Transport(paoflow.data_controller)

    # Transport along z => the (001) surface plane is spanned by x and y.
    transport.define_partition(central_atoms='ALL', transport_direction='z')

    # Surface-projected band structure. The SCF cell is written with ibrav=0, so
    # pass the equivalent Bravais index explicitly: the cell is simple cubic
    # (bcc Fe in its conventional cubic cell), hence ibrav=1. 'gG-X' runs in the
    # surface plane. Must be configured BEFORE build_hamiltonian_blocks, since
    # the k-path is constructed during Hamiltonian preparation.
    transport.configure_surface_bands(
        ibrav=1,
        band_path='gG-X',
        nk_path=191,
    )

    # Bulk/surface run: leads and conductor are the same material, no alignment.
    transport.configure_onsite_shifts(shift_L=0.0, shift_C=0.0, shift_R=0.0, shift_corr=0.0)
    transport.configure_lead_convergence(
        niterx=200, transfer_thr=1.0e-7, nprint=20, nfailx=5, surface=True
    )

    # Energies are referenced to E_F = 0: PAOFLOW subtracts the SCF Fermi level
    # when reading the QE eigenvalues, so no efermi_bulk shift is needed.
    transport.configure_energy_grid(
        emin=-9.0,
        emax=1.0,
        ne=300,
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
