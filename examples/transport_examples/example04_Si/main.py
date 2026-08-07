"""Surface-projected bulk band structure of Si(001).

Reproduces the lower panel of Fig. 10 of the PAOFLOW paper (Section 7.3). The
bulk PAO Hamiltonian is partitioned into principal layers along the surface
normal (``z``); the conductor Green's function is evaluated in surface mode,
i.e. with the right-lead self-energy dropped,

    G_s(k, E) = [ (E + i.delta) S - H_00 - Sigma_L ]^-1

so that its spectral function

    A(k, E) = -1/pi * Im Tr G_s(k, E)

is the surface-projected bulk band structure of a semi-infinite crystal.
Sweeping the transverse surface-BZ path Gamma-Xbar and the energy grid produces
the (ne x nk) spectral map written to ``surfband_surf.dat``.

The cell (see ``scf.in``) is the (001)-oriented simple-tetragonal 4-atom cell of
diamond Si, so a3 is the surface normal and a1/a2 span the (1x1) surface plane.

Run the QE steps first (see ``job.sh``), then this script.
"""

from PAOFLOW import PAOFLOW
from PAOFLOW.Transport import Transport


def main():
    paoflow = PAOFLOW.PAOFLOW(
        savedir='output/qe/silicon.save',
        outputdir='output/paoflow',
        smearing='gauss',
        npool=1,
        verbose=True,
        save_overlaps=True,
    )

    paoflow.read_atomic_proj_QE()
    paoflow.projectability(pthr=0.90)
    paoflow.pao_hamiltonian(
        shift_type=1,
        expand_wedge=False,
    )
    paoflow.projections()

    transport = Transport(paoflow.data_controller)

    # Transport along z = [001]. The principal layer is the whole 4-atom cell,
    # i.e. the four (001) atomic planes, so dimC = 4 x 9 = 36.
    transport.define_partition(central_atoms='ALL', transport_direction='z')

    # Surface band structure along the Gamma-Xbar line of the (1x1) surface BZ.
    # ibrav=6 (simple tetragonal) matches scf.in, and its tabulated X point sits
    # at (0, 1/2, 0), i.e. in the surface plane. Must be configured BEFORE
    # build_hamiltonian_blocks, since the k-path is built during preparation.
    transport.configure_surface_bands(
        ibrav=6,
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
        emin=-18.0,
        emax=0.4,
        ne=400,
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
