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

    # Energies are referenced to E_F = 0, but for this run that zero is the
    # valence-band MAXIMUM, not a Fermi level. With occupations='fixed' QE writes
    # <fermi_energy> equal to <highestOccupiedLevel> (both 0.19345 Ha here), and
    # PAOFLOW subtracts it in read_QE_xml.py. The QE gap is 0.000 .. +0.756 eV.
    #
    # This is why the map below looks like a semiconductor while Fig. 10 looks
    # metallic: the Si(001)-1x1 dangling-bond band fills the gap from +0.29 to
    # +0.74 eV, so drawing E_F at the bulk VBM puts the zero underneath the
    # entire surface band. Fig. 10 instead puts 0 at the Fermi level of the
    # *terminated* system (the metallic slab of panel 1), which lies inside that
    # band. Nothing here computes that level: the surface Green's function is
    # built from the truncated BULK Hamiltonian with no surface self-consistency
    # and no charge-neutrality condition, so the filling is not an output of this
    # calculation. Take the shift from the slab run to align the two panels.
    #
    # Window: the Si valence band is only 11.3 eV wide (QE band 1 bottoms out at
    # -11.34 eV, the PAO Hamiltonian at -11.43 eV), so emin=-18 left a third of
    # the panel dead. More importantly, emax must clear the bulk gap: the
    # Si(001)-1x1 dangling-bond surface state disperses from +0.29 to +0.71 eV
    # and is the brightest feature in the whole map -- the counterpart of the
    # states crossing E_F in Fig. 10. An emax of 0.4 clips it after 0.1 eV.
    transport.configure_energy_grid(
        emin=-12.5,
        emax=2.0,
        ne=450,
        delta=0.05,
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
