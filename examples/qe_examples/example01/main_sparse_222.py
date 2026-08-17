# nx=ny=nz=2 counterpart of main_sparse.py: N = 2^(nx+ny+nz) = 64 cells,
# nawf 18 -> 1152.  main_sparse.py is NOT modified; it stays the nx=1
# dense-parity reference.
#
# Run as:
#     OPENBLAS_NUM_THREADS=1 mpirun -n 4 python main_sparse_222.py
# The eigensolve at this size runs on the dense LAPACK branch, which is
# BLAS-threaded; 4 ranks each spawning threads would oversubscribe.
#
# Parameter choices that are NOT the defaults, and why:
#
#   nfft = 6,6,6   An n1 x n2 x n3 mesh in the supercell BZ samples
#                  (n1*2^nx) x (n2*2^ny) x (n3*2^nz) primitive-BZ points.
#                  main_sparse.py at nx=1 with 12^3 samples 24^3; here each
#                  axis folds by 4, so 24/4 = 6 gives *exactly the same*
#                  24^3 primitive sampling with 216 supercell k-points
#                  instead of 1728.  The Yates dk = (8 pi^3/omega/nkpnts)^(1/3)
#                  is then numerically identical between the two runs
#                  (omega_8 * 1728 = omega_64 * 216), so smearing widths, DOS
#                  and transport are directly comparable with no rescaling,
#                  and the 6^3 crystal mesh is a strict subset of the
#                  underlying 12^3 grid, so no new interpolation error.
#                  Do NOT leave interpolated_hamiltonian() at its defaults:
#                  zeros mean "twice the current grid" -> 24^3 supercell k,
#                  a 64x overshoot.
#
#   nk = 500       The band path is in the *primitive* reciprocal basis and
#                  doubling deliberately does not update b_vectors, so nk is
#                  purely a plotting-resolution knob and is unchanged by
#                  doubling.  Against ~360 folded bands, 2000 path points
#                  render as a solid smear.
#
#   energy_window  Sizes nev from the property range instead of bnd (=576
#                  here).  MUST come after doubling_Hamiltonian: the dense
#                  doubling_attr_arry doubles attr['bnd'] on every call.
#                  Note this changes the meaning of attr['bnd'], so
#                  bands_0.dat has ~360 columns rather than 576 and is not
#                  column-comparable with an unwindowed run.
#
#   rcut           Left at None on purpose.  It is a second, physically
#                  different truncation axis (bond length, not matrix-element
#                  magnitude), no value has been calibrated against output/
#                  yet, and with the encoded-key bond handling the run fits
#                  without it.  Calibrate at nx=1 against a dense reference
#                  before turning it on here.
#
#   do_pdos=False  1152 orbitals would mean 1152 orb_*_pdosdk_0.dat files.
#                  Consequence: compare_sparse.ipynb will not run against
#                  output_sparse_222/ unmodified, since it expects them.
from PAOFLOW.SparsePAOFLOW import SparsePAOFLOW


def main():
    paoflow = SparsePAOFLOW(
        savedir='silicon.save',
        outputdir='output_sparse_222',
        smearing='gauss',
        npool=1,
        verbose=True,
        threshold=1.0e-4,
        rcut=None,
        solver='auto',
    )
    paoflow.read_atomic_proj_QE()
    paoflow.projectability()
    paoflow.pao_hamiltonian()

    paoflow.doubling_Hamiltonian(nx=2, ny=2, nz=2)  # N = 64, nawf = 1152
    paoflow.energy_window(emin=-12.0, emax=2.2, margin=1.0)

    paoflow.bands(ibrav=2, nk=500)
    paoflow.interpolated_hamiltonian(nfft1=6, nfft2=6, nfft3=6)
    paoflow.pao_eigh()
    paoflow.gradient_and_momenta()
    paoflow.adaptive_smearing()
    paoflow.dos(emin=-12.0, emax=2.2, ne=1000, do_pdos=False)
    paoflow.transport(emin=-12.0, emax=2.2)

    paoflow.finish_execution()


if __name__ == '__main__':
    main()
