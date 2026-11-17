# SnTe monolayer: mirror Chern number

The SnTe (001) monolayer with fully relativistic PAW pseudopotentials and the minimal internal projections in the augmented spinor metric: mirror Chern number of the horizontal mirror (Z2Pack).

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name. Studio copies
  the inputs and the pseudopotentials, fills the **run** form and lists the runs below in the
  Runs tab.
- **Command line**: `paoflow-gen --example snte_mirror_chern` copies the case into `./snte_mirror_chern` and writes the
  drivers; then run the steps below in that folder.

## Steps

1. Self-consistent run: `pw.x -in scf.in`
2. PAOFLOW: `mpirun -np 4 python main.py`, then `python plot.py` (or the Plots tab)

Pseudopotentials (from `PSEUDOS/library`): `Sn.rel-pbe-dn-kjpaw_psl.1.0.0.UPF`, `Te.rel-pbe-dn-kjpaw_psl.1.0.0.UPF`.

## Notes from the original example (`examples/qe_examples/example17`)

SnTe — mirror Chern number (``minimal`` projection preset).

Self-contained driver: assumes ``pw.x`` has already produced
``SnTe.save/`` in this directory (run ``scf.in``).

Runs

    projections -> projectability -> pao_hamiltonian -> mirror_chern_number

using the ``minimal`` projection preset (pseudo-atomic wavefunctions
shipped in the Sn/Te UPFs) and reports the mirror Chern number C_M,
the Z2 index nu and the gap.  The mirror Chern step requires the
optional ``z2pack`` and ``tbmodels`` packages.

The pseudopotentials are fully relativistic PAW: the projections, the
overlap of the atomic orbitals and the orthonormality of the stored
spinor bands are taken in the augmented spinor metric
S = 1 + sum q^{s1 s2}_{kl} |beta_k s1><beta_l s2| (QE's ``qq_so``),
built automatically by ``projections()``; see
docs/internals/internal-basis.md.  With verbose=True the run prints the
deviation of <psi|S|psi> from the identity (6e-6 here) and the number of
projectors (68).

Reference output (QE 7.5, the scf.in of this folder, October 2026):

    [minimal] nawf =  36   Pn>0.95 bands =  32   C_M = +2.000   nu = +0.000   gap = +0.083 eV

with a mirror-symmetry residual [H, M_z] of 2e-6 over a 36 eV bandwidth.
