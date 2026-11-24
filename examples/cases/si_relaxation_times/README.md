# Si: Boltzmann transport with energy-dependent relaxation times

Bulk silicon: PAOFLOW computes the Boltzmann transport tensors in the constant relaxation-time
approximation, the chemical potential of n-type Si at $10^{19}$ electrons per cm$^3$ between 300
and 800 K, and its resistivity with acoustic-phonon and ionised-impurity relaxation times, one of
them a custom `TauModel`, all from the PAO Hamiltonian interpolated on a 36×36×36 mesh.

## Run it

- **PAOFLOW Studio**: *Examples…* → `si_relaxation_times`. Studio copies the case, lists its
  script and runs the steps below in the Runs tab.
- **Command line**: `paoflow-gen --example si_relaxation_times` copies the case into
  `./si_relaxation_times`; run the steps below in that folder.
- **Python**: `main.py` is the workflow; read it to change a call.

## Steps

| step | command | run time |
| ---- | ------- | -------- |
| Self-consistent run | `pw.x -in scf.in` | not measured |
| Non-self-consistent run on the full k grid | `pw.x -in nscf.in` | not measured |
| Atomic projections (read by main.py) | `projwfc.x -in proj.in` | not measured |
| Transport, doping and resistivity with relaxation-time models | `python main.py > main.log` | 8 min on 1 core, 2.9 GB |

To skip the QE runs, extract `tutorial01/silicon.save` of `tutorial_assets.tar.gz` (release
[tutorial-assets-v2](https://github.com/marcobn/PAOFLOW/releases/tag/tutorial-assets-v2)) into
this folder.

## Outputs

- `output/sigmagauss_0.dat`, `Seebeckgauss_0.dat`, `kappagauss_0.dat`, `PFgauss_0.dat`: $\sigma$,
  $S$, $\kappa_0$ and $S^2\sigma$; columns $T$, $\mu$, then $xx$, $yy$, $zz$, $xy$, $xz$, $yz$.
  Each `transport()` call rewrites them, so at the end they hold the last call of the resistivity
  loop (800 K, one $\mu$); the curves against $\mu$ at 300 K and constant $\tau$ are there only
  between the first `transport()` call and `doping()`.
- `output/doping_n1e+19.dat`: $T$ (K) and $\mu$ (eV) of n-type Si at $10^{19}$ cm$^{-3}$.
- `output/dosdk_0.dat`: the DOS that `doping()` integrates to count the electrons, against energy (eV).
- `output/resistivity.dat`: $T$ (K), the resistivity with the built-in acoustic and impurity models,
  and with the custom acoustic model and the built-in impurity model ($\Omega$ cm).

Explained in: [Boltzmann transport in silicon](https://paoflow.readthedocs.io/en/latest/learn/tutorials/tutorial02.html)
