# Si: bands, DOS and Boltzmann transport

Bulk silicon on the internal projections: PAOFLOW generates a basis of pseudo-atomic orbitals
from the pseudopotential, projects the Kohn–Sham wavefunctions onto it, and computes from the
PAO Hamiltonian the band structure, the total and projected DOS, and the conductivity, Seebeck
coefficient and electronic thermal conductivity at constant relaxation time.

## Run it

- **PAOFLOW Studio**: *Examples…* → `si_bands_transport`. Studio copies the case, fills the `run`
  form and runs the steps below in the Runs tab.
- **Command line**: `paoflow-gen --example si_bands_transport` copies the case into
  `./si_bands_transport` and writes the drivers; run the steps below in that folder.
- **Python**: the `main.py` written by the command above is the workflow, a plain PAOFLOW script;
  read it to change a call.

## Steps

| step | command | run time |
| ---- | ------- | -------- |
| Self-consistent run | `pw.x -in scf.in` | not measured |
| Non-self-consistent run on the full k grid | `pw.x -in nscf.in` | not measured |
| PAOFLOW | `mpirun -np 4 python main.py` | not measured; the basis generation of the first run takes about 2 minutes |
| Figures | `python plot.py` (or the Plots tab) | a few seconds |
| Plots of the bands, DOS and PDOS with GPAO (how-to) | `python plotting.py` | about 10 s on 1 core |

`main.py` projects on the `standard` basis, which it generates from the pseudopotential in
`BASIS_PS/` on the first run; choose *projwfc* in the form to use QE's `projwfc.x` instead, and
Studio adds `proj.in` and its run.

## Outputs

- `output/bands_0.dat`, `output/kpath_points.txt`: the bands along the fcc path (eV, relative to
  $E_F$) and the high-symmetry points of the path.
- `output/dosdk_0.dat`: the DOS (eV, states/eV per spin and cell).
- `output/Si1_3s_pdosdk_0.dat`, ...: the PDOS of each orbital, and `output/pdosdk_sum_0.dat` their
  sum.
- `output/sigmagauss_0.dat`, `output/Seebeckgauss_0.dat`, `output/kappagauss_0.dat`,
  `output/PFgauss_0.dat`: $\sigma/\tau$, $S$, $\kappa_0/\tau$ and $S^2\sigma/\tau$ at 300 K against
  $\mu$; columns $T$, $\mu$, then $xx$, $yy$, $zz$, $xy$, $xz$, $yz$.
- `si_bands_dos.png`, `si_pdos.png`: the figures of `plotting.py`.

Explained in: [Band structure and density of states of silicon](https://paoflow.readthedocs.io/en/latest/learn/tutorials/tutorial01.html), [How to plot PAOFLOW output with GPAO](https://paoflow.readthedocs.io/en/latest/learn/howtos/plotting.html)
