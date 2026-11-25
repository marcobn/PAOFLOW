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
| Self-consistent run | `pw.x -in scf.in` | under a second on 1 core |
| DFT bands along the path, before the nscf run, which overwrites them | `pw.x -in bands.in > bands.out` | a few seconds on 1 core |
| Non-self-consistent run on the full k grid | `pw.x -in nscf.in` | about 9 s on 1 core |
| PAOFLOW | `mpirun -np 4 python main.py` | 1 min 45 s on 1 core with the basis generation of the first run, 5.0 GB |
| Figures | `python plot.py` (or the Plots tab) | a few seconds |
| Plots of the bands, DOS and PDOS with GPAO (how-to) | `python plotting.py` | about 10 s on 1 core |
| Projections on the pseudo-orbitals | `projwfc.x -in proj.in` | a second on 1 core |
| PAO Hamiltonian and bands of each projection route (how-to) | `python projections.py > projections.log` | 18 s on 1 core |

To skip the QE runs, copy the case with `paoflow-gen --example si_bands_transport --with-data`
(in Studio, *Use the precomputed data of the case*): it extracts `silicon.save`, with the
wavefunctions that the internal projections read and the `atomic_proj.xml` of `projwfc.x`, and
`bands.out`, from `si_bands_transport-v2.tar.gz` of the
release [case-data](https://github.com/marcobn/PAOFLOW/releases/tag/case-data) (QE 7.4.1).

`main.py` projects on the `standard` basis, which it generates from the pseudopotential in
`BASIS_PS/` on the first run; choose *projwfc* in the form to read the projections of QE's
`projwfc.x` (`proj.in`) instead. `projections.py` builds the PAO Hamiltonian and the bands once for
each projection route: `projwfc.x`, the `minimal`, `standard` and `extended` presets, the basis
generated at run time and a basis chosen by hand.

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
- `output_<route>/bands_0.dat`, `projections.log`: the bands of each projection route of
  `projections.py` and its printed output, with the projectability of each band.
- `bands.out`: the DFT bands at 65 points of the band path, the reference of the projections
  how-to.

Explained in: [Band structure and density of states of silicon](https://paoflow.readthedocs.io/en/latest/learn/tutorials/tutorial01.html), [How to plot PAOFLOW output with GPAO](https://paoflow.readthedocs.io/en/latest/learn/howtos/plotting.html), [How to choose the projections onto the PAO basis](https://paoflow.readthedocs.io/en/latest/learn/howtos/projections.html)
