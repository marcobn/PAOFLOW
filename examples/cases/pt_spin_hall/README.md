# Pt: spin Hall conductivity (fully relativistic)

Platinum with spin–orbit coupling from fully relativistic pseudopotentials: PAOFLOW computes the
bands, the DOS, the Berry curvature along the band path and the spin Hall conductivity
$\sigma^z_{xy}(E)$ from the PAO Hamiltonian built on the `minimal` basis of the pseudopotential.

## Run it

- **PAOFLOW Studio**: *Examples…* → `pt_spin_hall`. Studio copies the case, fills the `run` form
  and runs the steps below in the Runs tab.
- **Command line**: `paoflow-gen --example pt_spin_hall` copies the case into `./pt_spin_hall`
  and writes the drivers; run the steps below in that folder.
- **Python**: the `main.py` written by the command above is the workflow, a plain PAOFLOW script;
  read it to change a call.

## Steps

| step | command | run time |
| ---- | ------- | -------- |
| Self-consistent run | `pw.x -in scf.in` | not measured |
| Non-self-consistent run on the full k grid | `pw.x -in nscf.in` | not measured |
| PAOFLOW | `mpirun -np 4 python main.py` | not measured |
| Figures | `python plot.py` (or the Plots tab) | a few seconds |

`main.py` projects on the pseudo-atomic orbitals of the pseudopotential, which it generates in
`BASIS_PS/` on the first run; choose *projwfc* in the form to use QE's `projwfc.x` instead, and
Studio adds `proj.in` and its run.

## Outputs

- `output/bands_0.dat`: the bands along the path of the fcc lattice (eV).
- `output/dosdk_0.dat`: the DOS (eV, states/eV).
- `output/Omega_z_xy.dat`: the Berry curvature $\Omega_{xy}$ of the occupied bands along the band path.
- `output/shcEf_z_xy.dat`: $\sigma^z_{xy}$ against the energy ($(\hbar/e)$ S/cm).

Explained in: [Linear response](https://paoflow.readthedocs.io/en/latest/learn/tutorials/tutorial03.html)
