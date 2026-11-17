# GaAs: n-type transport with scattering models

The chemical potential of GaAs doped with 3.5e17 electrons per cm^3 between 380 and 812 K, then conductivity, resistivity and Seebeck coefficient with acoustic, polar-optical, piezoelectric and ionized-impurity scattering (internal projections on PAW pseudopotentials).

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name. Studio copies
  the inputs and the pseudopotentials, fills the **run** form and lists the runs below in the
  Runs tab.
- **Command line**: `paoflow-gen --example gaas_doped_transport` copies the case into `./gaas_doped_transport` and writes the
  drivers; then run the steps below in that folder.

## Steps

1. Self-consistent run: `pw.x -in scf.in`
2. Non-self-consistent run on the full k grid: `pw.x -in nscf.in`
3. PAOFLOW: `mpirun -np 4 python main.py`, then `python plot.py` (or the Plots tab)

Pseudopotentials (from `PSEUDOS/library`): `Ga.pbe-dn-kjpaw_psl.1.0.0.UPF`, `As.pbe-n-kjpaw_psl.1.0.0.UPF`.

## Notes from the original example (`examples/qe_examples/example10`)

Transport properties of GaAs with energy and temperature dependent models for the relaxation time, tau. The interpolated k-mesh should be made finer to achieve convergence. To facilitate memory capacity of typical machines and ensure a reasonable runtime, in this example the mesh is turned down to 40x40x40. The chemical potential of the n-doped crystal lies in the gap just below the conduction-band minimum (the DOS is resolved on 8000 points over [-20, 2] eV so that the valence count reaches nelec). The conduction-band minimum of GaAs is a single light valley at Gamma, which the 40x40x40 mesh samples with one k-point (the next conduction states lie 0.2 eV higher): the chemical potential is set by the weight of that point (2 meV below the minimum at 380 K, against about 20 meV for a converged mesh) and the resistivity is far from converged.

Smearing: the DOS of the doping step uses a fixed 0.01 eV Gaussian and the transport uses the Fermi-Dirac derivative (smearing=None). The adaptive widths of adaptive_smearing() (floored with min_width=0.01, so the DOS counts every state) are 0.3-0.7 eV for the dispersive bands on this mesh, and their tails fill the gap far above the doping of 1.6e-5 electrons per cell, so the adaptive DOS cannot fix the chemical potential of the doped crystal. The flag at the top of main.py turns the adaptive kernel on for the transport step only.

main.py:
* Build the PAO Hamiltonian.
* Interpolate to a finer k-mesh.
* Compute the eigenvalues, gradient, and momenta.
* Compute the chemical potential for a range of temperatures at the specified doping concentration.
* Define a user-defined tau model.
* Calculate the transport properties with a combination of built-in and user-defined models for tau.
