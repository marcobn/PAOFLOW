# Quickstart

This page runs one calculation, the bands, density of states and Boltzmann transport of bulk silicon, in the three ways PAOFLOW offers: in the browser with PAOFLOW Studio, on the command line with `paoflow-gen`, and as a Python script. The three are the same calculation: they start from the case `si_bands_transport` of `examples/cases`, write the same `main.py` and produce the same files.

## Prerequisites

- PAOFLOW from a clone of the repository, which holds the cases (see [Installation](installation.md)):

  ```bash
  git clone https://github.com/marcobn/PAOFLOW.git
  cd PAOFLOW
  pip install -e ".[graphics,gui]"
  ```

- Quantum ESPRESSO, with `pw.x` on the `PATH`: the case starts from its inputs, a self-consistent and a non-self-consistent run of Si.

## The case

A case is a folder with the inputs of one calculation, a `README.md` that lists its steps and a `case.json` with the answers that `paoflow-gen` turns into a driver. `si_bands_transport` holds `scf.in` and `nscf.in` (Si on a $12\times12\times12$ grid) and names its pseudopotential. Its steps are:

1. `pw.x -in scf.in`, the self-consistent run;
2. `pw.x -in nscf.in`, the non-self-consistent run on the full k-grid, which writes `silicon.save/`;
3. `mpirun -np 4 python main.py`, the PAOFLOW driver, then `python plot.py` for the figures.

## In the browser: PAOFLOW Studio

```bash
paoflow-studio --root ~/runs
```

Open the printed URL. In the Setup tab choose *Examples…* → `si_bands_transport` and a folder name: Studio copies the inputs and the pseudopotential, fills in the form of the `run` workflow with the answers of the case, and lists the three steps in the Runs tab. Run them in order, then open the output folder in the Results tab. [Working in the browser](studio/browser-gui.md) describes every tab.

## On the command line: `paoflow-gen`

```bash
paoflow-gen --example si_bands_transport
cd si_bands_transport
mpirun -np 4 pw.x -in scf.in > scf.out
mpirun -np 4 pw.x -in nscf.in > nscf.out
mpirun -np 4 python main.py
python plot.py
```

`paoflow-gen --example` copies the case into `./si_bands_transport`, writes `main.py` and `plot.py` from its answers, and prints the steps. `paoflow-gen --example list` lists every case, and `paoflow-gen` alone asks the questions for a calculation of your own.

## In Python

The `main.py` written above is a plain PAOFLOW script. Without the configuration constants and the helper that generates the basis of the internal projections, its workflow is:

```python
from PAOFLOW import PAOFLOW

p = PAOFLOW.PAOFLOW(savedir='silicon.save', outputdir='output', smearing='gauss')
p.projections(basispath='BASIS_PS/', configuration='standard')  # internal projections
p.projectability(pthr=0.95)
p.pao_hamiltonian()
p.bands(ibrav=2, nk=2000, fname='bands')
p.interpolated_hamiltonian(nfft1=24, nfft2=24, nfft3=24)
p.pao_eigh()
p.gradient_and_momenta()
p.adaptive_smearing()
p.dos(do_dos=True, do_pdos=True, emin=-12.0, emax=2.2, ne=1000)
p.transport(tmin=300.0, tmax=300.0, nt=1, emin=-12.0, emax=2.2, ne=1000, write_to_file=True)
p.finish_execution()
```

Edit a call or its arguments in `main.py` and run it again; Studio's Setup tab and `paoflow-gen` write the same file when you change an answer of the form instead.

## The output

`output/` holds, among others:

| File                 | Contents                                                   |
| -------------------- | ---------------------------------------------------------- |
| `bands_0.dat`        | Band structure along the fcc path of `kpath_points.txt`    |
| `dosdk_0.dat`        | Total density of states                                    |
| `*_pdosdk_0.dat`     | Density of states projected on each orbital                |
| `sigmagauss_0.dat`   | Electrical conductivity over the relaxation time, against the chemical potential |
| `Seebeckgauss_0.dat` | Seebeck coefficient                                        |
| `kappagauss_0.dat`   | Electronic thermal conductivity over the relaxation time   |

## Next steps

[Band structure and density of states of silicon](learn/tutorials/tutorial01.ipynb) walks through this `main.py` step by step: what each call computes on Si, which parameters matter, and how to read the bands and the DOS. The [Learn](learn/index.md) section continues with the other tutorials, the how-tos and the theory pages, and the [API reference](api/index.md) documents every method.
