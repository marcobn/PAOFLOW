# Electron-phonon examples

One folder per system, the Quantum ESPRESSO runs under `qe/`, one `system.py` with the
configuration and one subfolder per property, so that every workflow can be run on its own
once the DFT/DFPT assets of the system exist:

```
<System>/
  README.md             the system, the QE steps, what every workflow shows, the numbers
  REPORT.md             (where a reference run has been documented)
  system.py             THE configuration: prefix, grids, masses, nelec, basis, long-range
                        model, cache; plus the per-workflow defaults (SUPERCONDUCTIVITY,
                        OPTICS, TRANSPORT, ARPES, PHONONS, COUPLING)
  qe/                   phonon/ (scf.in, ph.in), nscf/ (nscf.in [, epw.in, write_ukk.py]),
                        optics/ (nscf.in with many bands), run_qe.sh, kmesh.py;
                        the pseudopotentials go here (outputs are git-ignored)
  cache/                the coarse vertex g(R_e, R_p), built once and shared by every workflow
  phonons/              phonons.py          q2r + matdyn in PAOFLOW, LO-TO term, dispersion + DOS
  coupling/             quadrupoles.py      dynamical quadrupoles -> quadrupole.fmt
                        long_range_check.py |g(Gamma, q)| with/without the dipole model (polar)
  superconductivity/    eliashberg.py me.py me_aniso.py plot_me.py
  optics/               absorption.py emissivity.py dielectric.py plot_absorption.py
  transport/            mobility.py plot_mobility.py | resistivity.py plot_resistivity.py (metals)
  arpes/                arpes.py
  figures/
```

| system | what it shows | workflows |
|---|---|---|
| [`Si`](Si/README.md) | phonon-assisted absorption and thermal emissivity, carrier mobility, phonon linewidths and ARPES, quadrupoles | phonons, coupling, optics, transport, arpes |
| [`Pb`](Pb/README.md) | Eliashberg α²F, λ, Tc, Migdal–Eliashberg gap, resistivity / infrared optics / emissivity of a metal, the ARPES kink | phonons, superconductivity, optics, transport, arpes |
| [`MgB2`](MgB2/README.md) | two-gap superconductor: anisotropic Migdal–Eliashberg equations, the σ-band kink | phonons, superconductivity, arpes |
| [`SiC`](SiC/README.md) | polar crystal: Fröhlich coupling, quadrupoles, LO-TO splitting, mobility of a polar semiconductor | phonons, coupling, transport |

## How a system folder works

- **`system.py`** builds one `ElphonSystem` (`PAOFLOW.elphon.workflows`): the QE prefix, the
  coupling source (`'dfpt'`: PAOFLOW's native vertex from the `ph.x` `dvscf` files and the nscf
  wavefunctions; `'epw'`: EPW's `.epb` files), the coarse k- and q-grids, the masses, `nelec`,
  the PAO basis, the long-range (dipole + quadrupole) treatment and the cache directory. The
  dictionaries below it hold the defaults of the workflow drivers; every command-line option
  of a driver overrides them.
- **The drivers are thin.** Each one loads the nearest `system.py` (`load_system`), parses its
  options, calls one physics-named function of `PAOFLOW.elphon.workflows`
  (`eliashberg`, `phonon_assisted_absorption`, `carrier_mobility`, `metal_resistivity`,
  `fan_migdal`, `phonon_dispersion`, `quadrupoles`, ...) and writes `output/` next to itself.
  The drivers are identical in every system folder: they are copies of
  `src/PAOFLOW/gen/elphon_templates/` (a unit test keeps them in step), which is also what
  `paoflow-gen elphon` writes.
- **Dependencies between workflows are files**: `superconductivity/output/eliashberg.npz`
  (α²F, α²F_tr) is read by `transport/resistivity.py` and `arpes/arpes.py`,
  `superconductivity/output/fs_coupling.npz` by `me_aniso.py` and the anisotropic ARPES
  self-energy, `coupling/quadrupole.fmt` by `system.py` (`quadrupoles=`),
  `qe/optics/output` (the extended-basis dielectric function of `optics/dielectric.py`) by
  `optics/absorption.py --emissivity` and `transport/resistivity.py`.
- **Run directories.** To run a system elsewhere, copy `system.py` and the workflow folders
  you need, and point `phonon_dir`, `nscf_dir`, `save_dir`, `basis_dir` of the
  `ElphonSystem` at the data (absolute paths are fine).

## Generating a new system

```bash
paoflow-gen            # workflow: elphon
```

asks for a `pw.x` input, the prefix, the coupling source, the grids, `nbnd`, the masses,
`nelec`, whether the system is a metal, the PAO basis, a multi-select of workflows and the
per-workflow settings, and writes the folder above (`system.py`, `qe/` with `run_qe.sh`, the
selected workflow folders). Then:

```bash
cp <pseudopotentials> qe/ && (cd qe && NP=8 bash run_qe.sh)      # scf, ph.x (fildvscf), nscf [, EPW, optics nscf]
paoflow-genbasis-ps --pseudo-dir qe --out BASIS_PS --preset standard
mpirun -np 8 python <workflow>/<driver>.py                        # see the README of each system
```

See `docs/internals/Electron-Phonon-Coupling.md` (method, validation, the workflow layer) and
`docs/internals/cli-generators.md` (the generator).

## References

- **PAOFLOW:** M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S. Curtarolo, R. De Gennaro, M. Fornari, L. Liyanage, A. R. Supka and H. Wang, *PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the projections of electronic wavefunctions on atomic orbital bases, including characterization of topological materials*, Comput. Mater. Sci. **143**, 462 (2018); F. T. Cerasoli, A. R. Supka, A. Jayaraj, M. Costa, I. Siloi, J. Sławińska, S. Curtarolo, M. Fornari, D. Ceresoli and M. Buongiorno Nardelli, *Advanced modeling of materials with PAOFLOW 2.0*, Comput. Mater. Sci. **200**, 110828 (2021).
- **PAO projection:** L. A. Agapito, A. Ferretti, A. Calzolari, S. Curtarolo and M. Buongiorno Nardelli, Phys. Rev. B **88**, 165127 (2013); L. A. Agapito, S. Ismail-Beigi, S. Curtarolo, M. Fornari and M. Buongiorno Nardelli, Phys. Rev. B **93**, 035104 (2016).
- **Electron-phonon interaction from first principles (review):** F. Giustino, Rev. Mod. Phys. **89**, 015003 (2017).
- **DFPT:** S. Baroni, S. de Gironcoli, A. Dal Corso and P. Giannozzi, Rev. Mod. Phys. **73**, 515 (2001).
- **Quantum ESPRESSO:** P. Giannozzi *et al.*, J. Phys.: Condens. Matter **21**, 395502 (2009); **29**, 465901 (2017).
- **EPW** (the reference implementation the examples are validated against): S. Poncé, E. R. Margine, C. Verdi and F. Giustino, Comput. Phys. Commun. **209**, 116 (2016); H. Lee *et al.*, *Electron–phonon physics from first principles using the EPW code*, npj Comput. Mater. **9**, 156 (2023).
- **Wannier-type interpolation of the coupling:** F. Giustino, M. L. Cohen and S. G. Louie, Phys. Rev. B **76**, 165108 (2007).
- **Adaptive smearing:** J. R. Yates, X. Wang, D. Vanderbilt and I. Souza, Phys. Rev. B **75**, 195121 (2007).
