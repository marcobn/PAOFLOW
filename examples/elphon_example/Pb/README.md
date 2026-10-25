# Pb electron–phonon (Eliashberg) — PAOFLOW on EPW's coarse coupling

This example computes the isotropic Eliashberg properties of fcc lead (α²F, λ,
ω_log, Tc) with PAOFLOW's PAO interpolation. It then computes the
temperature-dependent superconducting gap from the isotropic Migdal–Eliashberg
equations, as in the second half of the EPW tutorial. The coarse coupling comes from
**EPW** (`epbwrite`). EPW evaluates both ψ_k and ψ_{k+q} from the nscf save, so
its matrix elements share the band gauge of PAOFLOW's projections on that save,
which the interpolation needs (see `docs/internals/Electron-Phonon-Coupling.md`).

The QE steps are those of the EPW tutorial 04 (Pb superconductivity,
<https://docs.epw-code.org/tutorials/tutorial_04/index.html>): 6³ coarse k and
q grids, with ph.x on the 16 irreducible q only. EPW is run **without Wannier
functions** (`wannierize = .false.`): it serves only to compute the coarse
matrix elements, and PAOFLOW does all the interpolation.

---

## Directory contents

| file | purpose |
|------|---------|
| `phonon/scf.in` | `pw.x` scf (8³ k, with symmetry) |
| `phonon/ph.in` | `ph.x` DFPT on the 6³ q-grid (irreducible q, `fildvscf`) |
| `epw/nscf.in` | `pw.x` nscf on the full 6³ grid as an explicit `K_POINTS crystal` list (`nbnd = 16`) |
| `epw/epw.in` | `epw.x` with `epbwrite = .true.`, `wannierize = .false.`, `exclude_bands = 1:5` |
| `epw/write_ukk.py` | writes `pb.ukk` (band bookkeeping EPW reads when `wannierize = .false.`) and empty `pb.bvec`/`pb.mmn` stubs |
| `epw/epw_wannier.in` | the tutorial's Wannierized EPW input (alternative; same `.epb` content) |
| `main.py` | PAOFLOW analysis: PAO electronic structure + Eliashberg on EPW's coupling |
| `me.py` | isotropic Migdal–Eliashberg on the α²F of `main.py`: gap vs T, real axis, linearised-kernel Tc |
| `plot_me.py` | plots the `me.py` results |
| `transport.py` | metal resistivity ρ(T), extended-Drude optics and thermal emissivity from α²F_tr (`main.py --transport`) |
| `plot_transport.py` | plots α²F vs α²F_tr, ρ(T) (optional measured overlay) and the emissivity |
| `optics/nscf.in`, `optics_eps.py` | optional: interband dielectric function on the extended basis (`dielectric_tensor(intraband=False)`) |

**Pseudopotential:** copy `Pb.upf` (ONCVPSP, from the EPW tutorial material) into
this directory. Both `phonon/` and `epw/` use `pseudo_dir = '../'`.

`paoflow-gen` (electron-phonon workflow, EPW source) produces this same layout
from a `pw.x` input of the system: `phonon/scf.in`, `phonon/ph.in`,
`epw/nscf.in`, `epw/epw.in`, `epw/write_ukk.py`, plus `main.elphon.py` (the
equivalent of `main.py`), `me.elphon.py` (the equivalent of `me.py`) and
`plot.elphon.py`.

---

## Procedure

### 1. Quantum ESPRESSO and EPW

```bash
cd phonon
mpirun -np 8 pw.x -in scf.in > scf.out
mpirun -np 8 ph.x -in ph.in  > ph.out
python3 /path/to/q-e/EPW/bin/pp.py          # prefix 'pb' -> collects dvscf, patterns, dyn into save/

cd ../epw
mkdir -p pb.save                                 # the nscf starts from the scf charge density
cp ../phonon/pb.save/{charge-density.dat,data-file-schema.xml} pb.save/
mpirun -np 8 pw.x -in nscf.in > nscf.out
python3 write_ukk.py                             # pb.ukk for wannierize = .false.
mpirun -np 8 epw.x -nk 8 -in epw.in > epw.out   # EPW: one process per pool (-np = -nk)
```

**Without EPW (default since October 2026).** `main.py` has
`COUPLING_SOURCE = 'dfpt'`: PAOFLOW computes the coarse coupling itself from the
`ph.x` outputs (`phonon/_ph0/`, `phonon/pb.dyn*`, or the `pp.py` `save/`) and the
nscf wavefunctions of `epw/pb.save`, so `pp.py`, `write_ukk.py` and `epw.x` are not
needed; only the scf, `ph.x` (with `fildvscf`) and the nscf steps above remain. The
native vertex reproduces EPW's `.epb` matrix elements to the rounding level (see
`docs/internals/dfpt-coupling-port.md`); set `COUPLING_SOURCE = 'epw'` to read the
`.epb` files instead (required for ultrasoft or PAW pseudopotentials).
With the native source the nscf may also be run on the irreducible wedge,
`K_POINTS automatic 6 6 6 0 0 0` (16 irreducible k instead of 216): set `UNFOLD_KPOINTS = True` in the driver and
PAOFLOW rotates the stored Bloch functions to the full grid by symmetry
(`docs/internals/dfpt-coupling-port.md`, phase 4B); EPW needs the explicit list.

The scf and the nscf run in separate directories, so the phonon calculation
keeps its own `phonon/pb.save` and the nscf overwrites nothing that ph.x or
`pp.py` produced. The nscf reads the scf charge density, hence the copy into
`epw/pb.save`.

`epw.x` writes `pb.epb1 … pb.epbN` (the coarse Bloch coupling, one file per
pool), which together with `pb.ukk` is all PAOFLOW reads. After printing "The
.epb files have been correctly written", EPW enters its Wannier stage, which
reads wannier90's `pb.bvec`/`pb.mmn`. `write_ukk.py` writes empty stubs of both
(zero b-vectors) so that stage runs on empty data and `epw.x` finishes cleanly.
Without them EPW stops there with a non-zero exit, which does not affect the
`.epb` files.
`../phonon/save/pb.phsave/patterns.1.xml` must exist (created by `pp.py`),
otherwise `epw.x` stops with "cannot open file for reading or writing".
Tested with QE/EPW 7.5 (EPW 6.0). The coarse-coupling cost scales with the
number of pools and with the kept bands, so use as many pools as k-points allow.

### 2. PAOFLOW basis

```bash
paoflow-genbasis-ps --pseudo Pb.upf --out BASIS_PS
```

`main.py` uses the `standard` configuration: 5d 6s 6p plus 6d 7s 7p, 18 orbitals.

### 3. PAOFLOW analysis

```bash
mpirun -np 8 python main.py               # dense k 48^3, dense q 24^3 (about 2 min on 8 cores)
mpirun -np 8 python main.py --nk 24 --nq 24
mpirun -np 8 python main.py --coarse-q    # q kept on the coarse 6^3 grid
```

Output in `output/`: `alpha2F.dat` (ω in meV, α²F) and `eliashberg.npz` (all
result arrays).

### 4. Superconducting gap (Migdal–Eliashberg)

```bash
python me.py                    # serial, a few seconds
python plot_me.py               # figure -> output/me/migdal_eliashberg.png
python me.py --a2f epw/pb.a2f   # optional: the same solver on EPW's own alpha^2F
```

`me.py` uses the parameters of the tutorial's Eliashberg steps:
- `muc = 0.1`, `wscut = 0.1` eV, `npade = 90`
- the 23 temperatures 0.3–6.0 K
- `degaussq = 0.15` meV (0.5 meV and 25 temperatures 0.25–6.25 K for the
  linearised kernel)

It writes `output/me/` in EPW's formats:
- `pb.imag_iso_<T>`: gap and Z on the imaginary axis
- `pb.pade_iso_<T>` and `pb.acon_iso_<T>`: real axis, from Padé and from the analytic continuation
- `pb.qdos_iso_<T>`: quasiparticle DOS
- `gap_vs_T.dat`: Δ(T) from the lowest Matsubara frequency and the two real-axis gap edges
- `max_eigenvalue.dat`: largest eigenvalue of the linearised kernel, which is 1 at Tc

### 5. Resistivity, infrared optics and thermal emissivity

```bash
mpirun -np 8 python main.py --transport   # adds alpha^2F_tr, lambda_tr, plasma frequency (2 min)
python transport.py                       # rho(T), extended-Drude eps(omega, T), emissivity
python plot_transport.py [--experiment rho.dat]   # -> output/transport/{metal_transport,emissivity}.png
# optional interband part on the extended basis (separate nscf with nbnd = 40):
mkdir -p optics/pb.save && cp phonon/pb.save/{charge-density.dat,data-file-schema.xml} optics/pb.save/
(cd optics && mpirun -np 4 pw.x -nk 4 -in nscf.in > nscf.out)
python optics_eps.py
python transport.py --interband-dielectric optics/output
```

**What `--transport` adds.** It weights the e-ph double-delta sum by the
velocity change, giving the transport Eliashberg function α²F_tr and λ_tr
(`output/alpha2F_tr.dat`). It also gives the Fermi-surface plasma frequency.

**What `transport.py` does.**
- It combines the e-ph channel (Allen's DC rate, and the frequency-dependent
  rate with mass enhancement for the optics) with an optional residual
  resistivity (`--rho0`) and Fermi-liquid e-e scattering (`--b-ee`), by
  Matthiessen's rule.
- It writes `output/transport/`: `resistivity.dat`, `eps_<T>K.dat`,
  `emish_<T>K.dat`, `emist.dat` and `emissivity.npz`.
- The same channels also drive PAOFLOW's Boltzmann transport:
  `pf.transport(scattering_channels=[TransportSpectralChannel.from_npz(...)])`.

**Pb results** (48³/24³, σ = 0.05 eV, band group velocities):
- λ = 1.20, λ_tr = 0.94, ħω_p = 8.34 eV.
- Resistivity in µΩ cm, linear above about 100 K:

  | T (K) | 20 | 50 | 100 | 200 | 300 | 400 | 600 |
  |---|---|---|---|---|---|---|---|
  | ρ | 0.52 | 2.4 | 5.2 | 10.8 | 16.2 | 21.7 | 32.6 |

  Experiment gives about 21 µΩ cm at 300 K.
- An earlier Boltzmann run on PAOFLOW's 48³ grid with the same channel, at
  the same chemical potential, gave 19.3 µΩ cm at 300 K. That run used
  the former non-local velocity term (removed in October 2026), which also
  altered the band velocities.
- Total hemispherical emissivity of a clean surface: 0.012 (100 K), 0.037
  (300 K), 0.071 (600 K).
- The interband part changes the totals by less than 2%. Its main feature is
  the parallel-band absorption near 2 eV.
- At room temperature the extended and simple Drude models nearly coincide,
  because k_BT is already comparable to the phonon energies. The Holstein
  enhancement of the infrared absorption appears for T well below the Debye
  temperature.

**Convergence.**
- **k-grid.** The Fermi-surface quantities need at least 48³ k at
  σ = 0.05 eV: at 24³, ħω_p is 6.9 eV and λ_tr is 1.23.
- **q-grid.** Refining q from 24³ to 48³ (k = 48³) changes λ from 1.20 to 1.14,
  λ_tr from 0.94 to 0.87 and ρ(300 K) by about 6%. ω_p does not
  depend on q.
- **Equal grids.** Equal k- and q-grids are not required for these
  Fermi-surface averages (the anisotropic Migdal–Eliashberg equations do need
  them), but the q-grid must still be converged.

---

## Reference results (Pb, μ* = 0.1, σ = 0.05 eV)

| | λ | ω_log | Tc McMillan | Tc Allen–Dynes |
|---|---|---|---|---|
| PAOFLOW, k 48³ / q 24³ (this example: EPW `wannierize = .false.`) | 1.221 | 4.50 meV | 4.77 K | 5.21 K |
| PAOFLOW, k 24³ / q 24³ (this example) | 1.066 | 5.19 meV | 4.61 K | 4.94 K |
| PAOFLOW, k 48³ / q 24³, on the Wannierized EPW run (`epw_wannier.in`) | 1.221 | 4.50 meV | 4.77 K | 5.21 K |
| EPW itself, k 48³ / q 24³ (Wannierized run) | 1.151 | 4.44 meV | 4.38 K | 4.76 K |
| EPW tutorial 04 | 1.158 | 4.40 meV | 4.37 K | 4.75 K |

Migdal–Eliashberg (`me.py`) on the k 48³ / q 24³ α²F, against the tutorial
(EPW's α²F, λ = 1.158):

| | Δ(iω₀), 0.3 K | Z(iω₀), 0.3 K | gap edge (Padé / acon), 0.3 K | Tc, linearised kernel | Tc, Δ²(T) → 0 |
|---|---|---|---|---|---|
| PAOFLOW, k 48³ / q 24³ | 1.016 meV | 2.110 | 1.037 / 1.037 meV | 5.69 K | 5.71 K |
| PAOFLOW, k 24³ / q 24³ | 0.937 meV | 1.997 | 0.952 / 0.952 meV | 5.42 K | 5.44 K |
| EPW tutorial 04 | 0.923 meV | 2.071 | | ≈ 5.25 K | |

The larger gap and Tc follow the larger λ and ω_log of the PAO α²F:
(Z₀ − 1)/λ = 0.925 in both PAOFLOW and EPW.

The remaining difference from EPW follows the density of states at E_F of the
two band interpolations: 0.251 /eV/spin for the 18-orbital PAO bands against
0.235 for EPW's 4 Wannier functions. `--coarse-q` (λ ≈ 1.6) is not converged in
q and is shown only as the cheaper workflow.

---

## Notes

- **Commensurate grids:** the coarse q-grid must divide the coarse k-grid (here
  6³ / 6³), and `NK_DENSE` must be a multiple of `NQ_DENSE`.
- **Explicit k list:** EPW requires the full grid as `K_POINTS crystal` in
  [0, 1). `PAOFLOW.gen.epw_inputs.kpoints_card((6, 6, 6))` writes exactly the
  list in `epw/nscf.in`. PAOFLOW recovers the grid from the list and must be run
  with `pao_hamiltonian(expand_wedge=False)`, as in `main.py`. With
  `COUPLING_SOURCE = 'dfpt'` the nscf may use `K_POINTS automatic 6 6 6 0 0 0`
  instead, with `UNFOLD_KPOINTS = True`.
- **Band exclusion without Wannier functions:** with `wannierize = .false.`
  EPW takes the excluded bands from `pb.ukk` (`write_placeholder_ukk(...,
  exclude_bands=..., nelec=...)`); `bands_skipped` in `epw.in` only documents
  the choice. `PAOFLOW.gen.epw_inputs.epw_input` writes the matching `epw.in`,
  and `paoflow-gen` (electron-phonon workflow) generates both.
- **Wannierized EPW runs** work too (`epw/epw_wannier.in`), as long as the
  outer disentanglement window contains every band (no `dis_win_min/max`).
