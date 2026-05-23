# AFLOWpi → PAOFLOW Migration Plan

> Generated from Copilot discussion — May 2026  
> Source repos: [`marcobn/AFLOWpi`](https://github.com/marcobn/AFLOWpi) | [`marcobn/PAOFLOW`](https://github.com/marcobn/PAOFLOW)

---

## Overview

The goal is to separate the phonon, thermal, and spectroscopy modules from AFLOWpi and integrate them natively into PAOFLOW — **replacing all QE post-processing binaries** (`fd.x`, `fd_ifc.x`, `matdyn.x`) with pure Python/numpy implementations. After migration, `pw.x` remains the only external dependency.

---

## AFLOWpi Source Modules Being Ported

| AFLOWpi File | Role |
|---|---|
| `src/run/src/phonon.py` | Orchestrates FD phonon workflow |
| `src/run/src/fd_phonon_sym.py` | Crystal symmetry force completion |
| `src/run/src/fd_fields.py` | Finite-field SCF for Born charges / ε∞ / Raman |
| `src/run/src/elastic.py` | ElaStic wrapper for elastic constants |
| `src/run/src/IR_spectra.py` | IR intensities from dynamical matrix + Born charges |
| `src/run/src/raman_spectra.py` | Polarization-resolved Raman spectra |
| `src/run/src/sound_velocity.py` | Sound velocities from acoustic branch fitting |
| `src/retr/src/thermal.py` | Grüneisen parameters, Callaway κ_L, Debye temperature |

---

## QE Binary Replacement Plan

All three QE post-processing binaries are pure math with no Fortran library dependencies — fully replaceable with numpy/scipy.

| Binary | Replacement Module | What it does |
|---|---|---|
| `fd.x` | `do_fd_supercell.py` | Supercell construction + displaced input generation |
| `fd_ifc.x` | `do_compute_ifc.py` | Finite-difference → IFC tensor Φ(R) |
| `matdyn.x` | `do_phonon_dispersion.py` | Fourier transform Φ(R)→D(q), LOTO correction, diagonalization → ω(q,ν) |

### `fd.x` → `do_fd_supercell.py`
For each irreducible atom and Cartesian direction:
- Copy supercell input
- Shift atom `i` by `±de` in direction `d` (Cartesian → crystal)
- Write `displaced.{p}.{d}.{a}.in`

Uses `pao_sym.py` (already in PAOFLOW) for irreducible displacement reduction.

### `fd_ifc.x` → `do_compute_ifc.py`
Finite-difference IFCs:
```
Φ_{αi,βj} = -[F_{βj}(+δ_{αi}) - F_{βj}(-δ_{αi})] / (2·de)   # central diff (innx=2)
Φ_{αi,βj} = -[F_{βj}(+δ_{αi}) - F_{βj}(0)] / de              # forward diff (innx=1)
```
- Load forces from QE XML output (`read_QE_xml.py` already exists in PAOFLOW)
- Apply symmetry completion (`do_fd_phonon_sym.py`)
- Enforce acoustic sum rule (ASR)

### `matdyn.x` → `do_phonon_dispersion.py`
1. Fourier-transform IFCs: `D(q) = Σ_R Φ(R)·e^{iq·R}`
2. LOTO non-analytic correction: `D_NA(q̂) = (4π/Ω)·(q̂·Z*_α)(q̂·Z*_β)/(q̂·ε∞·q̂)`
3. Mass-weight + diagonalize: `D(q)·e_{qν} = ω²_{qν}·e_{qν}`
4. phDOS via Gaussian or tetrahedron smearing

Reuses `get_R_grid_fft.py` and `get_K_grid_fft.py` FFT infrastructure already in PAOFLOW.

---

## Complete Module List

### New files in `src/PAOFLOW/defs/`

| New File | Source | Description |
|---|---|---|
| `qe_io.py` | AFLOWpi.retr utilities | QE input parser/writer (`_splitInput`, `_joinInput`, `_getPositions`, `getCellMatrixFromInput`, `_convertFractional`, `_getPosLabels`) |
| `do_fd_supercell.py` | Replaces `fd.x` | Supercell construction + displaced pw.x input files |
| `do_fd_phonon_sym.py` | `fd_phonon_sym.py` | Symmetry-based force array completion |
| `do_fd_fields.py` | `fd_fields.py` | Born charges Z*(κ), dielectric tensor ε∞, Raman tensors via finite fields |
| `do_compute_ifc.py` | Replaces `fd_ifc.x` | Forces → IFC tensor Φ(R), ASR enforcement |
| `do_phonon_dispersion.py` | Replaces `matdyn.x` | D(q), ω(q,ν), e(q,ν), phDOS, LOTO correction |
| `do_IR_spectra.py` | `IR_spectra.py` | IR intensities from Z* + dynamical matrix |
| `do_raman_spectra.py` | `raman_spectra.py` | Polarization-angle-resolved Raman spectra |
| `do_elastic.py` | `elastic.py` | ElaStic wrapper (optional external tool) |
| `do_thermal_properties.py` | `thermal.py` + `sound_velocity.py` | Full thermal workflow (see below) |
| `do_elph_coupling.py` | New | Electron-phonon coupling (see below) |

### Modified files

| File | Changes |
|---|---|
| `PAOFLOW.py` | Register new methods for each `do_*.py` module |
| `DataController.py` | Add new data fields for phonon/thermal/EPC quantities |
| `defs/input_default.xml` | Add parameter defaults for new modules |

---

## Thermal Properties Module: `do_thermal_properties.py`

Absorbs `src/retr/src/thermal.py` and `src/run/src/sound_velocity.py` entirely.

### Sub-component 1: Sound velocity
```python
sound_velocities(phonon_data, r_max=0.05, nk_r=8, nk_theta=30, nk_phi=30)
```
- Generate spherical q-mesh around Γ (pure numpy — from `radial_grid`)
- Evaluate ω(q) via native `do_phonon_dispersion.py` (no `matdyn.x`)
- Fit ω ∝ q slope per acoustic branch → v_TA, v_TA', v_LA in m/s
- Compute Debye temperatures θ_D per branch

### Sub-component 2: Grüneisen parameters
```python
gruneisen_parameters(omega_0, omega_plus, omega_minus, vol_0, vol_plus, vol_minus)
compute_gruneisen_projected(omega_0, omega_plus, omega_minus, vols, ap_weights)
```
- Requires 3 phonon calculations: V₀, V₊ = V₀(1+δ), V₋ = V₀(1-δ)
- Central difference: `γ(q,ν) = -(V₀/ω₀)·(ω₊ - ω₋)/(V₊ - V₋)`
- Atom-projected version using phDOS weights

### Sub-component 3: Callaway lattice thermal conductivity
```python
callaway_thermal_conductivity(v_i, theta_i, grun_i, mass, vol, T_range)
```
Direct port of `_do_therm` — already pure scipy/numpy.
Implements the Callaway model with Umklapp (τ_U) and Normal process (τ_N) scattering:

- **TA/TA' normal scattering rate**: `1/τ_N ∝ γ²·(V/M)·T⁵·x / v_s⁵`
- **LA normal scattering rate**: `1/τ_N ∝ γ²·(V/M)·T⁵·x² / v_s⁵`
- **Umklapp scattering rate**: `1/τ_U ∝ γ²·T³·x²·e^{-θ_D/3T} / (M·v_s²·θ_D)`
- Combined: `τ_C = 1/(1/τ_U + 1/τ_N)`
- Three integrals (Callaway correction) per branch via `scipy.integrate.quad`
- κ_L(T) = Σ_{branches} C_branch · [I₁ + I₂²/I₃]

Outputs: κ_L(T) per branch (TA, TA', LA) and total, over user-defined T range.

### Sub-component 4: Harmonic thermodynamic quantities (NEW — not in AFLOWpi)
```python
harmonic_thermodynamics(omega_q_nu, weights, T_range)
```
From the full phDOS `g(ω)`, computes all standard harmonic thermodynamic quantities:

| Quantity | Formula |
|---|---|
| Zero-point energy | `E_ZP = (ℏ/2) ∫ g(ω)·ω dω` |
| Vibrational free energy | `F(T) = k_BT ∫ g(ω)·ln[2sinh(ℏω/2k_BT)] dω` |
| Vibrational entropy | `S(T) = k_B ∫ g(ω)·[x·n_BE(x) - ln(1-e^{-x})] dω` |
| Heat capacity (const V) | `C_v(T) = k_B ∫ g(ω)·x²·e^x/(e^x-1)² dω` |

where `x = ℏω/k_BT` and `n_BE(x) = 1/(e^x - 1)`.

---

## Electron-Phonon Coupling Module: `do_elph_coupling.py` (New)

Enabled by having both H(k) from PAOFLOW and Φ(R) from `do_compute_ifc.py` in-memory simultaneously.

### Physical approach
- Use IFCs to construct phonon dynamical matrix D(q) and eigenvectors e(q,ν)
- Use PAOFLOW's PAO Hamiltonian H(k) from `do_build_pao_hamiltonian.py`
- Compute EPC matrix elements via finite differences of H(k):
  `g_{mn,ν}(k,q) = <mk+q| δV/δu_{qν} |nk>`
  approximated as `∂H_PAO/∂τ · e(q,ν) / √(2Mω_{qν})`
- Compute Eliashberg spectral function: `α²F(ω) = Σ_{k,q,ν} |g|²·δ(ε_k)·δ(ε_{k+q})·δ(ω-ω_{qν})`
- Electron-phonon coupling constant: `λ = 2 ∫ α²F(ω)/ω dω`
- Superconducting Tc via Allen-Dynes formula

### Key advantage over external tools
Direct in-memory access to both H(k) and Φ(R) enables EPC computation without any file I/O to external codes — impossible to do cleanly through `matdyn.x` / `ph.x` file interfaces.

---

## PAOFLOW Assets Reused

| Existing Asset | Role in Migration |
|---|---|
| `defs/read_QE_xml.py` | Force parsing from QE XML output (no new parsers needed) |
| `defs/pao_sym.py` | Symmetry operations, rotation matrices, k-path generation |
| `defs/get_R_grid_fft.py` | FFT R-grid for Φ(R)→D(q) (same infrastructure as H(R)→H(k)) |
| `defs/get_K_grid_fft.py` | FFT K-grid for Fourier interpolation |
| `defs/kpnts_interpolation_mesh.py` | q-path generation for phonon dispersion plots |
| `defs/do_build_pao_hamiltonian.py` | PAO H(k) for electron-phonon coupling |
| `defs/smearing.py` | Smearing functions for phDOS and α²F |

---

## Complete Native Pipeline

```
pw.x SCF at V₀, V₊, V₋ (only external dependency)
         ↓
do_fd_supercell.py        ← replaces fd.x
  build supercell, generate displaced inputs
         ↓
pw.x SCF on displaced supercells (forces)
         ↓
do_compute_ifc.py         ← replaces fd_ifc.x
  forces → IFC Φ(R), symmetry completion, ASR
         ↓
do_fd_fields.py
  Born charges Z*(κ), dielectric tensor ε∞
         ↓
do_phonon_dispersion.py   ← replaces matdyn.x
  D(q), LOTO correction, ω(q,ν), e(q,ν), phDOS
         ↓
do_thermal_properties.py
  ├── sound_velocities()                → v_s per acoustic branch
  ├── gruneisen_parameters()            → γ(q,ν)
  ├── callaway_thermal_conductivity()   → κ_L(T)
  └── harmonic_thermodynamics()         → F(T), S(T), C_v(T), E_ZP
         ↓
do_IR_spectra.py          IR intensities
do_raman_spectra.py       Raman tensor
do_elastic.py             Elastic constants (ElaStic, optional)
         ↓
do_elph_coupling.py       (NEW)
  g_{mn,ν}(k,q) → α²F(ω) → λ → T_c (Allen-Dynes)
```

---

## Shared Dependencies to Port First (Phase 1)

Before any physics module, extract these AFLOWpi utilities into `defs/qe_io.py`:

| AFLOWpi function | Purpose |
|---|---|
| `retr._splitInput` | Parse QE namelist input string → dict |
| `retr._joinInput` | Dict → QE namelist input string |
| `retr._getPositions` | Extract atomic positions matrix |
| `retr.getCellMatrixFromInput` | Extract cell matrix in Bohr |
| `retr._convertFractional` | Cartesian ↔ crystal coordinate conversion |
| `retr._getPosLabels` | Extract atom type labels |
| `retr._getPath` | k/q-path from input |
| `retr._cellMatrixToString` | Format cell matrix for QE input |
| `run.reduce_kpoints` | Scale k-grid for supercell |

Replace `AFLOWpi.run._oneRun` / `AFLOWpi.prep.totree` with a lightweight subprocess wrapper or pluggable executor.

---

## Implementation Order

| Phase | Modules | Priority |
|---|---|---|
| 1 | `qe_io.py` | Foundation — blocks everything else |
| 2 | `do_fd_phonon_sym.py` | Self-contained, highest value |
| 3 | `do_fd_fields.py` | Born charges / LOTO |
| 4 | `do_fd_supercell.py`, `do_compute_ifc.py` | Replace fd.x + fd_ifc.x |
| 5 | `do_phonon_dispersion.py` | Replace matdyn.x |
| 6 | `do_IR_spectra.py`, `do_raman_spectra.py` | Spectroscopy |
| 7 | `do_thermal_properties.py` | Full thermal workflow |
| 8 | `do_elastic.py` | Elastic constants |
| 9 | `do_elph_coupling.py` | Electron-phonon coupling (ultimate goal) |
