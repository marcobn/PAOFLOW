# 3C-SiC: polar electron-phonon coupling in PAOFLOW — reference run and validation

Run on October 26 2026 with QE 7.5 and PAOFLOW `develop-marco` (the DFPT coupling source and the
long-range model of `docs/internals/long-range-elphon-plan.md`). The example directory holds the
inputs and drivers; the numbers below come from running them as the README describes, plus the
EPW and `matdyn.x` reference runs kept in the workspace (`Workspace/PAOFLOW/dfpt_port/sic*` and
`SiC_example_run`).

## 1. System and settings

| item | value |
|---|---|
| structure | zinc-blende SiC, a = 8.237 bohr (4.359 Å, the EPW test value) |
| pseudopotentials, cutoff | PseudoDojo nc-sr-05 PBE standard, 60 Ry |
| scf / DFPT / nscf grids | 6³ k; 6³ q (16 irreducible, `epsil`); 6³ k nscf on the irreducible wedge (16 k), 12 bands |
| PAO basis | `standard` (Si 3s3p3d4s4p, C 2s2p3s3p3d4s4p: 26 orbitals), projectability 0.90 |
| dielectric data from `ph.x` | ε∞ = 7.51; Z*(Si) = +2.65, Z*(C) = −2.83, sum rule → ±2.74 |
| experiment | ε∞ = 6.52, ε0 = 9.72; ω_TO = 796 cm⁻¹, ω_LO = 972 cm⁻¹ [1] |

PBE overestimates ε∞ by 15 %, which the LO-TO splitting inherits (below).

## 2. Phonons: `q2r_matdyn` with the dipole term

![phonon dispersion and DOS](figures/sic_phonons.png)

`phonons.py` subtracts the rigid-ion force constants (ε∞, Z*, QE's Ewald parameter
α = (a/2π)²) from the 216 coarse dynamical matrices, interpolates the remainder and restores the
term at every q, taking the limit along the path at Γ.

| quantity | PAOFLOW | `matdyn.x` (q2r.x `zasr='simple'`) | experiment [1] |
|---|---|---|---|
| ω_TO(Γ) (cm⁻¹) | 790.2 | 790.2 | 796 |
| ω_LO(Γ) (cm⁻¹) | 952.3 | 952.3 | 972 |
| max deviation on 141 path points (cm⁻¹) | 0.010 (optical branches 1e-4) | — | |

The plain interpolation (dashed) leaves the LO branch degenerate with the TO at Γ and bends it
to 952 cm⁻¹ only at the first coarse point, the signature of a missing non-analytic term.
`output/phonons/sic.fc` carries the `T` block (α, ε∞, Z*), so `matdyn.x` reads it directly.

## 3. The Fröhlich vertex against EPW

EPW 6.0 was run on the same `ph.x` data with `lpolar = .true.` and four sp³ Wannier functions
on the valence bands, and its interpolated vertex was printed (`prtgkk`, `fsthick = 100 eV`; with
the default 10 eV the deep valence band is dropped from the printout) at k = Γ and at a general
k for seven q of the 12³ fine grid. PAOFLOW interpolated the same points from the native DFPT
vertex with the dipole model subtracted on the coarse grid and restored at the fine q. The
compared quantity is gauge invariant: the sum over the four valence bands of |g|²/(2ω) per
degenerate mode group.

![LO coupling at Gamma along (1,1,0): PAOFLOW with and without the model, and EPW](figures/g_lo_vs_epw.png)

| coarse grids | PAOFLOW / EPW, LO mode at |q| = 0.17 … 1.0 (2π/a) | worst over all modes and points |
|---|---|---|
| 3³ k / 3³ q | 1.18, 1.25, 1.15, 1.00, 1.61 | 61 % |
| 6³ k / 6³ q | 1.03, 1.00, 0.99, 1.00, 0.99 | 12 % (LA mode at the smallest q) |

At a coarse (k, q) the dense path reproduces the DFPT vertex to 0.4 % (the projectability of the
standard basis) and EPW agrees with its own `.epb` to 4e-15. Away from the coarse points the two
codes interpolate the short-range remainder with different localised bases, and the PAO
long-range term carries the physical valence-subspace overlap (0.93 at |q| = 1/12 along (1,1,0),
0.42 at 1/3) where the four-band Wannier model uses the identity; both are consistent
subtract-and-restore schemes and converge together with the coarse grid, as the 6³ column shows.
The same comparison without the model (dashed) shows what the plain interpolation misses: at the
smallest q the LO coupling is four times too small.

## 4. The coupling at k = Γ along Γ-X (`coupling.py`)

![LO coupling at Gamma along Gamma-X with and without the dipole term](figures/g_frohlich.png)

`coupling.py` evaluates the band-resolved vertex at k = Γ for q along Γ-X on the 24³ dense grid,
summed over the four valence bands and divided by 2ω (meV²), three ways: with the dipole model
subtracted on the coarse grid and restored at every q, with the plain interpolation of the full
coupling, and the analytic dipole term alone.

| \|q\| (2π/a) | with the model | plain interpolation | dipole term alone |
|---|---|---|---|
| 0.083 | 6.27 × 10⁶ | 0.41 × 10⁶ | 5.92 × 10⁶ |
| 0.167 | 1.76 × 10⁶ | 0.67 × 10⁶ | 1.37 × 10⁶ |
| 0.250 | 0.96 × 10⁶ | 0.76 × 10⁶ | 0.54 × 10⁶ |
| 0.333 (coarse point) | 0.673 × 10⁶ | 0.673 × 10⁶ | 0.25 × 10⁶ |

Near Γ the coupling is the Fröhlich term (1/|q|² in |g|²) and the plain interpolation is 15 times
too small; at the first coarse q-point the two routes coincide to all digits, the
subtract-and-restore identity. Beyond |q| ≈ 0.4 the short-range part dominates and the three
curves separate as expected.

## 5. Dynamical quadrupoles (`quadrupoles.py`)

The dipole (Fröhlich) term is the leading long-range piece, but the next order of the
multipole expansion, the dynamical quadrupoles Q^κ_{βαγ}, gives a q-independent long-range
coupling that the short-range interpolation misses and that interferes with the dipole term in
polar crystals (Brunin et al. 2020, Poncé et al. 2021). PAOFLOW extracts them from twelve
single-q `ph.x` runs (`fildrho`; six directions × |q| = 0.02 and 0.04 (2π/a), 10 s each on
8 cores) through the long-wavelength expansion of the induced charge,
ε_L(q̂)[−Ω⟨ñ⟩e^{iq·τ} − i q Z_ion] = −i q·Z* − ½ q·Q·q + O(q³): the imaginary part at order q
is the Born charge, the real part at order q² the quadrupole (Royo and Stengel 2019), with a
Richardson extrapolation in |q|.

| | PAOFLOW (this run) | Poncé et al. 2021, Table IV | Abinit LDA (same ref.) |
|---|---|---|---|
| Q_Si (= Q_{x,yz}, Si at the origin) | +7.49 (7.47 at |q| = 0.02, 7.43 at 0.04) | 7.41 | 7.03 |
| Q_C | −3.01 (−2.99, −2.93) | −2.63 | −1.42 |
| Z* from the linear term vs the Γ run | max deviation 0.05 e (Z* = +2.65 / −2.83) | | |

Only the |ε_{αβγ}| components survive, as the 4̄3m site symmetry requires. The Born charges
recovered from the same expansion check the screening bookkeeping of a polar crystal (a
finite-q DFPT run screens the total charge, and the rigid ion must be added back before the
division by ε_L) and fix the relative sign of the dipole and quadrupole terms, the one the
coupling depends on (EPW's `coeff_delta = facq (i q·Z* + ½ q·Q·q)`). The values are in
`quadrupole.fmt`; `--quadrupoles quadrupole.fmt` adds them to the long-range model of
`coupling.py` and `mobility.py`.

<!-- QUADRUPOLE_COUPLING -->

## 6. Carrier mobility (`mobility.py`)

`mobility.py` computes the state-resolved SERTA and MRTA scattering rates of the carriers within
0.3 eV of the band edges and the intrinsic mobilities. The conduction-band minimum of 3C-SiC lies
at X (DFT gap 1.35 eV, PBE), so the electron pockets are small and the dense grids must be fine:
24³ is far from converged, and the series below is the convergence check of this run. The
"plain interpolation" row omits the Fröhlich term, which is the dominant electron scattering
channel at room temperature.

| dense grids | treatment | μ_e SERTA (cm²/Vs) | μ_e MRTA | μ_h SERTA | μ_h MRTA |
|---|---|---|---|---|---|
| 24³ / 24³ | with the model | 68 | 69 | 89 | 92 |
| 36³ / 36³ | with the model | 1044 | 2867 | 114 | 129 |
| 48³ / 48³ | with the model | 858 | 1046 | 95 | 120 |
| 60³ / 60³ | with the model | 585 | 820 | 79 | 113 |
| 72³ / 72³ | with the model | 416 | 822 | 66 | 108 |
| 84³ / 84³ | with the model | 330 | 791 | 58 | 107 |
| 48³ / 48³ | plain interpolation | 4541 | 4828 | 172 | 170 |

Experiment: μ_e ≈ 900–1000 cm²/Vs, μ_h ≈ 40–50 cm²/Vs at 300 K in 3C-SiC [2]. The Fröhlich term
is the dominant electron scattering channel: without it the electron mobility is five times
larger (4541 against 858 cm²/Vs at 48³). The MRTA, which weights the forward-peaked polar
scattering by the momentum change, lies above the SERTA value, as it does for GaAs in the
literature. The hole mobility is overestimated with PBE masses and without spin-orbit coupling.

Convergence: the MRTA mobilities settle from 60³ on (electrons 820 → 822 → 791 cm²/Vs, holes
113 → 108 → 107 cm²/Vs, within 4 %), and the electron MRTA value of ≈ 800 cm²/Vs is close to the
measured 900–1000 cm²/Vs. The SERTA values keep decreasing with the grid (electrons 585 → 416 →
330 cm²/Vs, holes 79 → 66 → 58 cm²/Vs). A run without the acoustic modes (`eps_acoustic_cm =
400`, same 10 meV smearing) locates the drift: the optical-only electron SERTA mobility is 2060
(36³) and 6090 cm²/Vs (60³), so the acoustic contribution alone, 1/μ − 1/μ_optical, corresponds
to 2120 cm²/Vs at 36³ and 650 cm²/Vs at 60³ and keeps growing with the grid. The acoustic
branches carry a long-range term of their own in the dipole model (the Fröhlich sum over the
atoms does not cancel at order q because of the e^{-iq·τ_κ} phases, a piezoelectric-type
coupling with |g|² ∝ 1/q), and the total scattering rate of such a coupling diverges
logarithmically with the smallest q resolved unless free carriers screen it, while the
momentum-relaxation rate, weighted by 1 − cos θ, converges (Ridley, *Quantum processes in
semiconductors*). Two caveats apply to this acoustic term: the dipole model alone does not give
the physical piezoelectric coupling, which needs the dynamical quadrupoles and the clamped-ion
piezoelectric tensor (Brunin et al. 2020; `quadrupoles.py` of the Si example computes the
quadrupoles from finite-q DFPT and `QUADRUPOLES` adds them to the model), and intrinsic 3C-SiC
has no free carriers to screen it. The SERTA column should therefore be read as unconverged
by construction; the MRTA column is the physical one. The same drift is absent in the
plain-interpolation row, where no 1/q term exists.

Adaptive smearing (`--adaptive-smearing 0.7 --smearing-ev 0.001`): the width of the delta of
every final state m k+q is 0.7 |v_{m k+q}| Δq, floored at 1 meV, instead of the fixed 10 meV
(see "Adaptive smearing of the deltas" in `docs/internals/Electron-Phonon-Coupling.md`). The
grid then converges alone:

| dense grids | μ_e SERTA (cm²/Vs) | μ_e MRTA | μ_h SERTA | μ_h MRTA | widths min / median / max (meV) |
|---|---|---|---|---|---|
| 36³ / 36³ | 275 | 382 | 82 | 110 | 1 / 225 / 452 |
| 48³ / 48³ | 572 | 956 | 94 | 123 | 1 / 169 / 341 |
| 60³ / 60³ | 503 | 993 | 81 | 119 | 1 / 135 / 272 |
| 72³ / 72³ | 409 | 942 | 69 | 115 | 1 / 113 / 228 |

The widths are quoted over all states of the window bands; at the final states of the LO
absorption, 118 meV above the conduction-band minimum, |v| ≈ 4 eV·bohr gives 0.7 |v| Δq ≈ 85,
65, 50 and 45 meV at 36³ to 72³, against a level spacing |v| Δq of 120 to 65 meV, so the delta
is resolved on every grid, whereas the fixed 10 meV delta is well below the level spacing even
at 84³ (55 meV). The adaptive MRTA electron mobility is converged from 48³ on (956, 993,
942 cm²/Vs, within 5 %), where the fixed-width series was still moving at 60³; the hole MRTA
mobility agrees with the fixed-width value (115 against 107 cm²/Vs). The two converged MRTA
electron values, ≈ 950 (adaptive) and ≈ 800 cm²/Vs (fixed 10 meV at 72–84³), bracket the
remaining uncertainty of the delta treatment at these grids: the adaptive widths are still a
third of the LO energy at 72³ and smear the absorption threshold on the scale of k_BT, while the
fixed width is under-resolved. Both lie within the spread of the measured 900–1000 cm²/Vs.
The SERTA column drifts with the grid in both series (572 → 503 → 409 cm²/Vs adaptive), as
expected for the logarithmic divergence of the acoustic long-range term discussed above: a
smearing choice cannot cure it. The 36³ adaptive row is off because its widths (up to 450 meV)
exceed the LO energy and the window.

Temperature dependence at 84³ / 84³ (cm²/Vs):

| T (K) | μ_e SERTA | μ_e MRTA | μ_h SERTA | μ_h MRTA |
|---|---|---|---|---|
| 200 | 515 | 1201 | 95 | 183 |
| 300 | 330 | 791 | 58 | 107 |
| 400 | 219 | 506 | 36 | 63 |

The electron MRTA mobility falls as T^-1.25 between 200 and 400 K; the measured exponent for
3C-SiC is about −1.5 [2].

## 7. Cost

| step | wall time (8 cores, Mac Studio) |
|---|---|
| `pw.x` scf + `ph.x` 6³ with `epsil` | 4 min |
| nscf (16 k, 12 bands) + basis | 10 s |
| `phonons.py` (dispersion, DOS, `sic.fc`, figure) | 5 s |
| `coupling.py` (two 6³/6³ coarse vertices, 24³ dense electrons) | 9 min |
| `mobility.py` 24³, 36³, 48³, 60³, 72³, 84³ (dense k = dense q) | 4, 5, 6, 8, 12, 21 min |
| `mobility.py --adaptive-smearing 0.7` 36³, 48³, 60³, 72³ | 6, 9, 17, 34 min (the last two shared the cores with other jobs) |

[2] Properties of Silicon Carbide (G. L. Harris, ed., INSPEC 1995): 3C-SiC electron mobility
≈ 1000 cm²/Vs and hole mobility ≈ 40 cm²/Vs at 300 K.

[1] Landolt-Börnstein, Group III, Vol. 41A1b (3C-SiC): ε∞ = 6.52, ε0 = 9.72; Feldman et al.,
Phys. Rev. 173, 787 (1968): ω_TO = 796, ω_LO = 972 cm⁻¹.
