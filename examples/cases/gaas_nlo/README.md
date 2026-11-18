# GaAs: non-linear optics (shift current, injection currents, SHG)

Bulk GaAs (F-43m, PAW pseudopotentials) on a 62-orbital PAO basis: the second-harmonic
susceptibility χ_xyz(ω) and the shift and injection currents of a cubic crystal with a scissor,
from `PAOFLOW.nonlinear_optics`. The folder also holds the validation report of the non-linear
optics (`REPORT.md`: gapped graphene, GaAs and monolayer GeS against the literature), the script
that draws its figures (`make_report_figures.py`) and the digitised literature spectra of GaAs
(`data/`).

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name. Studio copies
  the inputs and the pseudopotentials and lists the runs below in the Runs tab.
- **Command line**: `paoflow-gen --example gaas_nlo` copies the case into `./gaas_nlo`; then
  run the steps below in that folder.

## Steps

1. Self-consistent run: `pw.x -in scf.in`
2. Non-self-consistent run on the 16³ grid (symmetry-reduced, 145 k-points, nbnd 50):
   `pw.x -in nscf.in`. The inputs are those of the case `gaas_doped_transport`: its `GaAs.save`
   can be linked here instead (`ln -s ../gaas_doped_transport/GaAs.save .`).
3. PAOFLOW: `mpirun -np 8 python main.py` (40³ k-points: about 2.5 min on 8 ranks); writes
   `output/{shift,shiftm,injectc,injectm}_<abc>.dat` and `shg{r,i,a}_<abc>.dat` (photon energy
   in eV; µA/V², A V⁻² s⁻¹ and pm/V).
4. Figures of the report: `python make_report_figures.py` writes `figures/fig2_gaas.{pdf,png}`
   from `output/`, and Fig. 1 and Fig. 3 from the `output/` folders of the cases
   `graphene_nlo_model` and `ges_nlo` when they sit next to this one (as in `examples/cases`;
   a missing folder skips its figure).

No projwfc is needed: the PAO projections are computed internally from the wavefunctions.
`main.py` builds the PAO Hamiltonian on 62 orbitals: the valence shells from the UPF
pseudo-orbitals, the augmenting shells from the all-electron tables of the `BASIS` folder of the
PAOFLOW checkout (`BASISPATH` in `main.py`; the default for PAW inputs since October 2026; 23
bands above 0.95), then calls
`PAOFLOW.nonlinear_optics(nk=(40, 40, 40), scissor=0.86, eta=0.05, components=['xyz', ...])`.

## What to look at

Targets (see `docs/internals/nonlinear-optics.md`): |χ_xyz(ω)| of Wang et al. 2017 Fig. 4
(scissors curve), χ_xyz(0) ≈ 1.5–2.5 × 10² pm/V (Rashkeev 1998, Nastos 2005), and the shift
current σ_xyz of Ibañez-Azpiroz et al. 2018. χ_xyz = χ_yzx = χ_zxy by symmetry at every
frequency, χ_xxx = 0; the shift and injection currents of a cubic T_d crystal vanish for the
diagonal components and the T-odd files (`shiftm`, `injectm`) vanish. Converge `NK` (60³ → 100³)
and `ETA` before quoting numbers.

## Reference runs

October 2026, scissor 0.86 eV, eta 0.05 eV, 62-orbital basis; 15 min at 40³ and 48 min at 60³ on
8 ranks: |χ_xyz| = 110 pm/V at 0 eV (40³ and 60³), 341/337 pm/V at 1.5 eV, peak 610 (40³) / 480
(60³) pm/V near 2.6 eV, the resonance region being unconverged (literature with a scissor:
χ_xyz(0) ≈ 100–170 pm/V, peaks of 400–600 pm/V); the shift current σ_xyz reaches −30 / −27 µA/V²
at 3.9–4 eV. χ_yzx agreed with χ_xyz within 5 % (40³) / 2–4 % (60³) and the forbidden χ_xxx was
5–11 % of the peak at the resonance in the first runs: the internal projections of PAW
pseudo-wavefunctions broke the point symmetry at a few k-points (a nosym nscf does not help).

Fixed in October 2026 (augmented metric of the projections, see `REPORT.md` section 3.2): the
Hamiltonian is now symmetric (interpolation spread 3e-3 meV between equivalent k-points instead
of 5–25 meV), the shift current obeys the T_d relations to 1e-10; the projectability vector
changed (12 instead of 18 bands above 0.95; use pthr = 0.90 to keep 21).

Reference runs with the current code (PAW, augmented projections, valence shells from the UPF, 23
bands above 0.95, 40³ reduced to 3091 k-points by symmetry: 135 s on 8 ranks): |χ_xyz| = 98 pm/V
at 0 eV, 318 at 1.5 eV, peak 695 pm/V at 2.63 eV, σ_xyz = −29 µA/V² at 3.9 eV; with
`position_dipoles=True` (the PAW augmentation of the position operator included, 4 min): 193
pm/V, 636, 874, −42 µA/V² (158, 559, 850, −38 without the augmentation). Every T_d relation holds
to 1e-8 of the peak (covariant kernels and symmetrised tensors, October 28 2026).

With the PseudoDojo norm-conserving Ga/As set (`PSEUDOS/nc-sr-05_pbe_standard_upf` of the
repository, 90 Ry, As 3d in valence; nelec 28) the same basis gives |χ_xyz(0)| = 103 pm/V, peak
492 pm/V at 2.54 eV, σ_xyz −28 µA/V² at 3.9 eV at 40³, and 178 pm/V, 677 pm/V, −41 µA/V² with
the position dipoles: the dipoles bring the static χ_xyz into the measured 166–180 pm/V
(`REPORT.md` section 3.2). The remaining 2–10 % difference between χ_xyz and χ_yzx (SHG only)
is the gauge of exactly degenerate bands on the Λ lines of the grid (the U(N)-covariant
treatment of the phase-2 plan, `docs/internals/nonlinear-optics-phase2-plan.md`).

## Report and data

- `REPORT.md`: validation report of `PAOFLOW.nonlinear_optics` on the three systems (cases
  `graphene_nlo_model`, `gaas_nlo`, `ges_nlo`), with the comparison with experiments and
  published calculations; its figures are in `figures/` (Fig. 1–3 redrawn by
  `make_report_figures.py`; Fig. 4 was drawn from `data/` and the validation runs by the agentic
  project `agentic/nlo/projects/GaAs_SHG_BergfeldDaum2003`).
- `data/`: the measured |χ_xyz| of Bergfeld & Daum 2003 and the Hughes–Sipe 1996 and
  Adolph–Bechstedt 1998 spectra, digitised from the published figures (provenance in the files).

## The non-linear optics cases

Shift current, injection currents and second-harmonic generation from the PAO Hamiltonian
(`PAOFLOW.nonlinear_optics`; theory, conventions and validation in
`docs/internals/nonlinear-optics.md`):

| case | system | what it shows | cost |
|---|---|---|---|
| `graphene_nlo_model` | gapped graphene, tight-binding model (`PAOFLOW(model=...)`) | the D3h selection rules of a 2D crystal to machine precision; 2D sheet units | seconds |
| `gaas_nlo` | GaAs (PAW, 16³ nscf), 62-orbital internal basis | χ_xyz(ω) and the shift current of a cubic crystal with a scissor; the symmetry error bar of a DFT Hamiltonian | 2.5 min at 40³ on 8 ranks |
| `ges_nlo` | monolayer GeS (Pmn2₁), relaxed with PBE, nosym 18×18×1 nscf, basis from the pseudopotentials | 2D polar crystal: allowed and forbidden components, sheet responses in nm µA/V² and pm²/V | 13 min at 200² |
| `mos2_nlo_model`, `kane_mele_nlo_model` | three-band MoS2 model, Kane-Mele model | D3h responses with spin-orbit coupling; the shift current across a topological transition | seconds |

Workflow:

1. QE: `scf.in`, then `nscf.in` on a uniform grid (symmetry-reduced or `nosym`/`noinv`: the
   symmetry expansion is exact). Ultrasoft / PAW pseudopotentials are projected in the augmented
   metric (since October 2026; before that the internal projections broke the point symmetry at a
   few k-points, `REPORT.md` §3.2): check the projectability vector, and prefer norm-conserving
   pseudopotentials when the basis projects the semicore bands poorly.
2. Basis: the repository `BASIS/` tables, or tables generated from the pseudopotentials with
   `paoflow-genbasis-ps` when the projectability of the valence bands is below 0.95 (GeS). With
   ultrasoft / PAW pseudopotentials the valence shells are taken from the UPF pseudo-orbitals
   automatically (`projections(pseudo_orbitals='auto')`) and only the augmenting shells from the
   tables (GaAs: 23 bands above 0.95).
3. PAOFLOW: `projections(internal=True, ...)`, `projectability()`, `pao_hamiltonian()`,
   `nonlinear_optics(nk=..., emax, eta, scissor, components, twoD)`, `finish_execution()`.
   Call `nonlinear_optics` before `interpolated_hamiltonian` / `pao_eigh` (they release `HRs`).
4. Plot: `GPAO.plot_optical(['shift', 'shga', ...])` or `make_report_figures.py` here.

`paoflow-gen` offers the same run as the property `nonlinear` (it writes `run_nonlinear()` with
`NLO_NK`, `NLO_EMAX`, `NLO_ETA`, `NLO_SCISSOR`, `NLO_COMPONENTS` and the plot entries), and the
agentic workflow `agentic/nlo/` compares the outputs with experiments and published calculations.

Outputs: `shift_<abc>.dat`, `shiftm_<abc>.dat` (µA/V²), `injectc_<abc>.dat`, `injectm_<abc>.dat`
(A V⁻² s⁻¹), `shgr_/shgi_/shga_<abc>.dat` (pm/V); with `twoD=True` the sheet responses
(nm µA/V², nm A V⁻² s⁻¹, pm²/V).

Pseudopotentials (from `PSEUDOS/library`): `Ga.pbe-dn-kjpaw_psl.1.0.0.UPF`,
`As.pbe-n-kjpaw_psl.1.0.0.UPF`.
