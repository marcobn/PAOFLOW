# Non-linear optics examples

Shift current, injection currents and second-harmonic generation from the PAO Hamiltonian
(`PAOFLOW.nonlinear_optics`; theory, conventions and validation in
[docs/internals/nonlinear-optics.md](../../docs/internals/nonlinear-optics.md)). The results of
the three systems, their figures and the comparison with the literature are in
[REPORT.md](REPORT.md).

| folder | system | what it shows | cost |
|---|---|---|---|
| `Graphene/` | gapped graphene, tight-binding model (`PAOFLOW(model=...)`) | the D3h selection rules of a 2D crystal to machine precision; 2D sheet units | seconds |
| `GaAs/` | GaAs, QE example10 inputs (PAW, 16³ nscf), 62-orbital internal basis | χ_xyz(ω) and the shift current of a cubic crystal with a scissor; the symmetry error bar of a DFT Hamiltonian | 15 min at 40³ on 8 ranks |
| `GeS/` | monolayer GeS (Pmn2₁), relaxed with PBE, nosym 18×18×1 nscf, basis from the pseudopotentials | 2D polar crystal: allowed and forbidden components, sheet responses in nm µA/V² and pm²/V | 13 min at 200² |

## Workflow

1. QE: `scf.in`, then `nscf.in` on a uniform grid (symmetry-reduced or `nosym`/`noinv`: the
   symmetry expansion is exact). Ultrasoft / PAW pseudopotentials are projected in the augmented metric
   (since October 2026; before that the internal projections broke the point symmetry at a few
   k-points, [REPORT.md](REPORT.md) §3.2): check the projectability vector, and prefer
   norm-conserving pseudopotentials when the basis projects the semicore bands poorly.
2. Basis: the repository `BASIS/` tables, or tables generated from the pseudopotentials with
   `paoflow-genbasis-ps` when the projectability of the valence bands is below 0.95 (GeS).
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
