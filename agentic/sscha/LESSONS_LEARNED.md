# Lessons learned (δ-TaN campaign and the implementation tests, November 2026)

Read together with `AGENT_WORKFLOW.md`. Each entry: **symptom → cause → what to do**, with the instance.

## Physics

| # | Symptom | Cause | What to do | Instance |
|---|---|---|---|---|
| 1 | All frequencies positive after the minimisation, yet the lattice is "unstable" in the literature | The auxiliary (SSCHA) matrices define a Gaussian and are positive by construction | Judge stability only on the free-energy Hessian (`sscha_hessian`, `SSCHARun.hessian()`) | δ-TaN 2800 K: auxiliary X 100.3 cm⁻¹; the Hessian is the physical result. |
| 2 | Stable on the supercell grid, unstable elsewhere | The supercell grid samples few q (2×2×2 fcc: Γ, X, L); the correction is exact only there | Check the exported Hessian matrices on a fine mesh (`grid_minimum`, `diagnostics(fine_mesh=...)`); enlarge the supercell if an instability lies between the grid points | δ-TaN: the 2×2×2 grid contains X, the deepest harmonic instability; the soft branch away from X (X–W–K) is checked on the 6³ harmonic grid. |
| 3 | The minimisation cannot start from the harmonic matrices | Imaginary harmonic frequencies have no Gaussian | `positive_definite=True` (default) makes them positive; or start from a converged neighbouring temperature (`start_from`) | δ-TaN harmonic X −223 cm⁻¹ → start 85.5 cm⁻¹; converged in 4 populations. |
| 4 | Hot electrons alone do not stabilise a metal, but change the instability | Fermi–Dirac occupations at k_B T smooth the Fermi-surface nesting that drives the soft mode | For metals at high T use `electronic_temperature=True` in the force runs; keep the comparison "cold vs hot electrons" explicit | δ-TaN X: −264 cm⁻¹ (cold) → −223 cm⁻¹ (Fermi–Dirac 2800 K), stabilised only by the SSCHA. |
| 5 | A warm-started run converges in its first population | Starting from the matrices of a nearby temperature puts the Gaussian close to the minimum | Not an error: check the error bars and rely on the Hessian population | δ-TaN 2000 K from the 2800 K matrices: converged after population 1 (auxiliary X 76.6 cm⁻¹). |
| 6 | T_s disagrees with a measured transition temperature | The SSCHA gives where the harmonic instability is lifted at fixed volume, with the chosen functional and electrons; experiments see competing phases, composition, thermal expansion | Compare as a bound or a range and state the differences; quote the bracket, not a single number | δ-TaN vs the eutectoid of the Ta–N phase diagram. |
| 7 | ω² vs T is not a straight line over the whole scan | The mean-field form ω² ∝ (T − T_s) holds only near T_s; far from it the soft mode saturates | `stability_temperature` interpolates in the bracketing pair and fits locally (the pair and one neighbour each side) | Synthetic and δ-TaN scans. |

## Running

| # | Symptom | Cause | What to do | Instance |
|---|---|---|---|---|
| 8 | A monitor reported a stopped run that was fine | Log greps match unrelated text ("stop" in "Check the stopping criteria") | Read the state from files: `main.sscha.py status`, `SSCHARun.status()` | δ-TaN monitors, November 6 2026. |
| 9 | The ensemble read a half-written output | A killed pw.x leaves an output without forces | Only outputs with `JOB DONE` and the total force are read; a population is minimised when 95 % finished; `run` resubmits the rest | Built into `completed_outputs` / `missing_configurations`. |
| 10 | Scratch fills up | Each configuration writes wavefunctions to its own `tmp_<prefix>_XXXX` | The drivers delete it after each run (local) or at the end of each array task (Slurm) | 800-configuration Hessian populations. |
| 11 | Results lost on a cluster | Scratch is purged (14 days on Snellius) | Copy `state.json`, `pop*/summary.json`, `pop*/dyn*`, `hessian/`, `force_constants_*/`, `*.fc` to permanent storage after each temperature | Snellius scratch-shared. |
| 12 | `sscha_setup` refuses a directory | It holds a run at another temperature, supercell or electronic temperature | One directory per run (`output/sscha_T<T>/`); never reuse one for different settings | Guard added November 7 2026. |
| 13 | `no Gamma dynamical matrix` or too few q | The ph.x grid does not contain the supercell q (or the prefix is not `<prefix>.dyn`) | Run ph.x on a multiple of the supercell (`main.sscha.py harmonic`) | — |
| 14 | A population has one configuration more than asked | python-sscha draws ± pairs | Sizes are made even | — |
| 15 | python-sscha does not build | Fortran extensions: missing BLAS/LAPACK `.pc` files; on Apple silicon an x86 `pkg-config`/`cmake` in `/usr/local/bin` breaks meson | Follow `docs/internals/sscha.md`, "Installation" | Mac Studio and Snellius builds, November 2026. |
