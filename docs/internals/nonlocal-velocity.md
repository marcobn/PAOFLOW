# Non-local velocity correction (removed, October 2026)

Until October 2026 PAOFLOW offered a correction Δp = m/(iħ)[V_NL, r] to the velocity operator,
assembled from the Kleinman–Bylander projectors of the UPF files by two-centre integrals:
`gradient_and_momenta(nonlocal_velocity=True)` / `attr['nonlocal_velocity']` on the `dHksp`
route, `nonlocal_velocity=<DataController>` (`NONLOCAL_VELOCITY`) in the electron–phonon
drivers. The term was removed; its last version is in commit `c564f569`.

## Why

PAOFLOW's velocity is the commutator v = i[H, r] of the PAO Hamiltonian, and that Hamiltonian
already contains the non-local pseudopotential (it is the full Kohn–Sham Hamiltonian projected on
the PAO orbitals). Its only approximation is the position operator: with r reduced to the orbital
centres (`dHksp` plus the intersite term) the missing piece is i[H, d] with the position dipoles
d of the Löwdin basis (`position-dipoles.md`), not i[V_NL, r]. Adding Δp double counts V_NL.

Measured against the exact plane-wave velocity of the Kohn–Sham states of Si
(`EPW_opt_audit/pao_dielectric/pw_dielectric_reference.py`: 12³ scf grid, 21 bands, extended
44-orbital basis, ∫ε₂ over 2.5–5 / 2.5–10 eV relative to exact):

| velocity | ∫ε₂ / exact | ε₁(0) (exact 13.62) |
|---|---|---|
| plain ∂H/∂k + intersite term | 1.04 / 1.02 | 14.42 |
| + Δp as it was | 1.09 / 1.09 | 15.08 |
| + Δp repaired (see below) | 0.93 / 0.90 | 13.23 |
| + position dipoles | 0.99 / 1.01 | 13.60 |
| + position dipoles + Δp repaired | 0.87 / 0.87 | 11.95 |

On bcc Fe and fcc Pt (`Workspace/PAOFLOW/dipole_hall/`) the repaired Δp moved the AHC / SHC at
E_F by +10 % / −7 % and doubled the Fe peak at −1.35 eV (the double counting amplified by the
1/Δε² of the Kubo formula at the d-band near-degeneracies); the dipoles move them by −1 % /
+3.5 % with unchanged curves.

The term as implemented also had two defects, found on 2026-10-21 by comparing it with the
plane-wave commutator between PAOFLOW's own orbitals (`EPW_opt_audit/compare/nl_orbital_check.py`,
`nl_fixed_check.py`, both historical now): the raw-orbital matrix was used in the Löwdin basis
without the transformation S^{-1/2} Δp S^{-1/2} (largest elements 4–5× too small), and it carried
the odd-|m| sign only, without the (−1)^l of PAOFLOW's orbitals (every s–p and p–d element
flipped). Repaired, it matched the exact commutator to 2–3 %, which is what made the double
counting measurable. The earlier validations (Cu ε₂ peak and Pt against QE `epsilon.x`) were
obtained with the defective term; `epsilon.x` itself uses the local momentum only and lies
15–19 % above the exact Si spectrum, so it is not a reference for the velocity operator.

## What remains

- `hamiltonian/nonlocal_velocity.py` keeps the PAO orbital catalog used by the position dipoles:
  `load_pao_orbitals`, `PAOCatalog` and its data classes, the cutoff helpers, `qe_m_index_to_std`,
  `pao_orbital_signs`, `lowdin_transform`, `scalar_to_jm_matrix`. The projector loader, the
  β–PAO tables, the k-space assembly, `inject_into_dHksp` and the (j, m_j) rotation of Δp are gone,
  together with their unit tests.
- `gradient_and_momenta(nonlocal_velocity=True)`, `attr['nonlocal_velocity']` and the drivers'
  `nonlocal_velocity=` raise `ValueError` pointing to the replacement.
- Replacement: `gradient_and_momenta(position_dipoles=True)` (`PAOFLOW.position_dipole_correction`,
  scalar, ad-hoc spin-orbit and (j, m_j) bases) and `position_dipoles=data_controller` in the
  electron–phonon drivers (`POSITION_DIPOLES = True`, the default of `paoflow-gen`).
- `do_adaptive_smearing` keeps using `velkp_bare`, the Hellmann–Feynman diagonal stored before
  the dipole term is folded into `dHksp`.
