# Position dipoles of the PAO basis in the velocity operator

## The problem

PAOFLOW's velocity operator in the orthonormal PAO Bloch basis is the k-derivative of the
Hamiltonian plus the inter-site term, `v = dH/dk + i[H, τ]`
(`elphon/elph_bloch.band_velocities`, the `Dnm` term of `hamiltonian/do_gradient`): the position
operator is taken diagonal on the orbital centres τ_i. For orthonormal localised orbitals ω the
exact tight-binding velocity has a third term,

    v(k) = dH/dk + i[H, τ] + i[H, d(k)],
    d_ij(R) = <ω_i(r − τ_i) | r − R − τ_j | ω_j(r − R − τ_j)>,

whose band-basis form `i(ε_n − ε_m) d_nm` is the analogue of EPW's Berry-connection term in
`vme = 'wannier'`. Its largest part is the on-site Δl = ±1 dipole (⟨s|r|p⟩, ⟨p|r|d⟩) of the same
atom. It vanishes on the diagonal and inside degenerate blocks, so group velocities (SERTA/MRTA,
adaptive smearing, Drude) are untouched.

## The implementation (`hamiltonian/position_dipoles.py`)

PAOFLOW's basis functions are the atomic orbitals φ symmetrically (Löwdin) orthogonalised,
ω = φ S^{-1/2}. With the two-centre tables of the atomic orbitals — the overlap S(R) and the
intrinsic dipole X(R) with the origin on the bra site (`_two_center`, the PAO radial function as
bra with g = rR at l ± 1, as the projector bra of the non-local velocity) — the Bloch-space dipole
of the orthonormal basis is

    d(k) = S^{-1/2} X S^{-1/2} + S^{-1/2} τ S^{1/2} − τ + i S^{-1/2} ∂_k S^{1/2}

(all at k; S(k) = Σ_R e^{ik·R} S(R), likewise X; ∂_k S^{1/2} from the eigendecomposition of S).
The three extra terms come from the cell-position operator, which does not commute with
S^{±1/2}, and from the bra-site origin of X; d(k) is Hermitian.

| function | role |
|---|---|
| `enumerate_pao_pairs` | (bra site, ket site, ΔR) within the orbital cutoffs |
| `build_position_tables` | S and X blocks per pair (cached radial Bessel transforms) |
| `assemble_position_kspace` | S(k), ∂_k S(k), X(k) by Bloch sums e^{ik·ΔR} |
| `lowdin_dipole_kspace` | the formula above; `level='intra-atomic'` keeps the on-site blocks with S ≡ 1 |
| `dipole_velocity_term` | i[H(k), d(k)] in the orbital basis |
| `position_dipole_operator` | data-controller-facing callable `K_cryst -> d(k)`, the `position_dipoles` input of `band_velocities` |

Hooks: `band_velocities(..., position_dipoles=)`,
`phonon_assisted_absorption_dense_q(..., position_dipoles=data_controller, position_dipole_level='full')`,
the Si example flag `POSITION_DIPOLES` (`--position-dipoles`, `--position-dipole-level`).
Off by default; `carrier_scattering` is unaffected (diagonal velocities). Scalar-relativistic
bases only (`NotImplementedError` for `dftSO`).

## Conventions (measured, not assumed)

PAOFLOW's atomic Bloch orbitals (`calc_atwfc_k`) are built in reciprocal space with the phase
i^l and the `calc_ylmg` harmonics, which carry an extra −1 on odd |m|. Transformed back to real
space they are (−1)^l (−1)^{|m| mod 2} times R(r) Y_lm(r̂) with the tesseral harmonics of
`_two_center`. `tesseral_signs` applies both factors on both orbital indices. Verified on Si
(3S 3P 3D 4S 4P, 26 orbitals): the two-centre S(k) reproduces the plane-wave overlap of the
PAOFLOW orbitals to 1e-3 (raw values up to 8.9 at Γ for the diffuse 4S/4P/3D Bloch sums), and
the Löwdin dipoles agree with the exact Berry connection i⟨u_i|∂_k u_j⟩ of the Löwdin
plane-wave orbitals to 2 % (max |d| 3.9 bohr). The odd-m rule alone (the `pao_sign` of
`nonlocal_velocity.py`) flips every s–p / p–d element.

The on-site blocks alone (`'intra-atomic'`) differ from the exact d by up to 2.7 bohr for this
basis: with diffuse shells the Löwdin terms are the bulk of d, not a correction.

## Validation (EPW tutorial-06 Si data, `EPW_opt_audit/compare/pw_velocity_check.py`, `dipole_pw_diagnostic.py`)

At the coarse nscf k-points the exact plane-wave velocity of the Kohn–Sham states,
`v = 2⟨n|k+G|m⟩ + ⟨n|i[V_NL, r]|m⟩` (Ry bohr; KB commutator from the k-derivative of the
projector form factors, checked by Hellmann–Feynman against the PAO group velocities), is
compared with the PAO velocity in gauge-invariant block sums of |v|²:

| pairs | plain PAO / exact | with d / exact |
|---|---|---|
| valence → CB1–CB4 | 1.4–2.3 | 0.96–1.13 |
| valence top → CB1 | 0.9–1.6 | 0.9–1.1 (1.45 at Γ) |
| CB1 → CB2 at X | 4 | 0.5 |

Direct ε₂ of Si at 12³ k against EPW (`vme='wannier'`, which contains the Berry term): plain
1.06, with d 0.98 (ω-weighted integral 2.5–5 eV). The CB1 → CB2 transition at the conduction
minimum is symmetry-forbidden on the valley axis in both the PAO and the plane-wave velocity;
off-axis it has the plane-wave magnitude already without d.

## The `dHksp` route (`gradient_and_momenta(position_dipoles=True)`)

`PAOFLOW.position_dipole_correction()` builds d(k) on the FFT k-grid
(`position_dipoles_on_grid`: scalar, ad-hoc spin-orbit and fully relativistic (j, m_j) bases,
the latter through `scalar_to_jm_matrix` with the per-channel (−1)^l) and adds
−i[H(k), d(k)] to `dHksp` (`inject_dipole_velocity_into_dHksp`; H(k) = V E V† from `pao_eigh`,
since `do_gradient` overwrites `Hksp`). Every property built on `pksp` then carries the full
commutator: `dielectric_tensor`, EELS, optical conductivity, anomalous and spin Hall, the ac
Hall conductivities, Rashba–Edelstein. Checks: on Si the pipeline reproduces the
electron–phonon implementation (∫ε₂ ×0.957 / ×0.993 over 2.5–5 / 2.5–10 eV against ×0.953 /
×0.995) and ε₁(0) 13.54 against the exact 13.62; Fe AHC and Pt SHC (12³ scf, 32³ interpolation,
`Workspace/PAOFLOW/dipole_hall/`): σ_xy(E_F) 494 → 490 S/cm and σ^z_xy(E_F) 2045 → 2117
(ħ/e) S/cm, i.e. −1 % and +3.5 %, with the curves unchanged in shape — the few-per-cent effect
expected from the Kubo structure (the dipole cross term carries one power of 1/Δε, the main
term two). The repaired non-local term on the same runs moves them by +10 % / −7 % at E_F
and doubles the Fe peak at −1.35 eV (1700 → 2920 S/cm), the signature of the double counting
amplified by 1/Δε² at the d-band near-degeneracies.

QE `epsilon.x` on the same system (full 12³ grid, `nosym`/`noinv`) reproduces the plane-wave
local-only kernel to 2 % (∫ε₂ 2.5–10 eV: 91.1 vs 92.9; exact 78.4): it is the local-momentum
reference and lies 15–19 % above the exact spectrum, which is why agreement with `epsilon.x` is
not a validation of a velocity operator.

## Open points
- The non-local correction Δp of `nonlocal_velocity.py` was fixed on Oct 21 2026 with the
  overlap tables and the signs of this module (`nonlocal-velocity.md`). It is not to be combined
  with the dipoles: v = i[H, τ + d] is already the full commutator of a Hamiltonian that
  contains V_NL, and adding Δp double counts (exact plane-wave ε₂ of Si: dipoles 0.99–1.01,
  dipoles + Δp 0.87). Keep `nonlocal_velocity=False` with `position_dipoles` on.
- The Hubbard term of eACBN0 electrons is inside H(R), so i[H, τ + d] contains its velocity
  commutator; nothing else is needed.
