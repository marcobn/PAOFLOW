# Gapped graphene: shift current and second-harmonic generation

```bash
python graphene_nonlinear.py      # 90 000 k-points, a few seconds -> output/
```

The honeycomb tight-binding model with a sublattice potential (`delta = 1 eV`, `t = 2.7 eV`) is
the simplest crystal without inversion symmetry: its point group is D3h with `a_1` along `x`.
`PAOFLOW.nonlinear_optics` evaluates, from the real-space Hamiltonian `HRs` of the model and the
orbital centres, the shift current, the injection currents and the SHG susceptibility of the
sheet (`twoD=True`: nm µA/V² and pm²/V, the slab height being the 10 alat vacuum of the model).

What to look at (`output/`):

- `shift_yyy.dat`, `shift_yxx.dat`, `shift_xxy.dat`: the D3h selection rules
  σ^{yyy} = −σ^{yxx} = −σ^{xxy} hold to machine precision and σ^{xxx} = 0 (`shift_xxx.dat`);
  the spectrum starts at the 1 eV gap (K point) with a peak of 14.7 nm µA/V² at 1.07 eV.
- `shgr_yyy.dat`, `shgi_yyy.dat`, `shga_yyy.dat`: Re, Im and |χ^{(2)}| with the two-photon
  resonance at half the gap (0.5 eV) and the one-photon resonance at 1 eV; the same relations
  as the shift current hold between the components.
- `shiftm_*.dat`, `injectm_*.dat` vanish (time-reversal symmetry) and `injectc_*.dat`
  vanishes by the mirror symmetry (no circular photogalvanic effect in a non-gyrotropic group).

`delta = 0` (graphene) restores inversion and every response vanishes. See
`docs/internals/nonlinear-optics.md` for the formulas and conventions.
