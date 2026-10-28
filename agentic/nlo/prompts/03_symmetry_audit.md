# 03: Symmetry audit of the run

```
For the point group <name, axis convention> and the output <folder>: list the forbidden components
(symmetry.forbidden_components), the relations between the allowed ones (symmetry.independent_relations), and for every
kind present (shga/shgr/shgi, shift, shiftm, injectc, injectm) at the photon energies <0, E_g/2, E_g, peak, ...>:
the leakage of the forbidden components present, the spread between equivalent components, the Kleinman error of the
SHG tensor at the lowest energy, and whether the T-odd kinds vanish (T-symmetric crystal). Write tables/symmetry.csv
and quote the largest violation as the symmetry error bar of the run. Say whether the nscf used nosym and what the
degeneracy tolerance was.
```

**Check:** one table with numbers per kind and energy; the error bar is stated and used in every later comparison.
