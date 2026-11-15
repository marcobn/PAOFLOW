# 04: Self-energy and λ extracted the same way on both sides

**When:** the comparison involves Σ(ω) or λ.

```
Define the bare-band protocol once (<the authors' choice: linear fit over <window>, parabola, DFT band>) and apply it to
both sides: on the measured dispersion (cv.linear_band over the window) and on the model's dressed dispersion
(ArpesModel.dispersion then cv.linear_band with the same window). Then extract Re Sigma = w - eps_b(k(w)) and
Im Sigma = -dk |v_b| / 2 on both (cv.self_energy_from_dispersion), subtract and state the impurity offset Im Sigma(0),
and tabulate lambda by every method available: slope of Re Sigma at E_F (mass_enhancement), velocity ratio
(ArpesModel.velocities with the same window), the authors' value, and the model's own Sigma(w) from
sigma_<kind>_<T>K.dat. Add the baseline Sigma from a tunnelling alpha^2F if one exists
(PAOFLOW.elphon.self_energy.eliashberg_self_energy at the measured T). Write the curves to data/model_curves.csv and
the lambda table to tables/lambda.csv.
```

**Check:** one protocol, applied to both sides; λ quoted by method; the model's exact bare band shown as extra information, not as the comparison.
