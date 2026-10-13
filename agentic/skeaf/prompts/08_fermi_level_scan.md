# 08: Fermi-level sensitivity

```
<Either> Propose a Fermi-level scan for <bands> over <angular window>: values of fermi_energy (eV!), num_interpolation,
angles, expected runtime; the output must go to a NEW folder (`skeaf_runs.skeaf_rotation` enforces it). <Or> Shifted runs are in <path> (tokens <-0.005, 0, 0.005>).
Call `fermi_level_scan(<path>, bands=[<b>], tokens=(<lo>, '0', <hi>), energy_step_meV=<meV>)`: report measured/Onsager
dF/dE (median, IQR), the typical shift per meV, and which orbits appear/disappear and where. Then judge any
rigid shift against EVERY peak and every direction; reject shifts that break good matches or differ between directions.
```
