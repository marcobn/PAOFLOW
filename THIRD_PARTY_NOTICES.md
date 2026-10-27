# Third-party notices

## chinook (MIT)

`src/PAOFLOW/arpes/` follows the photoemission formalism of **chinook**
(R. P. Day, B. Zwartsenberg, I. S. Elfimov and A. Damascelli, *Computational framework chinook
for angle-resolved photoemission spectroscopy*, npj Quantum Materials **4**, 54 (2019);
https://github.com/qmlab-ubc-ca/chinook, commit 24913de, February 2026).  The following parts
are adapted from chinook's code:

- `arpes/kinematics.py`: `kz_free_electron` (`klib.kz_kpt`), `momentum_mesh` (`klib.kmesh`),
  `angle_mesh` (`tilt.ang_mesh`, `tilt.k_mesh`), `final_state_directions` (`ARPES_lib.datacube`);
- `arpes/atomic_radial.py`: Slater's screening rules (`electron_configs.Z_eff`), the Slater and
  hydrogenic radial functions (`Slater_exec`, `hydrogenic_exec`) and the electron-configuration
  table `arpes/electron_configs.txt`;
- `arpes/spectral.py`: the Kramers–Kronig construction of the self-energy (`ARPES_lib.gen_SE_KK`)
  and the resolution broadening of `ARPES_lib.experiment.spectral`.

The matrix elements, radial integrals and spectral functions are re-implemented in the
real-harmonic PAO basis of PAOFLOW.  chinook's licence:

```
MIT License

Copyright (c) 2018 Ryan Patrick Day

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
