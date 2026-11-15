#!/usr/bin/env python3
"""Write Si2.ukk for epw.x with wannierize = .false. (run in this directory before epw.x).

EPW reads the band bookkeeping from this file only: the 21 nscf bands, all of them kept in the
electron-phonon calculation.
No Wannier functions are built; the identity rotations only feed
EPW's Wannier stage, which PAOFLOW does not use.  The empty Si2.bvec /
Si2.mmn written alongside let that stage finish.
"""

from PAOFLOW.gen.epw_inputs import write_placeholder_ukk

write_placeholder_ukk('Si2.ukk', nbnd=21, nk_total=6 * 6 * 6, exclude_bands=[], nelec=8)
print('Wrote Si2.ukk (and the empty wannier90 stubs Si2.bvec, Si2.mmn)')
