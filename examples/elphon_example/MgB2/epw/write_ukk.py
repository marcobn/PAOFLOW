#!/usr/bin/env python3
"""Write mgb2.ukk for epw.x with wannierize = .false. (run in this directory before epw.x).

EPW reads the band bookkeeping from this file only: the 24 nscf bands, all of them kept in the
electron-phonon calculation.
No Wannier functions are built; the identity rotations only feed
EPW's Wannier stage, which PAOFLOW does not use.  The empty mgb2.bvec /
mgb2.mmn written alongside let that stage finish.
"""

from PAOFLOW.gen.epw_inputs import write_placeholder_ukk

write_placeholder_ukk('mgb2.ukk', nbnd=24, nk_total=6 * 6 * 6, exclude_bands=[], nelec=16)
print('Wrote mgb2.ukk (and the empty wannier90 stubs mgb2.bvec, mgb2.mmn)')
