#!/usr/bin/env python3
"""Write pb.ukk for epw.x with wannierize = .false. (run in this directory before epw.x).

EPW reads the band bookkeeping from this file only: the 16 nscf bands, of which
the five 5d semicore bands (1-5) are excluded from the electron-phonon
calculation, matching ``bands_skipped`` in epw.in.  No Wannier functions are
built; the identity rotations only feed EPW's Wannier stage, which PAOFLOW does
not use.
"""

from PAOFLOW.gen.epw_inputs import write_placeholder_ukk

write_placeholder_ukk('pb.ukk', nbnd=16, nk_total=6**3, exclude_bands=[1, 2, 3, 4, 5], nelec=14)
print('Wrote pb.ukk')
