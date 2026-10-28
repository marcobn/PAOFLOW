# *************************************************************************************
# *   PAOFLOW *  Marco BUONGIORNO NARDELLI * University of North Texas 2016-2026      *
# *                                                                                   *
# *************************************************************************************
#
#  Copyright 2016-2026 - Marco BUONGIORNO NARDELLI (mbn@unt.edu) - AFLOW.ORG consortium
#
#  This file is part of AFLOW software.
#
#  AFLOW is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# *************************************************************************************

from PAOFLOW import PAOFLOW


def main():
    """Shift current, injection currents and SHG of gapped graphene (reduced grid of the example)."""
    model = {'label': 'graphene', 't': 2.7, 'delta': 1.0}
    paoflow = PAOFLOW.PAOFLOW(model=model, outputdir='./graphene_nonlinear', verbose=True)
    paoflow.nonlinear_optics(
        nk=(60, 60, 1),
        emin=0.0,
        emax=4.0,
        ne=201,
        eta=0.1,
        components=['yyy', 'yxx', 'xxx'],
        nelec=2,
        twoD=True,
    )
    paoflow.finish_execution()


if __name__ == '__main__':
    main()
