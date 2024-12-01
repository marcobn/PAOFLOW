#!/usr/bin/env python
# generate code for spinor spherical harmonics (MBN and DC)

import numpy as np
from fractions import Fraction

lm = [ (0,0), (1,0), (1,1), (1,-1),
       (2,0), (2,1), (2,-1), (2,2), (2,-2),
       (3,0), (3,1), (3,-1), (3,2), (3,-2), (3,3), (3,-3) ]

# construct the coefficient of the spinor spherical harmonics
counter = 0
for l in range(0,4):
    denom = 2*l+1

    # case -
    j = l-1/2
    if j < 0:
        pass
    else:
        mj = -j
        while mj <= j:
            m = mj + 0.5
            if mj <= j:
                fact1 = Fraction(int(l-m+1), denom)
                fact2 = Fraction(int(l+m), denom)
                print('    #l={0:d}, j={1:3.1f} m_j={2:4.1f} upper=sqrt({3})*Y({0:d},{4:d}) \t lower=-sqrt({5})*Y({0:d},{6:d}))'.format(l,j,mj,fact1,int(m-1),fact2,int(m)))
                mj += 1

                print('    ylmgso[:npw,{0}]='.format(counter), end='')
                if fact1 == 0:
                    print('0.0; ', end='')
                else:
                    print('sqrt({0})*ylmgc[:npw,{1}]; '.format(fact1,lm.index((l,m-1))), end='')
                print('ylmgso[npw:,{0}]='.format(counter), end='')
                if fact2 == 0:
                    print('0.0; ', end='')
                else:
                    print('-sqrt({0})*ylmgc[:npw,{1}]; '.format(fact2,lm.index((l,m))), end='')
                print()
                counter += 1
                print()


    # case +
    j = l+1/2
    if j < 0:
        pass
    else:
        mj = -j
        while mj <= j:
            m = mj - 0.5
            if mj <= j:
                fact1 = Fraction(int(l+m+1), denom)
                fact2 = Fraction(int(l-m), denom)
                print('    #l={0:d}, j={1:3.1f} m_j={2:4.1f} upper=sqrt({3})*Y({0:d},{4:d}) \t lower=sqrt({5})*Y({0:d},{6:d})'.format(l,j,mj,fact1,int(m),fact2,int(m+1)))
                mj += 1

                print('    ylmgso[:npw,{0}]='.format(counter), end='')
                if fact1 == 0:
                    print('0.0; ', end='')
                else:
                    print('sqrt({0})*ylmgc[:npw,{1}]; '.format(fact1,lm.index((l,m))), end='')
                print('ylmgso[npw:,{0}]='.format(counter), end='')
                if fact2 == 0:
                    print('0.0; ', end='')
                else:
                    print('sqrt({0})*ylmgc[:npw,{1}]; '.format(fact2,lm.index((l,m+1))), end='')
                print()
                counter += 1
                print()



