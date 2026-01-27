!
! Copyright (C) 2005 WanT Group, 2017 ERMES Group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
! Adapted from Numerical Recipies
!
!=-----------------------------------------------------=
   SUBROUTINE locate( xx, n, x, j )
!=-----------------------------------------------------=
   USE kinds, ONLY : dbl
   IMPLICIT NONE

   INTEGER,   INTENT(in)  :: n
   REAL(dbl), INTENT(in)  :: x,xx(n)
   INTEGER,   INTENT(out) :: j
   !
   INTEGER   :: jl,jm,ju

   jl=0
   ju=n+1
   !
10 if(ju-jl > 1) then
       !
       jm=(ju+jl)/2
       ! 
       if( (xx(n) > xx(1)) .eqv. (x > xx(jm))) then
          jl=jm
       else
          ju=jm
       endif
       !
   goto 10
   endif

   j=jl
   return
END SUBROUTINE locate
