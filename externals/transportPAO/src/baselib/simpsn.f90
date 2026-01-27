!
! Copyright (C) 2001 PWSCF group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
! Modified Jan 2005 by Andrea Ferretti
!
!-----------------------------------------------------------------------

subroutine simpson (mesh, func, rab, asum)
  !-----------------------------------------------------------------------
  !
  !     simpson's rule integrator for function stored on the
  !     radial logarithmic mesh
  !
  USE kinds, ONLY: dbl
  IMPLICIT NONE
  integer :: i, mesh

  REAL(KIND=dbl) :: rab (mesh), func (mesh), f1, f2, f3, r12, asum
  !     routine assumes that mesh is an odd number so run check
  !     if ( mesh+1 - ( (mesh+1) / 2 ) * 2 .ne. 1 ) then
  !       write(*,*) '***error in subroutine radlg'
  !       write(*,*) 'routine assumes mesh is odd but mesh =',mesh+1
  !       stop
  !     endif
  asum = 0.0_dbl
  r12 = 1.0_dbl / 12.0_dbl

  f3 = func (1) * rab (1) * r12
  DO i = 2, mesh - 1, 2
     f1 = f3
     f2 = func (i) * rab (i) * r12
     f3 = func (i + 1) * rab (i + 1) * r12
     asum = asum + 4.0_dbl * f1 + 16.0_dbl * f2 + 4.0_dbl * f3
  ENDDO

  RETURN
END SUBROUTINE simpson

