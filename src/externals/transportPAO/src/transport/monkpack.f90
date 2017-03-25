! 
! Copyright (C) 2006 WanT Group, 2017 ERMES Group
! 
! This file is distributed under the terms of the 
! GNU General Public License. See the file `License' 
! in the root directory of the present distribution, 
! or http://www.gnu.org/copyleft/gpl.txt . 
! 
!*********************************************************
SUBROUTINE monkpack(nk, s, vkpt)
   !*********************************************************
   !
   ! compute a regular monkhorst-pack grid according
   ! to the given parameters.
   ! output kpts are in crystal coords
   !
   USE kinds
   USE constants,         ONLY : TWO
   IMPLICIT NONE

   !
   ! input variables
   !
   INTEGER,  INTENT(in)      :: nk(3), s(3)
   REAL(dbl),INTENT(out)     :: vkpt(3,*)
   !
   ! local variables
   !
   CHARACTER(8)              :: subname="monkpack"
   REAL(dbl)                 :: u(3)
   INTEGER                   :: ik
   INTEGER                   :: i,j,k

!
!----------------------------------------
! main Body
!----------------------------------------
!

   IF ( ANY(nk(:) <= 0 )  ) CALL errore(subname,'invalid nk',1)
   IF ( ANY(s(:)  < 0 )   ) CALL errore(subname,'invalid s',1)
   IF ( ANY(s(:)  > 1 )   ) CALL errore(subname,'invalid s',2)
   
   !
   ! setup vkpt in crystal coordinates
   !
   ik = 0

   DO i=1,nk(1)
       u(1) = REAL( s(1) + 2*i ) / (TWO * nk(1))
       u(1) = u(1) - REAL( NINT( u(1)) )
       !
       DO j=1,nk(2)
           u(2) = REAL( s(2) + 2*j ) / (TWO * nk(2))
           u(2) = u(2) - REAL( NINT( u(2)) )
           !
           DO k=1,nk(3)
               ik = ik+1
               u(3) = REAL( s(3) + 2*k ) / (TWO * nk(3))
               u(3) = u(3) - REAL( NINT( u(3)) )
               !
               vkpt(:,ik) = u(:)
           ENDDO
       ENDDO
   ENDDO

END SUBROUTINE monkpack


