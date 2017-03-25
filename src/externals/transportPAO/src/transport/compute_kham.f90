!
! Copyright (C) 2006 WanT Group, 2017 ERMES Group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!***************************************************
    SUBROUTINE compute_kham( dimwann, nr, vr, wr, rham, vkpt, kham )
   !***************************************************
   !
   ! Calculates the hamiltonian on the bloch basis 
   ! once given on the Wannier basis
   !
   ! ham(:,:,k) = sum_R  w(R) * e^( i kR ) ham(:,:,R)    
   !
   ! units of vr and vkpt are considered consistent in
   ! such a way that  sum_i vr(i) * vkpt(i) 
   ! is the adimensional scalr product k dot R (given in cartesian coordinates)
   !
   USE kinds
   USE constants,      ONLY : CZERO
   USE log_module,     ONLY : log_push, log_pop
   USE timing_module,  ONLY : timing
   !
   IMPLICIT NONE
 
   !
   ! input variables
   !
   INTEGER,      INTENT(in)  :: dimwann, nr
   REAL(dbl),    INTENT(in)  :: vr( 3, nr),  wr(nr), vkpt(3)
   COMPLEX(dbl), INTENT(in)  :: rham( dimwann, dimwann, nr)
   COMPLEX(dbl), INTENT(out) :: kham( dimwann, dimwann )
 
   !
   ! local variables
   !
   INTEGER      :: i, j, ir 
   REAL(dbl)    :: arg
   COMPLEX(dbl) :: phase
   !
   ! end of declariations
   !

!
!------------------------------
! main body
!------------------------------
!
   CALL timing('compute_kham',OPR='start') 
   CALL log_push('compute_kham')
  
   kham( :, :) = CZERO
   !
   DO ir = 1, nr
       !
       arg =   DOT_PRODUCT( vkpt(:), vr(:, ir ) )
       phase = CMPLX( COS(arg), SIN(arg), dbl ) * wr(ir)
       !
       DO j = 1, dimwann
       DO i = 1, dimwann
           kham(i,j) = kham(i,j) + phase * rham(i, j, ir)
       ENDDO
       ENDDO
       !
   ENDDO

   CALL timing('compute_kham',OPR='stop') 
   CALL log_pop ('compute_kham')
    
END SUBROUTINE compute_kham

