!
! Copyright (C) 2009 WanT Group, 2017 ERMES Group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!***************************************************
    SUBROUTINE compute_rham( dimwann, vr, rham, nkpts, vkpt, wk, kham )
   !***************************************************
   !
   ! Calculates the hamiltonian on the bloch basis 
   ! once given on the Wannier basis
   !
   ! ham(:,:,R) = sum_k  w(k) * e^( -i kR ) ham(:,:,k)    
   !
   ! units of vr and vkpt are considered consistent in
   ! such a way that  sum_i vr_i(:) * vkpt_i(:) 
   ! is the adimensional scalar product k dot R (given in cartesian coordinates)
   !
   ! Note that in the case we are using parallelism we have to get rid of
   ! that independently
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
   INTEGER,      INTENT(IN)   :: dimwann, nkpts
   REAL(dbl),    INTENT(IN)   :: vr(3),  wk(nkpts), vkpt(3, nkpts)
   COMPLEX(dbl), INTENT(IN)   :: kham( dimwann, dimwann, nkpts )
   COMPLEX(dbl), INTENT(OUT)  :: rham( dimwann, dimwann )
 
   !
   ! local variables
   !
   INTEGER        :: i, j, ik 
   REAL(dbl)      :: arg
   COMPLEX(dbl)   :: phase
   CHARACTER(12)  :: subname='compute_rham'
   !
   ! end of declariations
   !

!
!------------------------------
! main body
!------------------------------
!
   CALL timing(subname,OPR='start') 
   CALL log_push(subname)
  
   rham( :, :) = CZERO
   !
   DO ik = 1, nkpts
       !
       arg =   DOT_PRODUCT( vkpt(:,ik), vr(:) )
       phase = CMPLX( COS(arg), -SIN(arg), dbl )
       !
       DO j = 1, dimwann
       DO i = 1, dimwann
           rham(i,j) = rham(i,j) + phase * wk(ik) * kham(i, j, ik)
       ENDDO
       ENDDO
       !
   ENDDO

   CALL timing(subname,OPR='stop') 
   CALL log_pop (subname)
    
END SUBROUTINE compute_rham

