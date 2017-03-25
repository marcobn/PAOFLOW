!
!      Copyright (C) 2005 WanT Group, 2017 ERMES Group
!
!      This file is distributed under the terms of the
!      GNU General Public License. See the file `License'
!      in the root directory of the present distribution,
!      or http://www.gnu.org/copyleft/gpl.txt .
!
!*******************************************************************
   SUBROUTINE fourier_par(kh, dim1, dim2, rh, ldr1, ldr2)
   !*******************************************************************
   !
   ! 2D Fourier transform R to k 
   !
   USE kinds,                ONLY : dbl
   USE constants,            ONLY : CZERO
   USE T_kpoints_module,     ONLY : nkpts_par, nrtot_par, table_par, wr_par
   USE log_module,           ONLY : log_push, log_pop
   USE timing_module
   !
   IMPLICIT NONE
  
   !
   ! IO variables
   !
   INTEGER,      INTENT(in)  :: dim1,dim2                 ! conductor/lead dimensions
   COMPLEX(dbl), INTENT(out) :: kh(dim1,dim2,nkpts_par)   ! reciprocal space matrix
   INTEGER,      INTENT(in)  :: ldr1, ldr2                ! auxiliary matrix lead dimension 
   COMPLEX(dbl), INTENT(in)  :: rh(ldr1,ldr2,nrtot_par)   ! real space matrix
   !
   ! local variables
   !
   INTEGER :: i, j, ik, ir      

!
!------------------------------
! main body
!------------------------------
!
    CALL timing('fourier_par',OPR='start')
    CALL log_push('fourier_par')

    DO ik = 1, nkpts_par
       !
       DO j = 1, dim2
       DO i = 1, dim1
          !
          kh(i,j,ik) = CZERO
          !
          DO ir = 1, nrtot_par
               kh(i,j,ik) = kh(i,j,ik) + wr_par(ir) * table_par(ir,ik) * rh(i,j,ir) 
          ENDDO
          !
       ENDDO
       ENDDO
    ENDDO

    CALL timing('fourier_par',OPR='stop')
    CALL log_pop('fourier_par')
    !
END SUBROUTINE fourier_par


