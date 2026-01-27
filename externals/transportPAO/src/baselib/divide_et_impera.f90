!
! Copyright (C) 2006 Andrea Ferretti
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!

!*********************************************
   SUBROUTINE divide_et_impera(ns_global, ne_global, ns_local,  ne_local,   mpime, nproc  )
   !*********************************************
   !
   ! given the global indexes of a loop, divide the 
   ! loop over the NPROC tasks, and determine the 
   ! extrema for the local process identified by MPIME
   !
   IMPLICIT NONE
   !
   INTEGER, INTENT(in)     :: ns_global, ne_global
   INTEGER, INTENT(in)     :: mpime, nproc
   INTEGER, INTENT(out)    :: ns_local, ne_local
   !
   ! local variables
   !
   CHARACTER(16)  :: subname='divide_et_impera'
   INTEGER, ALLOCATABLE :: dim_local(:)
   INTEGER        :: dim_global, dim_remind, i, ierr
   !
!
!----------------
! main body
!----------------
!
   IF ( nproc < 1 ) CALL errore(subname,'invalid nproc_',1)
   IF ( mpime < 0 ) CALL errore(subname,'invalid mpime',1)
   IF ( mpime >= nproc ) CALL errore(subname,'mpime too large',mpime)
   IF ( ne_global < ns_global ) CALL errore(subname,'invalid global indeces',1)
   !
   dim_global = ne_global - ns_global +1
   !
   ALLOCATE( dim_local( 0:nproc-1), STAT=ierr )
   IF (ierr/=0) CALL errore(subname,'allocating dim_local',ABS(ierr))
   !
   dim_local(:)  = dim_global / nproc
   !
   dim_remind = MOD( dim_global, nproc )
   !
   DO i=1, dim_remind
      dim_local(i-1) = dim_local(i-1) + 1
   ENDDO 
   !
   IF ( SUM(dim_local(:)) /= dim_global ) CALL errore(subname,'invalid sum rule',1)
   !
   ns_local  = ns_global + SUM( dim_local(0:mpime -1 ) )
   ne_local  = ns_local  + dim_local( mpime ) -1  
   !
   DEALLOCATE( dim_local, STAT=ierr )
   IF (ierr/=0) CALL errore(subname,'allocating dim_local',ABS(ierr))

END SUBROUTINE divide_et_impera 

