!
! Copyright (C) 2006 WanT Group, 2017 ERMES Group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!**********************************************************
   SUBROUTINE shutdown(main_name)
   !**********************************************************
   !
   ! This routine stops the code in a proper way,
   ! according to the MPI environment if the case.
   !
   USE io_global_module,    ONLY : stdout, ionode
   USE timing_module,       ONLY : global_list, timing, timing_deallocate, timing_overview
   USE log_module,          ONLY : log_deallocate, log_alloc => alloc, log_pop
   USE mp,                  ONLY : mp_end
   !
   IMPLICIT NONE

   !
   ! input variables
   !
   CHARACTER(*), INTENT(in) :: main_name

!
!------------
! main body
!------------
!
!

   !
   ! shutdown the clocks 
   !
   CALL timing(TRIM(main_name),OPR="stop")
   !
   IF ( ionode ) WRITE( stdout, "(/,2x, 70('='))" )
   !
   CALL timing_overview( UNIT=stdout, LIST=global_list, MAIN_NAME=TRIM(main_name))
   CALL timing_deallocate()


   !
   ! eventually shutdown log writing
   !
   CALL log_pop ( )
   IF ( log_alloc ) CALL log_deallocate()
        
   !
   ! GPU/CUDA, phiGEMM
   !
#if ( defined __CUDA || defined __PHIGEMM )
   !CALL phiGemmShutdown()
   CALL CloseCudaEnv()
#endif

   !
   ! shutdown the MPI
   !
   CALL mp_end()
   !
END SUBROUTINE shutdown

