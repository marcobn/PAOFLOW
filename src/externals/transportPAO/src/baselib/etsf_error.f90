!
! Copyright (C) 2001-2007 Quantum-ESPRESSO group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
#ifdef __ETSF_IO
!----------------------------------------------------------------------------
SUBROUTINE etsf_error(error_data,calling_routine,message,ierr)
  !----------------------------------------------------------------------------
  !
  USE io_global_module,  ONLY : ionode
  USE etsf_io_low_level, ONLY : etsf_io_low_error, etsf_io_low_error_handle
  !
  IMPLICIT NONE
  !
  TYPE(etsf_io_low_error), INTENT(IN) :: error_data 
  CHARACTER(LEN=*),        INTENT(IN) :: calling_routine, message
  INTEGER,                 INTENT(IN) :: ierr
  ! 

  !
  ! Handle ETSF_IO error data
  !
  IF ( ionode ) CALL etsf_io_low_error_handle(error_data) 
  !
  ! std call to error
  !
  CALL errore(TRIM(calling_routine), TRIM(message), ierr)

END SUBROUTINE etsf_error


#else
  !
  ! avoid compilation problems
  !
  SUBROUTINE etsf_error_aux__
     WRITE(*,*)
  END SUBROUTINE etsf_error_aux__
#endif

