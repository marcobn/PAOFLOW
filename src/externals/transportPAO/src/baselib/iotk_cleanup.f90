!
! Copyright (C) 2001-2007 Quantum-ESPRESSO group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!----------------------------------------------------------------------------
SUBROUTINE iotk_cleanup()
  !----------------------------------------------------------------------------
  !
  USE iotk_base,         ONLY : iotk_error_pool, iotk_error_pool_size, &
                                iotk_error_pool_used 
  USE iotk_error_interf, ONLY : iotk_error_clear
  !
  IMPLICIT NONE
  !
  INTEGER :: i
  !
  DO i = 1, iotk_error_pool_size
      !
      IF ( iotk_error_pool_used(i) ) CALL iotk_error_clear( iotk_error_pool(i) )
      !
  ENDDO

END SUBROUTINE iotk_cleanup

