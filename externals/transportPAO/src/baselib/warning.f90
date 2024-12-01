!
! Copyright (C) 2005 WanT Group, 2017 ERMES Group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
! based on the infomsg routine from Quantum-Espresso
!
!----------------------------------------------------------------------
SUBROUTINE warning( routine, message )
  !----------------------------------------------------------------------
  !
  ! ... This is a simple routine which writes an info message 
  ! ... from a given routine to output. 
  !
  USE io_global_module,  ONLY : stdout, ionode
  !
  IMPLICIT NONE
  !
  CHARACTER (LEN=*) :: routine, message
  ! the name of the calling routine
  ! the output message
  !
  IF ( ionode ) THEN
     !   
     WRITE( stdout , '(2X,"WARNING from routine ",A,":")' ) routine
     WRITE( stdout , '(2X,A,/)' ) message
     !   
  END IF
  !
  RETURN
  !
END SUBROUTINE warning

