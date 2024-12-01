!
! Copyright (C) 2002-2005 PWSCF group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
! Slightly modified by Andrea Ferretti
!
#if defined(__ABSOFT)
#  define getenv getenv_
#  define getarg getarg_
#  define iargc  iargc_
#endif
!
!----------------------------------------------------------------------------
SUBROUTINE input_from_file( iunit )
  !
  ! This subroutine checks program arguments and, if input file is present,
  ! attach input unit IUNIT to the specified file
  !
  USE io_global_module,    ONLY : ionode, ionode_id 
  USE mp,                  ONLY : mp_bcast
  !
  IMPLICIT NONE
  !
  ! input variables
  !
  INTEGER,  INTENT(IN)  :: iunit

  !
  ! local variables
  !
  INTEGER  :: iiarg, nargs
  INTEGER  :: ierr
  !
  ! do not define iargc as external: g95 does not like it
  INTEGER             :: iargc
  CHARACTER(LEN=256)  :: input_file
  !
  ! end of declariations
  !

!
!------------------------------
! main body
!------------------------------
!

  !
  ! ... Input from file ?
  !
  ierr  = 0
  !
  IF ( ionode ) THEN
     !
     nargs = iargc ()
     !
     DO iiarg = 1, ( nargs - 1 )
        !
        CALL getarg ( iiarg, input_file )
        !
        IF ( TRIM( input_file ) == '--input' .OR. &
             TRIM( input_file ) == '-input'  .OR. &
             TRIM( input_file ) == '--inp'   .OR. &
             TRIM( input_file ) == '-inp'    .OR. &
             TRIM( input_file ) == '--in'    .OR. & 
             TRIM( input_file ) == '-in'     .OR. &
             TRIM( input_file ) == '-i'        ) THEN
           !
           CALL getarg ( ( iiarg + 1 ) , input_file )
           !
           OPEN ( UNIT = iunit, FILE = input_file, FORM = 'FORMATTED', &
                  STATUS = 'OLD', IOSTAT = ierr )
        ENDIF
        !
     ENDDO
     !
  ENDIF
  !
  CALL mp_bcast(  ierr,         ionode_id )
  CALL mp_bcast(  input_file,   ionode_id )
  !
  IF ( ierr/=0 ) CALL errore( 'input_from_file', 'opening '//TRIM(input_file), ABS(ierr) )


END SUBROUTINE input_from_file

