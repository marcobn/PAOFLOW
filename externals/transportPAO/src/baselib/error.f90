!
! Copyright (C) 2001-2007 Quantum-ESPRESSO group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!----------------------------------------------------------------------------
SUBROUTINE errore( calling_routine, message, ierr )
  !----------------------------------------------------------------------------
  !
  ! ... This is a simple routine which writes an error message to output: 
  ! ... if ierr <= 0 it does nothing, 
  ! ... if ierr  > 0 it stops.
  !
  ! ...          **** Important note for parallel execution ***
  !
  ! ... in parallel execution unit 6 is written only by the first node;
  ! ... all other nodes have unit 6 redirected to nothing (/dev/null).
  ! ... As a consequence an error not occurring on the first node
  ! ... will be invisible. For T3E and ORIGIN machines, this problem
  ! ... is solved by writing an error message to unit * instead of 6.
  ! ... Whenever possible (IBM SP machines), we write to the standard
  ! ... error, unit 0 (the message will appear in the error files 
  ! ... produced by loadleveler).
  !
  USE io_global_module, ONLY : stdout
  USE parallel_include
  !
  IMPLICIT NONE
  !
  INTEGER,       PARAMETER :: crashunit = 50
  CHARACTER(10), PARAMETER :: crash_file = "CRASH"
  !
  CHARACTER(LEN=*), INTENT(IN) :: calling_routine, message
    ! the name of the calling calling_routinee
    ! the output messagee
  INTEGER,          INTENT(IN) :: ierr
    ! the error flag
#if defined (__PARA) && defined (__MPI)
  INTEGER                      :: mpime, mpierr
#endif
    ! the task id  
    !
  !
  !
  IF ( ierr <= 0 ) RETURN
  !
  ! ... the error message is written un the "*" unit
  !
  WRITE( UNIT = *, FMT = '(/,1X,78("%"))' )
  WRITE( UNIT = *, &
         FMT = '(5X,"from ",A," : error #",I10)' ) TRIM(calling_routine), ierr
  WRITE( UNIT = *, FMT = '(5X,A)' ) message
  WRITE( UNIT = *, FMT = '(1X,78("%"),/)' )
  !
#if defined (__PARA) && defined (__AIX)
  !
  ! ... in the case of ibm machines it is also written on the "0" unit
  ! ... which is automatically connected to stderr
  !
  WRITE( UNIT = 0, FMT = '(/,1X,78("%"))')
  WRITE( UNIT = 0, &
         FMT = '(5X,"from ",A," : error #",I10)' ) TRIM(calling_routine), ierr
  WRITE( UNIT = 0, FMT = '(5X,A)' ) message
  WRITE( UNIT = 0, FMT = '(1X,78("%"),/)' )
  !
#endif
  !
  WRITE( *, '("     stopping ...")' )
  !
  CALL flush_unit( stdout )
  !
#if defined (__PARA) && defined (__MPI)
  !
  mpime = 0
  !
  CALL MPI_COMM_RANK( MPI_COMM_WORLD, mpime, mpierr )
  !
  !  .. write the message to a file and close it before exiting
  !  .. this will prevent loss of information on systems that
  !  .. do not flush the open streams
  !  .. added by C.C.
  !
  OPEN( UNIT = crashunit, FILE = crash_file, &
        POSITION = 'APPEND', STATUS = 'UNKNOWN' )
  !      
  WRITE( UNIT = crashunit, FMT = '(/,1X,78("%"))' )
  WRITE( UNIT = crashunit, FMT = '(5X,"task #",I10)' ) mpime
  WRITE( UNIT = crashunit, &
         FMT = '(5X,"from ",A," : error #",I10)' ) TRIM(calling_routine), ierr
  WRITE( UNIT = crashunit, FMT = '(5X,A)' ) message
  WRITE( UNIT = crashunit, FMT = '(1X,78("%"),/)' )
  !
  CLOSE( UNIT = crashunit )
  !
  ! ... try to exit in a smooth way
  !
  CALL MPI_ABORT( MPI_COMM_WORLD, mpierr )
  !
  CALL MPI_FINALIZE( mpierr )
  !
#endif
  !
  STOP 2
  !
  RETURN
  !
END SUBROUTINE errore
