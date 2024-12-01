! 
! Copyright (C) 2006 WanT Group, 2017 ERMES Group
! 
! This file is distributed under the terms of the 
! GNU General Public License. See the file `License' 
! in the root directory of the present distribution, 
! or http://www.gnu.org/copyleft/gpl.txt . 
! 
# include "f_defs.h"
!
!*********************************************
   MODULE log_module
!*********************************************
   IMPLICIT NONE
   PRIVATE
   SAVE
   !
   ! This module contains basic routines to handle log writing on DEBUG_MODE
   ! 
   ! routines in this module:
   ! SUBROUTINE  log_init( iunit, filename, maxdim)
   ! SUBROUTINE  log_deallocate( )
   ! SUBROUTINE  log_push( name )
   ! SUBROUTINE  log_pop( [name] )
   !   ! a simple interface to the previous couple
   ! SUBROUTINE  log_traceback([name][,opr])   
   !

   INTEGER           :: logunit
   CHARACTER(256)    :: logfile
   !
   LOGICAL           :: debug_mode
   INTEGER           :: stack_maxdim
   INTEGER           :: stack_index
   CHARACTER(256), ALLOCATABLE :: stack(:)
   !
   LOGICAL           :: alloc 

!
! end of declarations
!

   PUBLIC :: log_init
   PUBLIC :: log_deallocate
   PUBLIC :: log_traceback
   PUBLIC :: log_push
   PUBLIC :: log_pop
   PUBLIC :: alloc

CONTAINS

!
! Subroutines
!   

!**********************************************************
   SUBROUTINE log_init( iunit, debug, filename, debug_level )
   !**********************************************************
      IMPLICIT NONE
      INTEGER,      INTENT(IN) :: iunit
      LOGICAL,      INTENT(IN) :: debug
      CHARACTER(*), INTENT(IN) :: filename
      INTEGER,      INTENT(IN) :: debug_level
      !
      INTEGER :: ierr

      IF ( alloc ) CALL errore('log_init', 'log_module already initialized',1)
      IF ( debug_level <=0 .AND. debug ) &
                   CALL errore('log_init', 'debug and debug_level incoherent',2)
      
      logunit      = iunit
      debug_mode   = debug
      logfile      = TRIM( filename )
      stack_maxdim = debug_level
      stack_index  = 0
      !
      ! overwrite using preprocessor
      !
#ifdef __DEBUG_MODE
      debug_mode = .TRUE.
      IF ( debug_level <=0 ) stack_maxdim = 50
#endif
      !
      IF ( .NOT. debug_mode ) RETURN 
      !
      ALLOCATE( stack( stack_maxdim ), STAT=ierr )
      IF ( ierr/=0 ) CALL errore('log_init', 'allocating stack', ABS(ierr) )  
      !
      OPEN( logunit, FILE=logfile, IOSTAT=ierr )
         IF (ierr/=0) CALL errore('log_init', 'opening logfile '//TRIM(logfile), ABS(ierr))
         !
#if defined HAVE_MALLINFO_FALSE
         WRITE( logunit, "(2x, ' Time',6x, 'Memory [kB]   Routines',/ )")
#else
         WRITE( logunit, "(2x, ' Time',6x, 'Routines',/ )")
#endif
         !
      CLOSE( logunit )
      !
      alloc   = .TRUE.
      !
   END SUBROUTINE log_init
   

!**********************************************************
   SUBROUTINE log_deallocate( )
   !**********************************************************
      IMPLICIT NONE
      INTEGER :: ierr
      !
      IF ( .NOT. alloc ) RETURN 
      !
      IF ( ALLOCATED( stack ) ) THEN
           !
           DEALLOCATE( stack, STAT=ierr )
           IF ( ierr/=0 ) CALL errore('log_deallocate', 'deallocating stack', ABS(ierr) )  
           !
      ENDIF
      !
      alloc = .FALSE. 
      !
   END SUBROUTINE log_deallocate


!**********************************************************
   SUBROUTINE log_push( name )
   !**********************************************************
      IMPLICIT NONE
      CHARACTER(*),    INTENT(IN) :: name
      !
      CHARACTER(9) :: cdate, ctime
      INTEGER      :: istack, ierr
#ifdef HAVE_MALLINFO_FALSE
      INTEGER      :: memory
#endif
      !
      IF ( .NOT. debug_mode ) RETURN
      IF ( .NOT. alloc )      RETURN
      !
      stack_index = stack_index + 1
      !
      ! avoid writing log after a preset level
      !
      IF ( stack_index > stack_maxdim ) RETURN
           
      !
      stack( stack_index ) = TRIM( name ) 
      !
      CALL date_and_tim( cdate, ctime )
      ! 
      ! 
      ! Writes info on log file 
      !
      OPEN( logunit, FILE=logfile, IOSTAT=ierr, POSITION="APPEND" )
         !
         IF (ierr/=0) CALL errore('log_push', 'opening logfile '//TRIM(logfile), ABS(ierr))
         !
#ifdef HAVE_MALLINFO_FALSE
            CALL memstat( memory )
            WRITE( logunit, "( 2x, a9, ' | ', I9, 2x, ' | ')", ADVANCE="no" ) ctime, memory
#else
            WRITE( logunit, "( 2x, a9, ' | ')", ADVANCE="no" ) ctime
#endif
         !
         DO istack = 1, stack_index-1
            !
            WRITE( logunit, "(A)", ADVANCE="no" ) "..| "
            !
         ENDDO
         !
         WRITE( logunit, "(A)" ) TRIM( stack(stack_index) )
         !
      CLOSE( logunit )
 
   END SUBROUTINE log_push
         
  
!**********************************************************
   SUBROUTINE log_pop( name )
   !**********************************************************
      IMPLICIT NONE
      CHARACTER(*), OPTIONAL, INTENT(IN) :: name
      !
      IF ( .NOT. debug_mode ) RETURN
      IF ( .NOT. alloc )      RETURN
      !
      IF ( stack_index == 0 ) &
           CALL errore( 'log_pop', 'no routine to pop', 2)

      stack_index = stack_index - 1
      !
      ! avoid operations after a preset level
      IF ( stack_index+1 > stack_maxdim ) RETURN
      !
      IF ( PRESENT( name ) ) THEN
          !
          IF ( TRIM( stack ( stack_index+1 ) ) /= TRIM(name)  ) &
               CALL errore( 'log_pop', &
                            'missing match: '//TRIM(name)//", "//TRIM( stack ( stack_index+1 ) ), 3)
          !
      ENDIF
      !
      ! the stack_index has already been decreased
      stack ( stack_index+1 ) = " "
      !
   END SUBROUTINE log_pop


!**********************************************************
   SUBROUTINE log_traceback(name,opr)
   !**********************************************************
      IMPLICIT NONE
      CHARACTER(*), OPTIONAL, INTENT(in)    :: name
      CHARACTER(*), OPTIONAL, INTENT(in)    :: opr
      CHARACTER(4)                          :: opr_

      IF ( .NOT. debug_mode ) RETURN
      IF ( .NOT. alloc )      RETURN
      !
      IF ( .NOT. PRESENT( name ) ) THEN 
           !
           opr_ = "pop"
           IF ( PRESENT( opr ) .AND. &
              ( TRIM( opr ) == "push" .OR. TRIM( opr ) == "PUSH" ) ) &
                CALL errore('log_traceback', 'NAME should be present to PUSH',1)
           !
      ELSEIF ( .NOT. PRESENT( opr ) )  THEN
           !
           opr_ = "push"
           !
      ENDIF
      ! 
      SELECT CASE ( TRIM(opr_) )
      !
      CASE ( "push", "PUSH" )
           !
           CALL log_push( name )
           !
      CASE ( "pop", "POP")
           !
           IF ( PRESENT( name ) ) THEN
              !
              CALL log_pop( name )
              !
           ELSE
              !
              CALL log_pop(  )
              !
           ENDIF
           !
      CASE DEFAULT
           !
           CALL errore('log_traceback', 'invalid OPR',2)
           !
      END SELECT
      
   END SUBROUTINE log_traceback
            

END MODULE log_module

