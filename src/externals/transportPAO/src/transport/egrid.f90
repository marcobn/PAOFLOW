!
! Copyright (C) 2005 WanT Group, 2017 ERMES Group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License\'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!*********************************************
   MODULE T_egrid_module
   !*********************************************
   !
   USE kinds,           ONLY : dbl
   USE log_module,      ONLY : log_push, log_pop
   !
   IMPLICIT NONE
   PRIVATE 
   SAVE
!
! Contains transport energy-grid data
! 
   
   INTEGER                :: ne        ! dimension of the energy grid
   INTEGER                :: ne_buffer ! how many frequencies are stored together
                                       ! for the correlation sgm
   REAL(dbl)              :: emin      !
   REAL(dbl)              :: emax      ! egrid extrema 
   !
   REAL(dbl):: de
   REAL(dbl), ALLOCATABLE :: egrid(:)  ! grid values
   !
   LOGICAL :: alloc = .FALSE.

!
! end delcarations
!

   PUBLIC :: ne, emin, emax
   PUBLIC :: ne_buffer
   PUBLIC :: egrid
   PUBLIC :: alloc
   !
   PUBLIC :: egrid_init, egrid_init_ph, de
   PUBLIC :: egrid_buffer_doread
   PUBLIC :: egrid_buffer_iend
   PUBLIC :: egrid_deallocate


CONTAINS

!
! subroutines
!

!**********************************************************
   SUBROUTINE egrid_init()
   !**********************************************************
   IMPLICIT NONE
       CHARACTER(10) :: subname="egrid_init"
       INTEGER       :: ie, ierr
       !
       CALL log_push ( 'egrid_init' )

       IF ( alloc )   CALL errore(subname,'already allocated', 1 )
       IF ( ne <= 0 ) CALL errore(subname,'invalid ne', -ne+1 )
       IF ( ne_buffer <= 0 ) CALL errore(subname,'invalid ne_buffer', -ne_buffer+1 )
       
       ALLOCATE( egrid(ne), STAT=ierr )
       IF (ierr/=0) CALL errore(subname,'allocating egrid', ABS(ierr))

       !
       ! setting the energy grid
       !
       de = (emax - emin) / REAL(ne -1, dbl)
       !
       DO ie = 1, ne
          egrid(ie) = emin + REAL(ie -1, dbl) * de
       ENDDO

       alloc = .TRUE.
       CALL log_pop ( 'egrid_init' )
       !
   END SUBROUTINE egrid_init

   !**********************************************************
   SUBROUTINE egrid_init_ph()
   !**********************************************************
   IMPLICIT NONE
       CHARACTER(13) :: subname="egrid_init_ph"
       INTEGER       :: ie, ierr
       !
       CALL log_push ( 'egrid_init_ph' )

       IF ( alloc )   CALL errore(subname,'already allocated', 1 )
       IF ( ne <= 0 ) CALL errore(subname,'invalid ne', -ne+1 )
       IF ( ne_buffer <= 0 ) CALL errore(subname,'invalid ne_buffer', -ne_buffer+1 )

       
       ALLOCATE( egrid(ne), STAT=ierr )
       IF (ierr/=0) CALL errore(subname,'allocating egrid', ABS(ierr))

       !
       ! setting the energy grid
       !
       de = (emax - emin) / REAL(ne -1, dbl)
       !
       DO ie = 1, ne
          egrid(ie) = emin + de * REAL(ie -1, dbl)
       ENDDO
       egrid(1)=egrid(2)/100.0

       alloc = .TRUE.
       CALL log_pop ( 'egrid_init_ph' )
       !
   END SUBROUTINE egrid_init_ph





!**********************************************************
   SUBROUTINE egrid_deallocate()
   !**********************************************************
   IMPLICIT NONE
       CHARACTER(16)      :: subname="egrid_deallocate"
       INTEGER :: ierr
       CALL log_push ( 'egrid_deallocate' )

       IF ( ALLOCATED(egrid) ) THEN
           DEALLOCATE(egrid, STAT=ierr)
           IF (ierr/=0) CALL errore(subname,'deallocating egrid',ABS(ierr))
       ENDIF
       !
       alloc = .FALSE.
       CALL log_pop ( 'egrid_deallocate' )
       !
   END SUBROUTINE egrid_deallocate


!**********************************************************
   FUNCTION egrid_buffer_doread(  ie_g, iomg_s, iomg_e, ne_buffer )
   !**********************************************************
   IMPLICIT NONE
       INTEGER       :: ie_g, iomg_s, iomg_e, ne_buffer
       LOGICAL       :: egrid_buffer_doread
       !
       CHARACTER(19) :: subname="egrid_buffer_doread"
       LOGICAL       :: doread
       INTEGER       :: ie
       ! 
       !
       doread = .FALSE.
       !
       IF ( iomg_e < 0 ) CALL errore(subname,"invalid iomg_e",10)
       !
       ie = ie_g -iomg_s + 1
       !
       IF ( ne_buffer == 1           .OR. & 
            MOD( ie, ne_buffer ) == 1 ) doread=.TRUE.
       !
       egrid_buffer_doread = doread
       !
   END FUNCTION

!**********************************************************
   FUNCTION egrid_buffer_iend(  ie_g, iomg_s, iomg_e, ne_buffer )
   !**********************************************************
   IMPLICIT NONE
       INTEGER       :: ie_g, iomg_s, iomg_e, ne_buffer
       INTEGER       :: egrid_buffer_iend
       !
       CHARACTER(17) :: subname="egrid_buffer_iend"
       INTEGER       :: iend
       ! 
       !
       IF ( iomg_s < 0 ) CALL errore(subname,"invalid iomg_s",10)
       IF ( ie_g + ne_buffer -1 <= iomg_e ) THEN
           !
           iend =  ie_g + ne_buffer -1
           !
       ELSE
           !
           iend =  iomg_e
           !
       ENDIF
       !
       egrid_buffer_iend = iend
       !
   END FUNCTION
   !
   !
END MODULE T_egrid_module

