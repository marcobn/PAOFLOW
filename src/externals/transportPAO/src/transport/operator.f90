!
! Copyright (C) 2007 WanT Group, 2017 ERMES Group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!----------------------------------------------------------------------------
MODULE operator_module
  !----------------------------------------------------------------------------
  !
  ! This module contains subroutines used to read and write
  ! operators and dynamical operators on the Wannier basis in
  ! iotk-XML fmt
  !
  USE kinds
  USE constants,      ONLY : ZERO
  USE files_module,   ONLY : file_exist, file_open, file_close
  USE iotk_module
  !
  IMPLICIT NONE
  !
  PRIVATE
  !
  CHARACTER( iotk_attlenx ) :: attr
  !
  !
  PUBLIC :: operator_read_init
  PUBLIC :: operator_read_close
  PUBLIC :: operator_read_aux
  PUBLIC :: operator_read_data
  !
  PUBLIC :: operator_write_init
  PUBLIC :: operator_write_close
  PUBLIC :: operator_write_aux
  PUBLIC :: operator_write_data
  !
CONTAINS

!
!-------------------------------------------
! ... basic (public) subroutines
!-------------------------------------------
!
    !
    !==========================
    ! READ routines
    !==========================
    !
    !------------------------------------------------------------------------
    SUBROUTINE operator_read_init( iun, filename, ierr )
      !------------------------------------------------------------------------
      !
      INTEGER,           INTENT(IN)  :: iun
      CHARACTER(LEN=*),  INTENT(IN)  :: filename
      INTEGER,           INTENT(OUT) :: ierr
      !
      ierr = 0
      !
      IF ( .NOT. file_exist(filename) ) THEN
          ierr = 1
          RETURN
      ENDIF
      !
      CALL file_open( iun, TRIM(filename), PATH="/", ACTION="read", IERR=ierr )
      IF ( ierr/=0 ) ierr = 2
      !
      RETURN
      !
    END SUBROUTINE operator_read_init
    !
    !
    !------------------------------------------------------------------------
    SUBROUTINE operator_read_close( iun, ierr )
      !------------------------------------------------------------------------
      !
      INTEGER,           INTENT(IN)  :: iun
      INTEGER,           INTENT(OUT) :: ierr
      !
      CALL file_close( iun, PATH="/", ACTION="read", IERR=ierr )
      IF ( ierr/=0 ) ierr = 2
      !
    END SUBROUTINE operator_read_close
    !
    !
    !------------------------------------------------------------------------
    SUBROUTINE operator_read_aux( iun, dimwann, dynamical, nomega, &
                                  iomg_s, iomg_e, grid, eunits, analyticity, &
                                  nr, vr, ivr, ierr ) 
      !------------------------------------------------------------------------
      !
      INTEGER,                     INTENT(IN)  :: iun
      INTEGER,           OPTIONAL, INTENT(OUT) :: dimwann, nomega, nr
      LOGICAL,           OPTIONAL, INTENT(OUT) :: dynamical
      INTEGER,           OPTIONAL, INTENT(OUT) :: iomg_s, iomg_e
      REAL(dbl),         OPTIONAL, INTENT(OUT) :: grid(:), vr(:,:)
      CHARACTER(LEN=*),  OPTIONAL, INTENT(OUT) :: eunits
      INTEGER,           OPTIONAL, INTENT(OUT) :: ivr(:,:)
      CHARACTER(LEN=*),  OPTIONAL, INTENT(OUT) :: analyticity
      INTEGER,                     INTENT(OUT) :: ierr
      !
      INTEGER           :: dimwann_, nomega_, nr_
      INTEGER           :: iomg_s_, iomg_e_
      LOGICAL           :: dynamical_
      CHARACTER(256)    :: analyticity_
      REAL(dbl), ALLOCATABLE :: grid_(:)
      !

      ierr=0
      !
      !
      CALL iotk_scan_empty( iun, "DATA", ATTR=attr, IERR=ierr )
      IF ( ierr /= 0 ) RETURN
      !
      CALL iotk_scan_attr( attr, "dimwann", dimwann_, IERR=ierr )
      IF (ierr/=0) RETURN
      CALL iotk_scan_attr( attr, "nrtot", nr_, IERR=ierr )
      IF ( ierr /= 0 ) RETURN
      CALL iotk_scan_attr( attr, "dynamical", dynamical_, IERR=ierr )
      IF ( ierr /= 0 ) RETURN
      !
      IF ( dynamical_ ) THEN
         !
         CALL iotk_scan_attr( attr, "nomega", nomega_, IERR=ierr )
         IF ( ierr /= 0 ) RETURN
         !
         CALL iotk_scan_attr( attr, "iomg_s", iomg_s_, IERR=ierr )
         IF ( ierr /= 0 ) RETURN
         !
         CALL iotk_scan_attr( attr, "iomg_e", iomg_e_, IERR=ierr )
         IF ( ierr /= 0 ) RETURN
         !
         CALL iotk_scan_attr( attr, "analyticity", analyticity_, IERR=ierr )
         IF ( ierr /= 0 ) RETURN
         !
      ELSE
         nomega_ = 1
         analyticity_ = ""
         iomg_s_ = 1
         iomg_e_ = 1
      ENDIF
      !
      IF ( PRESENT ( vr ) ) THEN
         !
         CALL iotk_scan_dat( iun, "VR", vr, IERR=ierr )
         IF (ierr/=0) RETURN
         !
      ENDIF
      !
      IF ( PRESENT ( ivr ) ) THEN
         !
         CALL iotk_scan_dat( iun, "IVR", ivr, IERR=ierr )
         IF (ierr/=0) RETURN
         !
      ENDIF
      !
      IF ( PRESENT ( grid ) .OR. PRESENT ( eunits ) ) THEN
         !
         IF ( dynamical_ ) THEN
            !
            ALLOCATE( grid_(nomega_), STAT=ierr )
            IF (ierr/=0) RETURN
            !
            CALL iotk_scan_dat( iun, "GRID", grid_, ATTR=attr, IERR=ierr )
            IF (ierr/=0) RETURN
            !
            IF ( PRESENT(grid) )   grid(:) = grid_(:)
            !
            IF ( PRESENT(eunits) ) THEN
                CALL iotk_scan_attr(attr, "units", eunits, IERR=ierr)
                IF (ierr/=0) RETURN
            ENDIF
            !
            DEALLOCATE( grid_, STAT=ierr )
            IF (ierr/=0) RETURN
            !
         ELSE
            grid(:) = ZERO
            eunits  = " " 
         ENDIF
         !
      ENDIF
      !
      ! 
      IF ( PRESENT(dimwann) )      dimwann      = dimwann_
      IF ( PRESENT(nr) )           nr           = nr_
      IF ( PRESENT(dynamical) )    dynamical    = dynamical_
      IF ( PRESENT(nomega) )       nomega       = nomega_
      IF ( PRESENT(analyticity) )  analyticity  = TRIM( analyticity_ )
      IF ( PRESENT(iomg_s) )       iomg_s       = iomg_s_
      IF ( PRESENT(iomg_e) )       iomg_e       = iomg_e_
      !
    END SUBROUTINE operator_read_aux
    !
    !
    !------------------------------------------------------------------------
    SUBROUTINE operator_read_data( iun, ie, r_opr, ierr )
      !------------------------------------------------------------------------
      !
      INTEGER,                 INTENT(IN)  :: iun
      INTEGER,       OPTIONAL, INTENT(IN)  :: ie
      COMPLEX(dbl),  OPTIONAL, INTENT(OUT) :: r_opr(:,:,:)
      INTEGER,                 INTENT(OUT) :: ierr
      !
      CHARACTER( 256 ) :: str
      INTEGER          :: ir
      LOGICAL          :: ldynam
      !
      ierr = 0
      !
      ldynam = .FALSE.
      IF ( PRESENT (ie) ) ldynam = .TRUE.
      !
      IF ( PRESENT ( r_opr ) ) THEN
         !
         str = "OPR"
         IF ( ldynam ) str = TRIM(str)//TRIM( iotk_index(ie) ) 
         !
         CALL iotk_scan_begin( iun, TRIM(str), IERR=ierr )
         IF ( ierr/=0 ) RETURN
         !
         DO ir = 1, SIZE( r_opr, 3 )
            !
            CALL iotk_scan_dat( iun, "VR"//TRIM(iotk_index(ir)), r_opr(:,:,ir), IERR=ierr )
            IF ( ierr/=0 ) RETURN
            !
         ENDDO
         !
         CALL iotk_scan_end( iun, TRIM(str), IERR=ierr )
         IF ( ierr/=0 ) RETURN
         !
      ENDIF
      !
    END SUBROUTINE operator_read_data
    !
    !
    !==========================
    ! WRITE routines
    !==========================
    !
    !------------------------------------------------------------------------
    SUBROUTINE operator_write_aux( iun, dimwann, dynamical, nomega, iomg_s, iomg_e, &
                                   grid, eunits, analyticity, nrtot, vr, ivr ) 
      !------------------------------------------------------------------------
      !
      INTEGER,                     INTENT(IN)  :: iun
      INTEGER,                     INTENT(IN)  :: dimwann, nomega, nrtot
      INTEGER,           OPTIONAL, INTENT(IN)  :: iomg_s, iomg_e
      LOGICAL,                     INTENT(IN)  :: dynamical
      REAL(dbl),         OPTIONAL, INTENT(IN)  :: vr(:,:)
      REAL(dbl),         OPTIONAL, INTENT(IN)  :: grid(:)
      INTEGER,           OPTIONAL, INTENT(IN)  :: ivr(:,:)
      CHARACTER(LEN=*),  OPTIONAL, INTENT(IN)  :: analyticity
      CHARACTER(LEN=*),  OPTIONAL, INTENT(IN)  :: eunits
      !
      CHARACTER(256)    :: analyticity_
      CHARACTER(18)     :: subname="operator_write_aux"
      !
      !
      IF ( dynamical .AND. .NOT. PRESENT(grid) ) CALL errore(subname,'grid must be present',10)
      IF ( dynamical .AND. .NOT. PRESENT(analyticity) ) CALL errore(subname,'analyt must be present',10)
      IF ( .NOT. PRESENT(vr) .AND. .NOT. PRESENT(ivr) ) CALL errore(subname,'both VR and IVR not present',10)
      !
      IF ( .NOT. dynamical .AND. nomega /= 1 ) CALL errore(subname,'invalid nomega',10)
      !
      analyticity_ = ""
      IF (PRESENT( analyticity)) analyticity_ = TRIM(analyticity)
      
      CALL iotk_write_attr( attr, "dimwann", dimwann, FIRST=.TRUE. )
      CALL iotk_write_attr( attr, "nrtot", nrtot )
      CALL iotk_write_attr( attr, "dynamical", dynamical )
      CALL iotk_write_attr( attr, "nomega", nomega )
      !
      IF ( PRESENT(iomg_s) ) CALL iotk_write_attr( attr, "iomg_s", iomg_s )
      IF ( PRESENT(iomg_e) ) CALL iotk_write_attr( attr, "iomg_e", iomg_e )
      !
      IF ( dynamical ) CALL iotk_write_attr( attr, "analyticity", TRIM(analyticity_) )
      !
      CALL iotk_write_empty( iun, "DATA", ATTR=attr )
      !
      !
      IF ( PRESENT ( vr ) ) THEN
          !
          CALL iotk_write_dat( iun, "VR", vr, COLUMNS=SIZE(vr,1) )
          !
      ENDIF
      !
      IF ( PRESENT ( ivr ) ) THEN
          !
          CALL iotk_write_dat( iun, "IVR", ivr, COLUMNS=SIZE(ivr,1) )
          !
      ENDIF
      !
      IF ( PRESENT ( grid ) ) THEN
          !
          IF ( PRESENT( eunits ) ) THEN
              CALL iotk_write_attr(attr, "units", TRIM(eunits), FIRST=.TRUE.)
              CALL iotk_write_dat( iun, "GRID", grid, COLUMNS=4, ATTR=attr )
          ELSE
              CALL iotk_write_dat( iun, "GRID", grid, COLUMNS=4 )
          ENDIF
          !
      ENDIF
      !
      RETURN
      ! 
    END SUBROUTINE operator_write_aux
    !
    !
    !------------------------------------------------------------------------
    SUBROUTINE operator_write_data( iun, r_opr, dynamical, ie )
      !------------------------------------------------------------------------
      !
      INTEGER,                 INTENT(IN)  :: iun
      COMPLEX(dbl),            INTENT(IN)  :: r_opr(:,:,:)
      LOGICAL,                 INTENT(IN)  :: dynamical
      INTEGER,       OPTIONAL, INTENT(IN)  :: ie
      !
      CHARACTER( 19 )  :: subname="operator_write_data"
      CHARACTER( 256 ) :: str
      INTEGER          :: ir
      !
      !
      IF ( dynamical .AND. .NOT. PRESENT (ie) ) CALL errore(subname,'ie is needed',10)
      !
      str = "OPR"
      IF ( dynamical ) str = TRIM(str)//TRIM( iotk_index(ie) ) 
      !
      CALL iotk_write_begin( iun, TRIM(str) )
      !
      DO ir = 1, SIZE( r_opr, 3 )
          !
          CALL iotk_write_dat( iun, "VR"//TRIM(iotk_index(ir)), r_opr(:,:,ir) )
          !
      ENDDO
      !
      CALL iotk_write_end( iun, TRIM(str) )
      !
      RETURN
      ! 
    END SUBROUTINE operator_write_data
    !
    !
    !------------------------------------------------------------------------
    SUBROUTINE operator_write_init( iun, filename, binary )
      !------------------------------------------------------------------------
      !
      INTEGER,           INTENT(IN)  :: iun
      CHARACTER(LEN=*),  INTENT(IN)  :: filename
      LOGICAL, OPTIONAL, INTENT(IN)  :: binary
      !
      CHARACTER(19) :: subname="operator_write_init"
      CHARACTER(20) :: form
      LOGICAL       :: binary_ = .TRUE.
      INTEGER       :: ierr
      !
      IF ( PRESENT(binary) ) binary_ = binary
      !
      form = "UNFORMATTED"
      IF ( .NOT. binary_ ) form="FORMATTED"
      !
      CALL file_open( iun, TRIM(filename), PATH="/", ACTION="write", FORM=TRIM(form), IERR=ierr )
      IF ( ierr/=0 ) CALL errore(subname,'opening '//TRIM(filename), ABS(ierr))
      !
    END SUBROUTINE operator_write_init
    ! 
    ! 
    !------------------------------------------------------------------------
    SUBROUTINE operator_write_close( iun )
      !------------------------------------------------------------------------
      !
      INTEGER,           INTENT(IN)  :: iun
      !
      CHARACTER(20) :: subname="operator_write_close"
      INTEGER       :: ierr
      !
      CALL file_close( iun, PATH="/", ACTION="write", IERR=ierr )
      IF ( ierr/=0 ) CALL errore(subname,'closing file', ABS(ierr))
      !
    END SUBROUTINE operator_write_close
    !
    !
END MODULE operator_module

