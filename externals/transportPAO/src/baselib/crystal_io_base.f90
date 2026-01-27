!
! Copyright (C) 2008 WanT Group, 2017 ERMES Group
! This file is distributed under the terms of the
! GNU Lesser General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/lgpl.txt .
!
!----------------------------------------------------------------------------
MODULE crystal_io_base_module
  !----------------------------------------------------------------------------
  !
  USE iotk_module
  IMPLICIT NONE
  !
  PRIVATE
  !
  ! some default for kinds
  !
  INTEGER,   PARAMETER :: dbl = SELECTED_REAL_KIND( 14, 200 )
  !
  !
  TYPE crio_atm_orbital
       !
       INTEGER                :: norb              ! number of atomic orbitals
       CHARACTER(30)          :: units             ! units for the position of the atomic orbitals
       REAL(dbl)              :: coord(3)          ! position of the atomic orbitals
       INTEGER,       POINTER :: seq(:)            ! sequence numbers (index) in the basis array
       CHARACTER(10), POINTER :: orb_type(:)       ! type of the orbitals
       !
       LOGICAL                :: alloc
       !
  END TYPE crio_atm_orbital
  
  !
  ! end of declarations
  !

  PUBLIC :: dbl
  !
  PUBLIC :: crio_atm_orbital
  PUBLIC :: crio_atm_orbital_allocate, crio_atm_orbital_deallocate

CONTAINS

!
!-------------------------------------------
! ... basic (public) subroutines
!-------------------------------------------
!
    !------------------------------------------------------------------------
    SUBROUTINE crio_atm_orbital_allocate( norb, obj, ierr )
      !------------------------------------------------------------------------
      IMPLICIT NONE
      !
      INTEGER,                 INTENT(IN)    :: norb
      TYPE( crio_atm_orbital), INTENT(INOUT) :: obj
      INTEGER,                 INTENT(OUT)   :: ierr
      !
      ierr = 0
      !
      obj%norb = norb
      !
      ALLOCATE( obj%seq( norb ), STAT=ierr )
      IF ( ierr/=0 ) RETURN
      !
      ALLOCATE( obj%orb_type( norb ), STAT=ierr )
      IF ( ierr/=0 ) RETURN
      ! 
      obj%alloc = .TRUE.
      !
    END SUBROUTINE crio_atm_orbital_allocate
    !
    !
    !------------------------------------------------------------------------
    SUBROUTINE crio_atm_orbital_deallocate( obj, ierr )
      !------------------------------------------------------------------------
      IMPLICIT NONE
      !
      TYPE( crio_atm_orbital), INTENT(INOUT) :: obj
      INTEGER,                 INTENT(OUT)   :: ierr
      !
      ierr = 0
      !
      IF (ASSOCIATED(obj%seq) ) THEN
          DEALLOCATE( obj%seq, STAT=ierr )
          IF ( ierr/=0 ) RETURN
      ENDIF
      !
      IF (ASSOCIATED(obj%orb_type) ) THEN
          DEALLOCATE( obj%orb_type, STAT=ierr )
          IF ( ierr/=0 ) RETURN
      ENDIF
      ! 
      obj%alloc = .FALSE.
      !
    END SUBROUTINE crio_atm_orbital_deallocate
    !
END MODULE crystal_io_base_module

