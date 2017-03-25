!
! Copyright (C) 2005 WanT Group, 2017 ERMES Group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License\'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!*********************************************
   MODULE T_hamiltonian_module
   !*********************************************
   !
   USE kinds,                  ONLY : dbl
   USE constants,              ONLY : ZERO
   USE parameters,             ONLY : nstrx
   USE log_module,             ONLY : log_push, log_pop
   USE T_kpoints_module,       ONLY : nkpts_par
   USE T_operator_blc_module
   !
   IMPLICIT NONE
   PRIVATE 
   SAVE
!
! Contains the description of the system in terms of 
! Hamiltonian, Overlap, and correlation Sigma blocks
! 
    ! 
    ! dimensions
    !
    INTEGER                   :: dimL
    INTEGER                   :: dimC
    INTEGER                   :: dimR
    INTEGER                   :: dimx, dimx_lead
    !
    INTEGER                   :: nspin
    INTEGER                   :: ispin
    REAL(dbl)                 :: shift_L
    REAL(dbl)                 :: shift_C
    REAL(dbl)                 :: shift_R
    REAL(dbl)                 :: shift_corr
    !
    TYPE( operator_blc )      :: blc_00L 
    TYPE( operator_blc )      :: blc_01L 
    TYPE( operator_blc )      :: blc_00R 
    TYPE( operator_blc )      :: blc_01R 
    TYPE( operator_blc )      :: blc_00C
    TYPE( operator_blc )      :: blc_LC
    TYPE( operator_blc )      :: blc_CR
    !
    LOGICAL :: alloc = .FALSE.


!
! end delcarations
!

   PUBLIC :: dimL, dimC, dimR, dimx, dimx_lead
   PUBLIC :: nspin, ispin
   PUBLIC :: nkpts_par
   !
   PUBLIC :: shift_L, shift_C, shift_R, shift_corr
   !
   PUBLIC :: blc_00L
   PUBLIC :: blc_01L
   PUBLIC :: blc_00R
   PUBLIC :: blc_01R
   PUBLIC :: blc_00C
   PUBLIC :: blc_LC
   PUBLIC :: blc_CR
   !
   PUBLIC :: alloc
   !
   PUBLIC :: hamiltonian_allocate
   PUBLIC :: hamiltonian_deallocate
   PUBLIC :: hamiltonian_memusage


CONTAINS

!
! subroutines
!

!**********************************************************
   SUBROUTINE hamiltonian_allocate()
   !**********************************************************
   IMPLICIT NONE
      CHARACTER(20)      :: subname="hamiltonian_allocate"

      CALL log_push( subname )

      IF ( alloc )       CALL errore(subname,'already allocated', 1 )
      IF ( dimL <= 0 )   CALL errore(subname,'invalid dimL', 1 )
      IF ( dimR <= 0 )   CALL errore(subname,'invalid dimR', 1 )
      IF ( dimC <= 0 )   CALL errore(subname,'invalid dimC', 1 )
      IF ( nkpts_par <= 0 )   CALL errore(subname,'invalid nkpts_par', 1 )
      !
      dimx      = MAX( dimC, dimR, dimL )
      dimx_lead = MAX( dimR, dimL )

      !
      ! init data
      !
      CALL operator_blc_init( blc_00L, "block_00L")
      CALL operator_blc_init( blc_01L, "block_01L")
      CALL operator_blc_init( blc_00R, "block_00R")
      CALL operator_blc_init( blc_01R, "block_01R")
      CALL operator_blc_init( blc_00C, "block_00C")
      CALL operator_blc_init( blc_LC,  "block_LC")
      CALL operator_blc_init( blc_CR,  "block_CR")

      !
      ! allocations
      !
      CALL operator_blc_allocate( dimL, dimL, nkpts_par, OBJ=blc_00L )
      CALL operator_blc_allocate( dimL, dimL, nkpts_par, OBJ=blc_01L )
      !
      CALL operator_blc_allocate( dimR, dimR, nkpts_par, OBJ=blc_00R )
      CALL operator_blc_allocate( dimR, dimR, nkpts_par, OBJ=blc_01R )
      !
      CALL operator_blc_allocate( dimC, dimC, nkpts_par, OBJ=blc_00C )
      CALL operator_blc_allocate( dimL, dimC, nkpts_par, OBJ=blc_LC )
      CALL operator_blc_allocate( dimC, dimR, nkpts_par, OBJ=blc_CR )
      !
      alloc = .TRUE.

      CALL log_pop( subname )

   END SUBROUTINE hamiltonian_allocate


!**********************************************************
   SUBROUTINE hamiltonian_deallocate()
   !**********************************************************
   IMPLICIT NONE
      CHARACTER(22)      :: subname="hamiltonian_deallocate"

      CALL log_push( subname )

      IF ( .NOT. alloc ) RETURN

      IF ( blc_00L%alloc ) CALL operator_blc_deallocate( OBJ=blc_00L )
      IF ( blc_01L%alloc ) CALL operator_blc_deallocate( OBJ=blc_01L )
      !
      IF ( blc_00R%alloc ) CALL operator_blc_deallocate( OBJ=blc_00R )
      IF ( blc_01R%alloc ) CALL operator_blc_deallocate( OBJ=blc_01R )
      !
      IF ( blc_00C%alloc ) CALL operator_blc_deallocate( OBJ=blc_00C )
      IF ( blc_LC%alloc  ) CALL operator_blc_deallocate( OBJ=blc_LC )
      IF ( blc_CR%alloc  ) CALL operator_blc_deallocate( OBJ=blc_CR )
      !
      alloc = .FALSE.   

      CALL log_pop( subname )

   END SUBROUTINE hamiltonian_deallocate


!**********************************************************
   REAL(dbl) FUNCTION hamiltonian_memusage(memtype)
   !**********************************************************
   IMPLICIT NONE
       !
       CHARACTER(*)   :: memtype
       !
       REAL(dbl) :: cost
       !
       cost = ZERO
       IF ( blc_00L%alloc )    cost = cost + operator_blc_memusage(blc_00L,  memtype)
       IF ( blc_01L%alloc )    cost = cost + operator_blc_memusage(blc_01L,  memtype)
       IF ( blc_00R%alloc )    cost = cost + operator_blc_memusage(blc_00R,  memtype)
       IF ( blc_01R%alloc )    cost = cost + operator_blc_memusage(blc_01R,  memtype)
       IF ( blc_00C%alloc )    cost = cost + operator_blc_memusage(blc_00C,  memtype)
       IF ( blc_LC%alloc )     cost = cost + operator_blc_memusage(blc_LC,   memtype)
       IF ( blc_CR%alloc )     cost = cost + operator_blc_memusage(blc_CR,   memtype)
       !
       hamiltonian_memusage = cost
       !
   END FUNCTION hamiltonian_memusage


END MODULE T_hamiltonian_module

