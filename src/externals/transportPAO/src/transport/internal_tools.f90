!
! Copyright (C) 2010 WanT Group, 2017 ERMES Group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License\'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
! 
! 
!*********************************************
   MODULE internal_tools_module
!*********************************************
   !
   USE kinds,              ONLY : dbl
   USE constants,          ONLY : BOHR => bohr_radius_angs, ZERO, ONE, &
                                  CZERO, CONE, TWO, RYD
   USE parameters,         ONLY : nstrx
   USE timing_module,      ONLY : timing
   USE log_module,         ONLY : log_push, log_pop
   USE converters_module,  ONLY : cart2cry, cry2cart
   USE parser_module,      ONLY : change_case
   USE iotk_module
   !
   IMPLICIT NONE 
   PRIVATE
   SAVE

   !
   ! global variables of the module
   !
   CHARACTER(nstrx)   :: pre_postfix
   CHARACTER(nstrx)   :: file_ham
   CHARACTER(nstrx)   :: file_space
   !
   CHARACTER(nstrx)   :: attr
   !
   LOGICAL            :: init = .FALSE.

   !
   ! contains:
   ! SUBROUTINE  internal_tools_init( pre_postfix_ )
   ! SUBROUTINE  internal_tools_get_prefix( filein, prefix_ )
   ! SUBROUTINE  internal_tools_get_dims( [nbnd, nkpts, dimwann] )
   ! SUBROUTINE  internal_tools_get_eig(  nbnd, nkpts, eig )
   ! SUBROUTINE  internal_tools_get_lattice( alat, avec, bvec )
   ! SUBROUTINE  internal_tools_get_kpoints( nkpts[, vkpt, wk, bvec] )
   ! FUNCTION    file_is_internal( filein )
   !
   PUBLIC :: internal_tools_get_dims
   PUBLIC :: internal_tools_get_eig
   PUBLIC :: internal_tools_get_lattice
   PUBLIC :: internal_tools_get_kpoints
   PUBLIC :: file_is_internal

CONTAINS


!**********************************************************
   SUBROUTINE internal_tools_init( pre_postfix_ )
   !**********************************************************
   !
   ! define module global variables
   !
   IMPLICIT NONE
   CHARACTER(*),   INTENT(IN) :: pre_postfix_
   !
   pre_postfix   = TRIM( pre_postfix_ )
   file_ham      = TRIM( pre_postfix_ ) // '.ham'
   file_space    = TRIM( pre_postfix_ ) // '.space'
   !
   init = .TRUE.
   !
END SUBROUTINE internal_tools_init
   

!**********************************************************
   SUBROUTINE internal_tools_get_prefix( filein, pre_postfix_ )
   !**********************************************************
   !
   ! extract the prefix (basename) of the input file.
   ! If the extension of the file is not ".ham" an
   ! empty prefix is issued
   !
   IMPLICIT NONE
   CHARACTER(*),   INTENT(IN)  :: filein
   CHARACTER(*),   INTENT(OUT) :: pre_postfix_ 
   !
   INTEGER      :: ilen
   CHARACTER(4) :: suffix='.ham'
   !
   pre_postfix_  = ' '
   !
   ilen = LEN_TRIM( filein )
   !
   IF ( filein(ilen-3:ilen) == suffix ) THEN
       !
       pre_postfix_ = filein(1:ilen-4)
       !
   ENDIF
   !
END SUBROUTINE internal_tools_get_prefix
   

!**********************************************************
   SUBROUTINE internal_tools_get_dims( nbnd, nkpts, dimwann, nrtot, nr, nspin, spin_component )
   !**********************************************************
   !
   ! get the dimensions of the problem
   ! need to have the module initialized
   !
   IMPLICIT NONE
   ! 
   INTEGER,      OPTIONAL, INTENT(OUT) :: nbnd, nkpts, dimwann, nrtot, nr(3), nspin
   CHARACTER(*), OPTIONAL, INTENT(OUT) :: spin_component
   !
   CHARACTER(23)  :: subname='internal_tools_get_dims'
   INTEGER        :: nbnd_, nkpts_, dimwann_, iunit
   INTEGER        :: nrtot_, nr_(3), nspin_
   CHARACTER(256) :: spin_component_
   INTEGER        :: ierr
   
   CALL log_push( subname )
   !
   IF ( .NOT. init ) CALL errore(subname,'module not init',10)
   CALL iotk_free_unit( iunit )


   !
   ! get dimensions from .ham
   !
   CALL iotk_open_read( iunit, FILE=TRIM(file_ham), IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'opening '//TRIM(file_ham),ABS(ierr) )
   !
   CALL iotk_scan_begin( iunit, 'HAMILTONIAN', IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'searching for HAMILTONIAN',ABS(ierr) )
   !
   CALL iotk_scan_empty( iunit, 'DATA', ATTR=attr, IERR=ierr)
   IF ( ierr/= 0 ) CALL errore(subname,'scanning for DATA', ABS(ierr))
   !
   CALL iotk_scan_attr(attr, 'dimwann', dimwann_, IERR=ierr)
   IF ( ierr/= 0 ) CALL errore(subname,'scanning for dimwann', ABS(ierr))
   CALL iotk_scan_attr(attr, 'nrtot', nrtot_, IERR=ierr)
   IF ( ierr/= 0 ) CALL errore(subname,'scanning for nrtot_', ABS(ierr))
   CALL iotk_scan_attr(attr, 'nr', nr_, IERR=ierr)
   IF ( ierr/= 0 ) CALL errore(subname,'scanning for nr', ABS(ierr))
   !
   CALL iotk_scan_end( iunit, 'HAMILTONIAN', IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'searching end for HAMILTONIAN',ABS(ierr) )
   !
   CALL iotk_close_read( iunit, IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'closing '//TRIM(file_ham),ABS(ierr) )
   !
   IF ( PRESENT( dimwann ) )  dimwann = dimwann_
   IF ( PRESENT( nrtot ) )      nrtot = nrtot_
   IF ( PRESENT( nr ) )            nr = nr_
  
   !
   ! get dimensions from .space
   !
   CALL iotk_open_read( iunit, FILE=TRIM(file_space), IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'opening '//TRIM(file_space),ABS(ierr) )
   !
   CALL iotk_scan_begin( iunit, 'WINDOWS', IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'searching for HAMILTONIAN',ABS(ierr) )
   !
   CALL iotk_scan_empty( iunit, 'DATA', ATTR=attr, IERR=ierr)
   IF ( ierr/= 0 ) CALL errore(subname,'scanning for DATA', ABS(ierr))
   !
   CALL iotk_scan_attr(attr, 'nbnd', nbnd_, IERR=ierr)
   IF ( ierr/= 0 ) CALL errore(subname,'scanning for nbnd', ABS(ierr))
   CALL iotk_scan_attr(attr, 'nkpts', nkpts_, IERR=ierr)
   IF ( ierr/= 0 ) CALL errore(subname,'scanning for nkpts', ABS(ierr))
   CALL iotk_scan_attr(attr, 'nspin', nspin_, IERR=ierr)
   IF ( ierr/= 0 ) CALL errore(subname,'scanning for nspin', ABS(ierr))
   CALL iotk_scan_attr(attr, 'spin_component', spin_component_, IERR=ierr)
   IF ( ierr/= 0 ) CALL errore(subname,'scanning for spin_component', ABS(ierr))
   !
   CALL iotk_scan_end( iunit, 'WINDOWS', IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'searching end for WINDOWS',ABS(ierr) )
   !
   CALL iotk_close_read( iunit, IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'closing '//TRIM(file_space),ABS(ierr) )
   !
   IF ( PRESENT( nbnd ) )        nbnd = nbnd_
   IF ( PRESENT( nkpts ) )      nkpts = nkpts_
   IF ( PRESENT( nspin ) )      nspin = nspin_
   IF ( PRESENT( spin_component ) ) &
                       spin_component = TRIM(spin_component_)
   !
   CALL log_pop( subname )
   RETURN
   !
END SUBROUTINE internal_tools_get_dims


!**********************************************************
   SUBROUTINE internal_tools_get_eig( nbnd, nkpts, nspin, isp, eig )
   !**********************************************************
   !
   ! read eigenvalues
   !
   IMPLICIT NONE
   ! 
   INTEGER,   INTENT(IN)   :: nbnd, nkpts, nspin, isp
   REAL(dbl), INTENT(OUT)  :: eig(nbnd,nkpts)
   !
   CHARACTER(22) :: subname='internal_tools_get_eig'
   INTEGER       :: iunit
   INTEGER       :: ierr
   
   CALL log_push( subname )
   !
   IF ( .NOT. init ) CALL errore(subname,'module not init',10)
   CALL iotk_free_unit( iunit )

   !
   ! get dimensions from .space
   !
   CALL iotk_open_read( iunit, FILE=TRIM(file_space), IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'opening '//TRIM(file_space),ABS(ierr) )
   !
   CALL iotk_scan_begin( iunit, 'WINDOWS', IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'searching for HAMILTONIAN',ABS(ierr) )
   !
   IF ( nspin == 2 ) THEN
       CALL iotk_scan_begin( iunit, 'SPIN'//TRIM(iotk_index(isp)), IERR=ierr )
       IF ( ierr/=0 ) CALL errore(subname,'searching for SPIN'//TRIM(iotk_index(isp)) ,ABS(ierr) )
   ENDIF
   !
   CALL iotk_scan_dat( iunit, 'EIG', eig, IERR=ierr)
   IF ( ierr/=0 ) CALL errore(subname,'searching end for EIG',ABS(ierr) )
   !
   IF ( nspin == 2 ) THEN
       CALL iotk_scan_end( iunit, 'SPIN'//TRIM(iotk_index(isp)), IERR=ierr )
       IF ( ierr/=0 ) CALL errore(subname,'searching end for SPIN'//TRIM(iotk_index(isp)) ,ABS(ierr) )
   ENDIF
   !
   CALL iotk_scan_end( iunit, 'WINDOWS', IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'searching end for WINDOWS',ABS(ierr) )
   !
   CALL iotk_close_read( iunit, IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'closing '//TRIM(file_space),ABS(ierr) )
   !
   !
   CALL log_pop( subname )
   RETURN
   !
END SUBROUTINE internal_tools_get_eig

   
!**********************************************************
   SUBROUTINE internal_tools_get_lattice( alat, avec, bvec )
   !**********************************************************
   !
   ! read lattice data from internal files
   ! avec in bohr units
   ! bvec in bohr^-1 units
   !
   IMPLICIT NONE
   ! 
   REAL(dbl),    INTENT(OUT)  :: alat, avec(3,3), bvec(3,3)
   !
   CHARACTER(26) :: subname='internal_tools_get_lattice'
   INTEGER       :: iunit
   INTEGER       :: ierr
   

   CALL log_push( subname )
   !
   IF ( .NOT. init ) CALL errore(subname,'module not init',10)
   CALL iotk_free_unit( iunit )
   
   !
   ! get data from .ham
   !
   CALL iotk_open_read( iunit, FILE=TRIM(file_ham), IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'opening '//TRIM(file_ham),ABS(ierr) )
   !
   CALL iotk_scan_begin( iunit, 'HAMILTONIAN', IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'searching for HAMILTONIAN',ABS(ierr) )
   !
   CALL iotk_scan_dat( iunit, 'DIRECT_LATTICE', avec, IERR=ierr)
   IF ( ierr/=0 ) CALL errore(subname,'searching for DIRECT_LATTICE',ABS(ierr) )
   !
   CALL iotk_scan_dat( iunit, 'RECIPROCAL_LATTICE', bvec, IERR=ierr)
   IF ( ierr/=0 ) CALL errore(subname,'searching for RECIPROCAL_LATTICE',ABS(ierr) )
   !
   CALL iotk_scan_end( iunit, 'HAMILTONIAN', IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'searching end for HAMILTONIAN',ABS(ierr) )
   !
   CALL iotk_close_read( iunit, IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'closing '//TRIM(file_ham),ABS(ierr) )

   !
   ! init alat
   !
   alat = DOT_PRODUCT( avec(:,1), avec(:,1) )
   alat = SQRT( alat )
   !
   CALL log_pop( subname )
   RETURN
   !
END SUBROUTINE internal_tools_get_lattice


!**********************************************************
   SUBROUTINE internal_tools_get_kpoints( nkpts, vkpt, wk, bvec )
   !**********************************************************
   !
   ! read kpts data. vkpt is in crystal units
   !
   IMPLICIT NONE
   ! 
   INTEGER,             INTENT(IN)  :: nkpts
   REAL(dbl), OPTIONAL, INTENT(OUT) :: vkpt(3,nkpts)
   REAL(dbl), OPTIONAL, INTENT(OUT) :: wk(nkpts)
   REAL(dbl), OPTIONAL, INTENT(OUT) :: bvec(3,3)
   !
   CHARACTER(26) :: subname='internal_tools_get_kpoints'
   REAL(dbl)     :: bvec_(3,3)
   INTEGER       :: iunit
   INTEGER       :: ierr
   
   CALL log_push( subname )
   !
   IF ( .NOT. init ) CALL errore(subname,'module not init',10)
   CALL iotk_free_unit( iunit )

   !
   ! get data from .space
   !
   CALL iotk_open_read( iunit, FILE=TRIM(file_ham), IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'opening '//TRIM(file_ham),ABS(ierr) )
   !
   CALL iotk_scan_begin( iunit, 'HAMILTONIAN', IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'searching for HAMILTONIAN',ABS(ierr) )
   !
   CALL iotk_scan_dat( iunit, 'RECIPROCAL_LATTICE', bvec_, IERR=ierr)
   IF ( ierr/=0 ) CALL errore(subname,'searching for RECIPROCAL_LATTICE',ABS(ierr) )
   !
   IF ( PRESENT( vkpt ) ) THEN
       CALL iotk_scan_dat( iunit, 'VKPT', vkpt, IERR=ierr)
       IF ( ierr/=0 ) CALL errore(subname,'searching for VKPT',ABS(ierr) )
   ENDIF
   !
   IF ( PRESENT( wk ) ) THEN
       CALL iotk_scan_dat( iunit, 'WK', wk, IERR=ierr)
       IF ( ierr/=0 ) CALL errore(subname,'searching for WK',ABS(ierr) )
   ENDIF
   !
   CALL iotk_scan_end( iunit, 'HAMILTONIAN', IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'searching end for HAMILTONIAN',ABS(ierr) )
   !
   CALL iotk_close_read( iunit, IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'closing '//TRIM(file_ham),ABS(ierr) )

   IF ( PRESENT( bvec) )    bvec = bvec_
   !
   CALL log_pop( subname )
   RETURN
   !
END SUBROUTINE internal_tools_get_kpoints


!**********************************************************
   LOGICAL FUNCTION file_is_internal( filename )
   !**********************************************************
   !
   ! check for internal fmt
   !
   CHARACTER(*) :: filename
   !
   CHARACTER(256) :: prefix_
   LOGICAL   :: lerror, lopnd
   INTEGER   :: ierr, iunit
     !
     CALL iotk_free_unit( iunit )
     !
     file_is_internal = .FALSE.
     lerror = .FALSE.
     !
     CALL internal_tools_get_prefix( filename, prefix_ )
     !    
     IF ( LEN_TRIM( prefix_ ) /= 0 ) THEN 
         CALL internal_tools_init( prefix_ )
     ELSE 
         lerror = .TRUE.
         RETURN
     ENDIF
     !
     CALL iotk_open_read( iunit, TRIM(filename), IERR=ierr )
     IF ( ierr /= 0 ) lerror = .TRUE.
     !
     CALL iotk_scan_begin( iunit, "HAMILTONIAN", IERR=ierr )
     IF ( ierr /= 0 ) lerror = .TRUE.
     !
     CALL iotk_scan_end( iunit, "HAMILTONIAN", IERR=ierr )
     IF ( ierr /= 0 ) lerror = .TRUE.
     !
     CALL iotk_close_read( iunit, IERR=ierr )
     IF ( ierr /= 0 ) lerror = .TRUE.
     !
     !
     IF ( lerror ) THEN
         !
         INQUIRE( iunit, OPENED=lopnd )
         IF( lopnd ) CLOSE( iunit )
         !
         RETURN
         !
     ENDIF
     !
     file_is_internal = .TRUE.
     !
  END FUNCTION file_is_internal


END MODULE internal_tools_module

