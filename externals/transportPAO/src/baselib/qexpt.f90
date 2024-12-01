!
! Copyright (C) 2006 WanT Group, 2017 ERMES Group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!----------------------------------------------------------------------------
MODULE qexpt_module
  !----------------------------------------------------------------------------
  !
  ! This module contains some common subroutines used to read
  ! data produced by the pw_export.x utility in Quantum-espresso
  !
  ! written by Andrea Ferretti (2006)
  !
  USE kinds,     ONLY : dbl
  USE constants, ONLY : e2
  USE iotk_module
  IMPLICIT NONE
  !
  PRIVATE
  SAVE
  !
  ! definitions for the fmt
  !
  CHARACTER(5), PARAMETER :: fmt_name = "QEXPT"
  CHARACTER(5), PARAMETER :: fmt_version = "1.1.0"
  !
  ! internal data to be set
  !
  CHARACTER(256)   :: datadir
  INTEGER          :: iunpun
  !
  ! end of declarations
  !
  PUBLIC :: qexpt_init,  qexpt_openfile, qexpt_closefile
  !
  PUBLIC :: qexpt_read_header, qexpt_read_cell, qexpt_read_ions,              &
            qexpt_read_symmetry, qexpt_read_spin, qexpt_read_bz,              &
            qexpt_read_bands, qexpt_read_planewaves, qexpt_read_gk,           &
            qexpt_read_wfc
  !
  CHARACTER(iotk_attlenx) :: attr
  !
CONTAINS
  !
!
!-------------------------------------------
! ... basic (public) subroutines
!-------------------------------------------
!
    !------------------------------------------------------------------------
    SUBROUTINE qexpt_init( iunpun_, datadir_ )
      !------------------------------------------------------------------------
      !
      ! just init module data
      !
      IMPLICIT NONE
      INTEGER,           INTENT(IN) :: iunpun_
      CHARACTER(*),      INTENT(IN) :: datadir_
      !
      iunpun  = iunpun_
      datadir = TRIM(datadir_)
      !
    END SUBROUTINE qexpt_init


    !------------------------------------------------------------------------
    SUBROUTINE qexpt_openfile( filename, action, binary, ierr)
      !------------------------------------------------------------------------
      !
      ! open data file
      !
      IMPLICIT NONE
      !
      CHARACTER(*),       INTENT(IN)  :: filename
      CHARACTER(*),       INTENT(IN)  :: action      ! ("read"|"write")
      LOGICAL, OPTIONAL,  INTENT(IN)  :: binary
      INTEGER,            INTENT(OUT) :: ierr
      !
      LOGICAL  :: binary_ 
     
      ierr = 0
      binary_ = .FALSE.
      IF ( PRESENT(binary) ) binary_ = binary
     
      !
      SELECT CASE ( TRIM(action) )
      CASE ( "read", "READ" )
          !
          CALL iotk_open_read ( iunpun, FILE = TRIM(filename), IERR=ierr )
          !
      CASE ( "write", "WRITE" )
          !
          CALL iotk_open_write( iunpun, FILE = TRIM(filename), BINARY=binary_, IERR=ierr )
          !
      CASE DEFAULT
          ierr = 1
      END SELECT
          
    END SUBROUTINE qexpt_openfile
      

    !------------------------------------------------------------------------
    SUBROUTINE qexpt_closefile( action, ierr)
      !------------------------------------------------------------------------
      !
      ! close data file
      !
      IMPLICIT NONE
      !
      CHARACTER(*),  INTENT(IN)  :: action      ! ("read"|"write")
      INTEGER,       INTENT(OUT) :: ierr
      !
      ierr = 0
      !
      SELECT CASE ( TRIM(action) )
      CASE ( "read", "READ" )
          !
          CALL iotk_close_read( iunpun, IERR=ierr )
          !
      CASE ( "write", "WRITE" )
          !
          CALL iotk_close_write( iunpun, IERR=ierr )
          !
      CASE DEFAULT
          ierr = 2
      END SELECT
      !
    END SUBROUTINE qexpt_closefile

!
!-------------------------------------------
! ... read subroutines
!-------------------------------------------
!
    !
    !------------------------------------------------------------------------
    SUBROUTINE qexpt_read_header( creator_name, creator_version, &
                                  format_name, format_version, ierr )
      !------------------------------------------------------------------------
      !
      IMPLICIT NONE
      CHARACTER(LEN=*),  OPTIONAL, INTENT(OUT) :: creator_name, creator_version
      CHARACTER(LEN=*),  OPTIONAL, INTENT(OUT) :: format_name, format_version
      INTEGER,           OPTIONAL, INTENT(OUT) :: ierr

      CHARACTER(256) :: creator_name_, creator_version_
      CHARACTER(256) :: format_name_,     format_version_

      ierr = 0
      !
      !
      CALL iotk_scan_begin( iunpun, "Header", IERR=ierr )
      IF (ierr/=0) RETURN
      !
      CALL iotk_scan_empty( iunpun, "format", ATTR=attr, IERR=ierr )
      IF (ierr/=0) RETURN
      !
      CALL iotk_scan_attr(attr, "name", format_name_, IERR=ierr)
      IF (ierr/=0) RETURN
      CALL iotk_scan_attr(attr, "version", format_version_, IERR=ierr )
      IF (ierr/=0) RETURN
      !
      CALL iotk_scan_empty( iunpun, "creator", ATTR=attr, IERR=ierr )
      IF (ierr/=0) RETURN
      !
      CALL iotk_scan_attr(attr, "name", creator_name_, IERR=ierr)
      IF (ierr/=0) RETURN
      CALL iotk_scan_attr(attr, "version", creator_version_, IERR=ierr )
      IF (ierr/=0) RETURN
      !
      CALL iotk_scan_end( iunpun, "Header", IERR=ierr )
      IF (ierr/=0) RETURN
      !
      !
      IF ( PRESENT(creator_name) )     creator_name    = TRIM(creator_name_)
      IF ( PRESENT(creator_version) )  creator_version = TRIM(creator_version_)
      IF ( PRESENT(format_name) )      format_name     = TRIM(format_name_)
      IF ( PRESENT(format_version) )   format_version  = TRIM(format_version_)
      !
    END SUBROUTINE qexpt_read_header
    !
    !
    !------------------------------------------------------------------------
    SUBROUTINE qexpt_read_cell( alat, a1, a2, a3, b1, b2, b3, ierr )
      !------------------------------------------------------------------------
      !
      REAL(dbl), OPTIONAL, INTENT(OUT) :: alat
      REAL(dbl), OPTIONAL, INTENT(OUT) :: a1(3), a2(3), a3(3)
      REAL(dbl), OPTIONAL, INTENT(OUT) :: b1(3), b2(3), b3(3)
      INTEGER,             INTENT(OUT) :: ierr
      !
      REAL(dbl) :: alat_
      REAL(dbl) :: a1_(3), a2_(3), a3_(3)
      REAL(dbl) :: b1_(3), b2_(3), b3_(3)
      !
      ierr = 0
      !
      CALL iotk_scan_begin(iunpun,"Cell",IERR=ierr)
      IF (ierr/=0) RETURN  

      CALL iotk_scan_empty(iunpun,"Data",ATTR=attr,IERR=ierr)
      IF (ierr/=0) RETURN
      CALL iotk_scan_attr(attr,"alat",alat_,IERR=ierr)
      IF (ierr/=0) RETURN
      !
      CALL iotk_scan_empty(iunpun,"a1",ATTR=attr,IERR=ierr)
      IF (ierr/=0) RETURN
      CALL iotk_scan_attr(attr,"xyz",a1_,IERR=ierr)
      IF (ierr/=0) RETURN
      CALL iotk_scan_empty(iunpun,"a2",ATTR=attr,IERR=ierr)
      IF (ierr/=0) RETURN
      CALL iotk_scan_attr(attr,"xyz",a2_,IERR=ierr)
      IF (ierr/=0) RETURN
      CALL iotk_scan_empty(iunpun,"a3",ATTR=attr,IERR=ierr)
      IF (ierr/=0) RETURN
      CALL iotk_scan_attr(attr,"xyz",a3_,IERR=ierr)
      IF (ierr/=0) RETURN
      !
      CALL iotk_scan_empty(iunpun,"b1",ATTR=attr,IERR=ierr)
      IF (ierr/=0) RETURN
      CALL iotk_scan_attr(attr,"xyz",b1_,IERR=ierr)
      IF (ierr/=0) RETURN
      CALL iotk_scan_empty(iunpun,"b2",ATTR=attr,IERR=ierr)
      IF (ierr/=0) RETURN
      CALL iotk_scan_attr(attr,"xyz",b2_,IERR=ierr)
      IF (ierr/=0) RETURN
      CALL iotk_scan_empty(iunpun,"b3",ATTR=attr,IERR=ierr)
      IF (ierr/=0) RETURN
      CALL iotk_scan_attr(attr,"xyz",b3_,IERR=ierr)
      IF (ierr/=0) RETURN
      !
      CALL iotk_scan_end(iunpun,"Cell",IERR=ierr)
      IF (ierr/=0) RETURN
      !
      IF ( PRESENT( alat ) )    alat = alat_
      IF ( PRESENT( a1 ) )      a1   = a1_
      IF ( PRESENT( a2 ) )      a2   = a2_
      IF ( PRESENT( a3 ) )      a3   = a3_
      IF ( PRESENT( b1 ) )      b1   = b1_
      IF ( PRESENT( b2 ) )      b2   = b2_
      IF ( PRESENT( b3 ) )      b3   = b3_

    END SUBROUTINE qexpt_read_cell

    !
    !------------------------------------------------------------------------
    SUBROUTINE qexpt_read_ions( nsp, nat, atm, symb, pseudo_dir, psfile, tau, ierr )
      !------------------------------------------------------------------------
      !
      INTEGER,          OPTIONAL, INTENT(OUT) :: nsp, nat
      CHARACTER(LEN=*), OPTIONAL, INTENT(OUT) :: atm(:), symb(:) 
      CHARACTER(LEN=*), OPTIONAL, INTENT(OUT) :: pseudo_dir, psfile(:)
      REAL(dbl),        OPTIONAL, INTENT(OUT) :: tau(:,:)
      INTEGER,                    INTENT(OUT) :: ierr
      !
      INTEGER                     :: nat_, nsp_
      CHARACTER(256)              :: pseudo_dir_
      CHARACTER(3),   ALLOCATABLE :: atm_(:), symb_(:)       
      CHARACTER(256), ALLOCATABLE :: psfile_(:)       
      REAL(dbl),      ALLOCATABLE :: tau_(:,:)
      !
      REAL(dbl) :: alat      
      INTEGER   :: i

      !
      ierr=0
      !
      ! ... this is just to get alat for internal units conversion
      !
      CALL iotk_scan_begin( iunpun, "Cell", IERR=ierr )
      IF (ierr/=0) RETURN
      !
      CALL iotk_scan_empty( iunpun, "Data", ATTR=attr, IERR=ierr )
      IF (ierr/=0) RETURN
      CALL iotk_scan_attr( attr, "alat", alat, IERR=ierr )
      IF (ierr/=0) RETURN
      !
      CALL iotk_scan_end( iunpun, "Cell", IERR=ierr )
      IF (ierr/=0) RETURN

      !
      ! ... start reading actual ION data
      !
      CALL iotk_scan_begin( iunpun, "Atoms", IERR=ierr )
      IF (ierr/=0) RETURN
      !
      CALL iotk_scan_empty( iunpun, "Data", ATTR=attr, IERR=ierr )
      IF (ierr/=0) RETURN
      !
      CALL iotk_scan_attr( attr, "natoms", nat_ )
      IF (ierr/=0) RETURN
      !
      CALL iotk_scan_attr( attr, "nspecies", nsp_ )
      IF (ierr/=0) RETURN
      !
      IF ( PRESENT(nat) )   nat = nat_
      IF ( PRESENT(nsp) )   nsp = nsp_
      ! 
      !
      ALLOCATE( tau_(3,nat_) ) 
      ALLOCATE( symb_(nat_) ) 
      !
      CALL iotk_scan_begin( iunpun, "Positions", IERR=ierr )
      IF (ierr/=0) RETURN
      !
      DO i = 1, nat_
         !
         CALL iotk_scan_empty( iunpun, &
                               "atom" // TRIM( iotk_index(i) ), ATTR=attr, IERR=ierr )
         IF (ierr/=0) RETURN
         !
         CALL iotk_scan_attr( attr, "type",   symb_(i),   IERR=ierr )
         IF (ierr/=0) RETURN
         CALL iotk_scan_attr( attr, "xyz",    tau_(:,i), IERR=ierr )
         IF (ierr/=0) RETURN
         !
      ENDDO
      !
      CALL iotk_scan_end( iunpun, "Positions", IERR=ierr )
      IF (ierr/=0) RETURN
      !
      ALLOCATE( psfile_(nsp_) ) 
      ALLOCATE( atm_(nsp_) ) 
      !
      CALL iotk_scan_begin( iunpun, "Types", IERR=ierr )
      IF (ierr/=0) RETURN
      !
      CALL iotk_scan_empty( iunpun, "Data", ATTR=attr, IERR=ierr )
      IF (ierr/=0) RETURN
      !
      CALL iotk_scan_attr( attr, "pseudo_dir", pseudo_dir_, IERR=ierr )
      IF (ierr/=0) RETURN
      !
      DO i = 1, nsp_
         !
         CALL iotk_scan_empty( iunpun, "specie"//TRIM( iotk_index(i) ), ATTR=attr, IERR=ierr )
         IF (ierr/=0) RETURN
         !
         CALL iotk_scan_attr( attr, "type", atm_(i), IERR=ierr )
         IF (ierr/=0) RETURN
         CALL iotk_scan_attr( attr, "pseudo_file", psfile_(i), IERR=ierr )
         IF (ierr/=0) RETURN
         !
      ENDDO
      !
      CALL iotk_scan_end( iunpun, "Types", IERR=ierr )
      IF (ierr/=0) RETURN
      !
      CALL iotk_scan_end( iunpun, "Atoms", IERR=ierr )
      IF (ierr/=0) RETURN
      !
      !
      ! ... convert also from alat to bohr
      !
      IF ( PRESENT(nsp) )         nsp    = nsp_
      IF ( PRESENT(nat) )         nat    = nat_
      IF ( PRESENT(tau) )         tau(1:3, 1:nat_)    = tau_ * alat
      IF ( PRESENT(symb) )        symb(1:nat_)        = symb_
      IF ( PRESENT(atm) )         atm(1:nsp_)         = atm_
      IF ( PRESENT(psfile) )      psfile(1:nsp_)      = psfile_
      IF ( PRESENT(pseudo_dir) )  pseudo_dir          = TRIM(pseudo_dir_)
      !
      DEALLOCATE( atm_ )
      DEALLOCATE( psfile_ )
      DEALLOCATE( symb_ )
      DEALLOCATE( tau_ )
     
    END SUBROUTINE qexpt_read_ions


    !------------------------------------------------------------------------
    SUBROUTINE qexpt_read_symmetry( nsym, invsym, s, trasl, sname, ierr )
      !------------------------------------------------------------------------
      !
      INTEGER,          OPTIONAL, INTENT(OUT) :: nsym
      LOGICAL,          OPTIONAL, INTENT(OUT) :: invsym
      INTEGER,          OPTIONAL, INTENT(OUT) :: s(:,:,:)
      REAL(dbl),        OPTIONAL, INTENT(OUT) :: trasl(:,:)
      CHARACTER(*),     OPTIONAL, INTENT(OUT) :: sname(:)
      INTEGER,                    INTENT(OUT) :: ierr
      !
      INTEGER        :: nsym_
      LOGICAL        :: invsym_
      INTEGER        :: s_(3,3,48)
      REAL(dbl)      :: trasl_(3, 48)
      CHARACTER(256) :: sname_(48)
      !
      INTEGER        :: i

      !
      ierr=0
      !
      !
      CALL iotk_scan_begin( iunpun, "Symmetry", IERR=ierr )
      IF (ierr/=0) RETURN
      !
      CALL iotk_scan_empty( iunpun, "symmops", ATTR=attr, IERR=ierr )
      IF (ierr/=0) RETURN
      !
      CALL iotk_scan_attr( attr, "nsym", nsym_, IERR=ierr )
      IF (ierr/=0) RETURN
      CALL iotk_scan_attr( attr, "invsym", invsym_, IERR=ierr )
      IF (ierr/=0) RETURN
      !
      !
      DO i = 1, nsym_
           !
           CALL iotk_scan_empty( iunpun, "info"//TRIM( iotk_index(i)), ATTR=attr, IERR=ierr )
           IF (ierr/=0) RETURN
           CALL iotk_scan_attr( attr, "name", sname_(i), IERR=ierr )
           IF (ierr/=0) RETURN
           !
           CALL iotk_scan_dat( iunpun, "sym"//TRIM( iotk_index(i)), s_(:,:,i), IERR=ierr )
           IF (ierr/=0) RETURN
           !
           CALL iotk_scan_dat( iunpun, "trasl"//TRIM( iotk_index(i)), &
                                        trasl_(:,i), IERR=ierr )
           IF (ierr/=0) RETURN
           !
      ENDDO
      !
      CALL iotk_scan_end( iunpun, "Symmetry", IERR=ierr )
      IF (ierr/=0) RETURN
      !
      !
      IF ( PRESENT( nsym ) )       nsym      = nsym_
      IF ( PRESENT( invsym ) )     invsym    = invsym_
      IF ( PRESENT( sname ) )      sname( 1:nsym_)        = sname_ ( 1:nsym)
      IF ( PRESENT( s ) )          s( 1:3, 1:3, 1:nsym_)  = s_ ( :, :, 1:nsym_)
      IF ( PRESENT( trasl ) )      trasl(  1:3, 1:nsym_)  = trasl_( :, 1:nsym_)
      !
    END SUBROUTINE qexpt_read_symmetry


    !------------------------------------------------------------------------
    SUBROUTINE qexpt_read_planewaves( ecutwfc, ecutrho, npwx, &
                                      nr1, nr2, nr3, ngm, igv, cutoff_units, ierr )
      !------------------------------------------------------------------------
      !
      INTEGER,      OPTIONAL, INTENT(OUT) :: npwx, nr1, nr2, nr3, ngm
      INTEGER,      OPTIONAL, INTENT(OUT) :: igv(:,:)
      REAL(dbl),    OPTIONAL, INTENT(OUT) :: ecutwfc, ecutrho
      CHARACTER(*), OPTIONAL, INTENT(OUT) :: cutoff_units
      INTEGER,                INTENT(OUT) :: ierr
      !
      INTEGER        :: npwx_, nr1_, nr2_, nr3_, ngm_
      REAL(dbl)      :: ecutwfc_, ecutrho_
      CHARACTER(256) :: cutoff_units_
      !

      ierr = 0
      !
      ! ... dimensions
      !
      CALL iotk_scan_begin( iunpun, "Other_parameters", IERR=ierr )
      IF (ierr/=0) RETURN
      !
      CALL iotk_scan_empty( iunpun, "Cutoff", ATTR=attr, IERR=ierr)
      IF (ierr/=0) RETURN
      !
      CALL iotk_scan_attr( attr, "wfc", ecutwfc_ ,IERR=ierr)
      IF (ierr/=0) RETURN
      CALL iotk_scan_attr( attr, "rho", ecutrho_ ,IERR=ierr)
      IF (ierr/=0) RETURN
      CALL iotk_scan_attr( attr, "units", cutoff_units_, IERR=ierr)
      IF (ierr/=0) RETURN
      !
      CALL iotk_scan_empty( iunpun, "Space_grid", ATTR=attr, IERR=ierr)
      IF (ierr/=0) RETURN
      !
      CALL iotk_scan_attr( attr, "nr1", nr1_, IERR=ierr)
      IF (ierr/=0) RETURN
      CALL iotk_scan_attr( attr, "nr2", nr2_, IERR=ierr)
      IF (ierr/=0) RETURN
      CALL iotk_scan_attr( attr, "nr3", nr3_, IERR=ierr)
      IF (ierr/=0) RETURN
      !
      CALL iotk_scan_end( iunpun, 'Other_parameters', IERR=ierr)
      IF (ierr/=0) RETURN

      !
      !
      ! ... Main G grid (density)
      !
      CALL iotk_scan_begin( iunpun, "Main_grid", ATTR=attr, IERR=ierr)
      IF (ierr/=0) RETURN
      !
      CALL iotk_scan_attr( attr, "npw", ngm_, IERR=ierr)
      IF (ierr/=0) RETURN
      !
      IF ( PRESENT( igv ) ) THEN 
          !
          CALL iotk_scan_dat( iunpun, "g", igv(1:3, 1:ngm_) , IERR=ierr)
          IF (ierr/=0) RETURN
          !
      ENDIF
      !
      CALL iotk_scan_end( iunpun, "Main_grid", IERR=ierr)
      IF (ierr/=0) RETURN
      !
      CALL iotk_scan_begin( iunpun, "Wfc_grids", ATTR=attr, IERR=ierr)
      IF (ierr/=0) RETURN
      CALL iotk_scan_attr( attr, "npwx", npwx_, IERR=ierr)
      IF (ierr/=0) RETURN
      CALL iotk_scan_end( iunpun, "Wfc_grids", IERR=ierr)
      IF (ierr/=0) RETURN
      !
      !
      IF ( PRESENT( ecutwfc ) )           ecutwfc      = ecutwfc_
      IF ( PRESENT( ecutrho ) )           ecutrho      = ecutrho_
      IF ( PRESENT( npwx ) )              npwx         = npwx_
      IF ( PRESENT( nr1 ) )               nr1          = nr1_
      IF ( PRESENT( nr2 ) )               nr2          = nr2_
      IF ( PRESENT( nr3 ) )               nr3          = nr3_
      IF ( PRESENT( ngm ) )               ngm          = ngm_
      IF ( PRESENT( cutoff_units ) )      cutoff_units = TRIM( cutoff_units_ )
      !
    END SUBROUTINE qexpt_read_planewaves


    !------------------------------------------------------------------------
    SUBROUTINE qexpt_read_gk( ik, npwk, index, igk, ierr )
      !------------------------------------------------------------------------
      !
      INTEGER,                INTENT(IN)  :: ik
      INTEGER,      OPTIONAL, INTENT(OUT) :: npwk
      INTEGER,      OPTIONAL, INTENT(OUT) :: igk(:,:), index(:)
      INTEGER,                INTENT(OUT) :: ierr
      !
      CHARACTER(256) :: filename
      INTEGER :: iungk
      INTEGER :: npwk_
      !
      
      ierr = 0
      CALL iotk_free_unit( iungk )
      filename = TRIM(datadir) // '/' // 'grid' //TRIM( iotk_index(ik) )
      !
      CALL iotk_open_read ( iungk, FILE = TRIM(filename), ATTR=attr, IERR=ierr )
      IF (ierr/=0)  RETURN
      !
      CALL iotk_scan_attr( attr, 'npw', npwk_, IERR=ierr)
      IF (ierr/=0)  RETURN
      !
      IF ( PRESENT( index ) ) THEN
          !
          CALL iotk_scan_dat( iungk, 'index', index(1:npwk_), IERR=ierr)
          IF (ierr/=0)  RETURN
          !
      ENDIF
      !
      IF ( PRESENT( igk ) ) THEN
          !
          CALL iotk_scan_dat( iungk, 'grid', igk(1:3, 1:npwk_), IERR=ierr)
          IF (ierr/=0)  RETURN
          !
      ENDIF
      !
      CALL iotk_close_read ( iungk, IERR=ierr )
      IF (ierr/=0)  RETURN
      !
      !
      IF ( PRESENT( npwk ) )       npwk  = npwk_
      !
    END SUBROUTINE qexpt_read_gk

    !
    !------------------------------------------------------------------------
    SUBROUTINE qexpt_read_spin( nspin, nkstot, ierr )
      !------------------------------------------------------------------------
      !
      ! nkstot is the total number of kpts which is double of the actual
      ! number of kpts if nspin == 2 (according to espresso internal convention)
      !
      INTEGER,          OPTIONAL, INTENT(OUT) :: nspin, nkstot
      INTEGER,                    INTENT(OUT) :: ierr
      !
      INTEGER                     :: nspin_,  nkstot_
      !

      ierr = 0
      !
      CALL iotk_scan_begin( iunpun, "Dimensions", IERR=ierr )
      IF (ierr/=0) RETURN
      !
      CALL iotk_scan_empty( iunpun, "Kpoints", ATTR=attr, IERR=ierr )
      IF (ierr/=0) RETURN
      CALL iotk_scan_attr( attr, "nspin", nspin_, IERR=ierr )
      IF (ierr/=0) RETURN
      CALL iotk_scan_attr( attr, "nktot", nkstot_, IERR=ierr )
      IF (ierr/=0) RETURN
      !
      CALL iotk_scan_end( iunpun, "Dimensions", IERR=ierr )
      IF (ierr/=0) RETURN
      !
      IF ( PRESENT( nspin ) )      nspin  = nspin_
      IF ( PRESENT( nkstot ) )     nkstot = nkstot_
      ! 
    END SUBROUTINE qexpt_read_spin


   !------------------------------------------------------------------------
    SUBROUTINE qexpt_read_bz( num_k_points, xk, wk, k1, k2, k3, nk1, nk2, nk3, ierr )
      !------------------------------------------------------------------------
      !
      INTEGER,   OPTIONAL, INTENT(OUT) :: num_k_points, k1, k2, k3, nk1, nk2, nk3
      REAL(dbl), OPTIONAL, INTENT(OUT) :: xk(:,:), wk(:)
      INTEGER,             INTENT(OUT) :: ierr
      !
      INTEGER                :: num_k_points_, k1_, k2_, k3_, nk1_, nk2_, nk3_, nspin_
      REAL(dbl), ALLOCATABLE :: xk_(:,:), wk_(:)
      !

      ierr = 0
      !
      ! first need to know whether nkpts is doubled due to nspin == 2
      !
      CALL iotk_scan_begin( iunpun, "Dimensions", IERR=ierr )
      IF ( ierr/=0 ) RETURN
      !
      CALL iotk_scan_empty( iunpun, "Kpoints", ATTR=attr, IERR=ierr )
      IF ( ierr/=0 ) RETURN
      !
      CALL iotk_scan_attr( attr, "nspin", nspin_, IERR=ierr )
      IF ( ierr/=0 ) RETURN
      CALL iotk_scan_attr( attr, "nktot", num_k_points_, IERR=ierr )
      IF ( ierr/=0 ) RETURN
      CALL iotk_scan_attr( attr, "nk1", nk1_, IERR=ierr )
      IF ( ierr/=0 ) RETURN
      CALL iotk_scan_attr( attr, "nk2", nk2_, IERR=ierr )
      IF ( ierr/=0 ) RETURN
      CALL iotk_scan_attr( attr, "nk3", nk3_, IERR=ierr )
      IF ( ierr/=0 ) RETURN
      CALL iotk_scan_attr( attr, "s1", k1_, IERR=ierr )
      IF ( ierr/=0 ) RETURN
      CALL iotk_scan_attr( attr, "s2", k2_, IERR=ierr )
      IF ( ierr/=0 ) RETURN
      CALL iotk_scan_attr( attr, "s3", k3_, IERR=ierr )
      IF ( ierr/=0 ) RETURN
      !
      CALL iotk_scan_end( iunpun, "Dimensions", IERR=ierr )
      IF ( ierr/=0 ) RETURN
      !
      SELECT CASE ( nspin_ ) 
      CASE ( 1 )
         ! do nothing
      CASE ( 2 )
         num_k_points_ = num_k_points_ / nspin_
      CASE DEFAULT
         ierr = 1
         RETURN
      END SELECT
      !
      !
      ALLOCATE ( xk_(3, nspin_ * num_k_points_)   )
      ALLOCATE ( wk_(   nspin_ * num_k_points_)   )
      !
      CALL iotk_scan_begin( iunpun, "Kmesh", IERR=ierr )
      IF ( ierr/=0 ) RETURN
      !
      CALL iotk_scan_dat( iunpun, "weights", wk_, IERR=ierr )
      IF ( ierr/=0 ) RETURN
      !
      CALL iotk_scan_dat( iunpun, "k", xk_, IERR=ierr )
      IF ( ierr/=0 ) RETURN
      !
      CALL iotk_scan_end( iunpun, "Kmesh", IERR=ierr )
      IF ( ierr/=0 ) RETURN
      !
      !
      IF ( PRESENT( num_k_points ) )       num_k_points  = num_k_points_
      IF ( PRESENT( nk1 ) )                nk1           = nk1_
      IF ( PRESENT( nk2 ) )                nk2           = nk2_
      IF ( PRESENT( nk3 ) )                nk3           = nk3_
      IF ( PRESENT( k1 ) )                  k1           =  k1_
      IF ( PRESENT( k2 ) )                  k2           =  k2_
      IF ( PRESENT( k3 ) )                  k3           =  k3_
      IF ( PRESENT( xk ) )                 xk(1:3,1:num_k_points_) = xk_(:,1:num_k_points_)
      IF ( PRESENT( wk ) )                 wk(1:num_k_points_)     = wk_(1:num_k_points_)
      !
      DEALLOCATE( xk_ )
      DEALLOCATE( wk_ )
      !
    END SUBROUTINE qexpt_read_bz


    !------------------------------------------------------------------------
    SUBROUTINE qexpt_read_bands( nbnd, num_k_points, nspin, ef, nelec, &
                                 xk, wk, eig, eig_s, energy_units, k_units, ierr )
      !------------------------------------------------------------------------
      !
      INTEGER,      OPTIONAL, INTENT(OUT) :: nbnd, num_k_points, nspin
      REAL(dbl),    OPTIONAL, INTENT(OUT) :: ef, nelec, xk(:,:), wk(:)
      REAL(dbl),    OPTIONAL, INTENT(OUT) :: eig(:,:), eig_s(:,:,:)
      CHARACTER(*), OPTIONAL, INTENT(OUT) :: energy_units, k_units
      INTEGER,                INTENT(OUT) :: ierr
      !
      INTEGER        :: nbnd_, num_k_points_, nspin_
      REAL(dbl)      :: ef_
      CHARACTER(256) :: energy_units_
      INTEGER        :: ik, ispin, ik_eff
      REAL(dbl), ALLOCATABLE :: xk_(:,:), wk_(:), eig_s_(:,:,:)
      !

      ierr = 0
      !
      !
      ! first read NELEC from a different main section
      !
      IF ( PRESENT( nelec ) ) THEN
           !
           CALL iotk_scan_begin( iunpun, "Other_parameters", IERR=ierr )
           IF ( ierr/=0 ) RETURN
           !
           CALL iotk_scan_empty( iunpun, "Charge", ATTR=attr, IERR=ierr )
           IF ( ierr/=0 ) RETURN
           CALL iotk_scan_attr( attr, "nelec", nelec, IERR=ierr )
           IF ( ierr/=0 ) RETURN
           !
           CALL iotk_scan_end( iunpun, "Other_parameters", IERR=ierr )
           IF ( ierr/=0 ) RETURN
           !
      ENDIF
      !
      ! then the true main section
      !
      CALL iotk_scan_begin( iunpun, "Eigenvalues", ATTR=attr, IERR=ierr )
      IF ( ierr/=0 ) RETURN
      !
      CALL iotk_scan_attr( attr, "nspin", nspin_, IERR=ierr )
      IF ( ierr/=0 ) RETURN
      CALL iotk_scan_attr( attr, "nk", num_k_points_, IERR=ierr )
      IF ( ierr/=0 ) RETURN
      CALL iotk_scan_attr( attr, "nbnd", nbnd_, IERR=ierr )
      IF ( ierr/=0 ) RETURN
      CALL iotk_scan_attr( attr, "efermi", ef_, IERR=ierr )
      IF ( ierr/=0 ) RETURN
      CALL iotk_scan_attr( attr, "units", energy_units_, IERR=ierr )
      IF ( ierr/=0 ) RETURN
      !
      ! due to the doubling convention in espresso
      !
      num_k_points_ = num_k_points_ / nspin_
      !
      !
      ALLOCATE( xk_(3, num_k_points_ * nspin_ ) )
      ALLOCATE( wk_(   num_k_points_ * nspin_ ) )
      ALLOCATE( eig_s_( nbnd_, num_k_points_, nspin_ ) )
      !
      DO ispin = 1, nspin_
      DO ik    = 1, num_k_points_
           !
           ik_eff = ik + ( ispin-1 ) * num_k_points_
           !
           CALL iotk_scan_dat( iunpun, "e"//TRIM( iotk_index(ik_eff)),  &
                               eig_s_(:, ik, ispin) , IERR=ierr)
           IF ( ierr/=0 ) RETURN
           !
      ENDDO
      ENDDO
      !
      CALL iotk_scan_end( iunpun, "Eigenvalues", IERR=ierr )
      IF ( ierr/=0 ) RETURN
      !
      !      
      IF ( PRESENT( xk ) .OR. PRESENT( wk) ) THEN
          !
          ! read kpt data 
          !
          CALL iotk_scan_begin( iunpun, "Kmesh", IERR=ierr )
          IF ( ierr/=0 ) RETURN
          !
          CALL iotk_scan_dat( iunpun, "weights", wk_, IERR=ierr)
          IF ( ierr/=0 ) RETURN
          !
          CALL iotk_scan_dat( iunpun, "k", xk_, IERR=ierr)
          IF ( ierr/=0 ) RETURN
          !
          CALL iotk_scan_end( iunpun, "Kmesh", IERR=ierr )
          IF ( ierr/=0 ) RETURN
          !
      ENDIF
      !
      !
      IF ( PRESENT( nbnd ) )             nbnd           = nbnd_
      IF ( PRESENT( num_k_points ) )     num_k_points   = num_k_points_
      IF ( PRESENT( nspin ) )            nspin          = nspin_
      IF ( PRESENT( ef ) )               ef             = ef_
      IF ( PRESENT( energy_units ) )     energy_units   = TRIM( energy_units_ )
      IF ( PRESENT( k_units ) )          k_units        = "2 pi / a"
      IF ( PRESENT( xk ) )               xk(1:3, 1:num_k_points_)  = xk_(:,1:num_k_points_)
      IF ( PRESENT( wk ) )               wk(     1:num_k_points_)  = wk_(  1:num_k_points_)
      IF ( PRESENT( eig ) )              eig  (1:nbnd_, 1:num_k_points_ )  = eig_s_(:,:, 1)
      IF ( PRESENT( eig_s ) )            eig_s(1:nbnd_, 1:num_k_points_, 1:nspin_ ) = eig_s_(:,:,:)
      !
      DEALLOCATE( xk_)
      DEALLOCATE( wk_)
      DEALLOCATE( eig_s_ )
      !
    END SUBROUTINE qexpt_read_bands
    !
    !
    !------------------------------------------------------------------------
    SUBROUTINE qexpt_read_wfc( ibnds, ibnde, ik, ispin, igk, ngw, igwx, wf, wf_kindip, ierr )
      !------------------------------------------------------------------------
      !
      ! read wfc from IBNDS to IBNDE, for kpt IK and spin ISPIN
      ! WF is the wfc on its proper k+g grid, while WF_KINDIP is the same wfc
      ! but on a truncated rho grid (k-point indipendent)
      !
      INTEGER,                 INTENT(IN)  :: ibnds, ibnde, ik, ispin
      INTEGER,       OPTIONAL, INTENT(IN)  :: igk(:)
      INTEGER,       OPTIONAL, INTENT(OUT) :: ngw, igwx
      COMPLEX(dbl),  OPTIONAL, INTENT(OUT) :: wf(:,:), wf_kindip(:,:)
      INTEGER,                 INTENT(OUT) :: ierr
      !
      !
      ! data to be saved for performance purposes
      !
      INTEGER,  SAVE :: nspin, nkpts
      LOGICAL,  SAVE :: first_call = .TRUE.
      !
      INTEGER        :: ngw_, igwx_, ig, ib, ik_eff, lindex
      INTEGER        :: iunwfc
      CHARACTER(256) :: filename
      COMPLEX(dbl),  ALLOCATABLE :: wf_(:)

      ierr = 0
      !
      !
      ! at the first call read some aux dimensions
      !
      IF ( first_call ) THEN
           !
           CALL iotk_scan_begin( iunpun, 'Eigenvalues', ATTR=attr, IERR=ierr)
           IF ( ierr /=0 ) RETURN
           !
           CALL iotk_scan_attr( attr, 'nspin', nspin, IERR=ierr)
           IF ( ierr /=0 ) RETURN
           CALL iotk_scan_attr( attr, 'nk',  nkpts, IERR=ierr)
           IF ( ierr /=0 ) RETURN
           !
           CALL iotk_scan_end( iunpun, 'Eigenvalues', IERR=ierr)
           IF ( ierr /=0 ) RETURN
           !
           first_call = .FALSE.
           !
      ENDIF
      !
      !
      ik_eff = ik + ( ispin -1 ) * nkpts / nspin
      !
      ! read the main data
      !
      CALL iotk_free_unit( iunwfc )
      filename = TRIM(datadir) // '/' // 'wfc' //TRIM( iotk_index(ik_eff) )
      !
      CALL iotk_open_read ( iunwfc, FILE = TRIM(filename), IERR=ierr )
      IF (ierr/=0)  RETURN
      !
      !
      CALL iotk_scan_empty( iunwfc, 'Info', ATTR=attr, IERR=ierr)
      IF ( ierr /=0 ) RETURN
      !
      CALL iotk_scan_attr( attr, 'ngw',  ngw_, IERR=ierr)
      IF ( ierr /=0 ) RETURN
      CALL iotk_scan_attr( attr, 'igwx', igwx_, IERR=ierr)
      IF ( ierr /=0 ) RETURN
      !
      !
      IF ( PRESENT( wf ) ) THEN
          !
          lindex = 0
          !
          DO ib = ibnds, ibnde
              !
              lindex = lindex + 1
              !
              CALL iotk_scan_dat( iunwfc,  'Wfc'//TRIM(iotk_index( ib )) , &
                                  wf( 1:igwx_, lindex ), IERR=ierr)
              IF ( ierr /=0 ) RETURN
              !
          ENDDO
          !
      ENDIF
      !
      !
      IF ( PRESENT( wf_kindip ) ) THEN
          !
          ALLOCATE( wf_(igwx_ ), STAT=ierr )
          IF (ierr/=0) RETURN
          !
          IF ( .NOT. PRESENT( igk ) ) THEN
              ierr = 3
              RETURN
          ENDIF
          !
          IF ( MAXVAL( igk( 1: igwx_ ) ) > SIZE( wf_kindip, 1)  ) THEN
              ierr = 4
              RETURN
          ENDIF
          !
          !
          lindex = 0
          !
          DO ib = ibnds, ibnde
              !
              lindex = lindex + 1
              !
              CALL iotk_scan_dat( iunwfc, "Wfc"//TRIM(iotk_index( ib ) ), &
                                           wf_(1:igwx_), IERR=ierr )
              IF (ierr/=0) RETURN
              !
              ! use the igk map to do the transformation
              !
              wf_kindip(:, lindex) = 0.0_dbl
              !
              DO ig = 1, igwx_
                  !
                  wf_kindip( igk( ig ), lindex ) = wf_( ig )
                  !
              ENDDO
              !
          ENDDO
          !
          DEALLOCATE( wf_, STAT=ierr )
          IF (ierr/=0) RETURN
          !
      ENDIF
      !
      !
      CALL iotk_close_read ( iunwfc, IERR=ierr )
      IF (ierr/=0)  RETURN
      !
      !
      IF ( PRESENT( ngw ) )     ngw    = ngw_
      IF ( PRESENT( igwx ) )    igwx   = igwx_
      !
    END SUBROUTINE qexpt_read_wfc
    !
    !
END MODULE qexpt_module

