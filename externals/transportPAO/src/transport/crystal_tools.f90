!
! Copyright (C) 2008 WanT Group, 2017 ERMES Group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License\'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!*********************************************
   MODULE crystal_tools_module
!*********************************************
   !
   USE kinds,              ONLY : dbl
   USE constants,          ONLY : BOHR => bohr_radius_angs, ZERO, ONE, TWO, RYD
   USE parameters,         ONLY : nstrx
   USE timing_module,      ONLY : timing
   USE log_module,         ONLY : log_push, log_pop
   USE converters_module,  ONLY : cart2cry, cry2cart
   USE parser_module,      ONLY : change_case
   USE crystal_io_module
   USE iotk_module
   !
   IMPLICIT NONE 
   PRIVATE
   SAVE
   !
   ! contains:
   ! SUBROUTINE  crystal_to_internal( filein, fileout, filetype, do_orthoovp )
   ! FUNCTION    file_is_crystal( filein )
   !
   PUBLIC :: crystal_to_internal
   PUBLIC :: file_is_crystal

CONTAINS

!**********************************************************
   SUBROUTINE crystal_to_internal( filein, fileout, filetype, do_orthoovp )
   !**********************************************************
   !
   ! Convert the datafile written by the CRYSTAL06/09 program to
   ! the internal representation.
   !
   ! FILETYPE values are:
   !  - ham, hamiltonian
   !  - space, subspace
   !
   IMPLICIT NONE

   LOGICAL, PARAMETER :: binary = .TRUE.
   
   !
   ! input variables
   !
   CHARACTER(*), INTENT(IN) :: filein
   CHARACTER(*), INTENT(IN) :: fileout
   CHARACTER(*), INTENT(IN) :: filetype
   LOGICAL, OPTIONAL, INTENT(IN) :: do_orthoovp
   
   !
   ! local variables
   !
   INTEGER                     :: iunit, ounit
   LOGICAL                     :: do_orthoovp_
   !
   CHARACTER(19)               :: subname="crystal_to_internal"
   !
   CHARACTER(nstrx)            :: attr, r_units, a_units, b_units, k_units, h_units, e_units
   CHARACTER(nstrx)            :: filetype_
   INTEGER                     :: dimwann, nkpts, nbnd, nk(3), shift(3), nrtot, nr(3), nspin 
   INTEGER                     :: auxdim1, auxdim2, auxdim3
   INTEGER                     :: ierr, ir, isp
   !
   LOGICAL                     :: write_ham, write_space
   !
   REAL(dbl)                   :: dlatt(3,3), rlatt(3,3), norm, efermi
   INTEGER,        ALLOCATABLE :: ivr(:,:)
   INTEGER,        ALLOCATABLE :: itmp(:)
   REAL(dbl),      ALLOCATABLE :: vkpt_cry(:,:), vkpt(:,:), wk(:), wr(:), vr(:,:)
   REAL(dbl),      ALLOCATABLE :: eig(:,:,:)
   REAL(dbl),      ALLOCATABLE :: rham(:,:,:,:), rovp(:,:,:)

!
!------------------------------
! main body
!------------------------------
!
   CALL timing( subname, OPR='start' )
   CALL log_push( subname )

   !
   ! search for units indipendently of io_module
   !
   CALL iotk_free_unit( iunit )
   CALL iotk_free_unit( ounit )

   !
   ! select the operation to do
   !
   write_ham     = .FALSE.
   write_space   = .FALSE.
   !
   filetype_ = TRIM( filetype )
   CALL change_case( filetype_, 'lower' )
   !
   SELECT CASE( TRIM( filetype_ ) )
   !
   CASE( 'ham', 'hamiltonian' )
      write_ham   = .TRUE.
   CASE( 'space', 'subspace' )
      write_space = .TRUE.
   CASE DEFAULT
      CALL errore(subname, 'invalid filetype: '//TRIM(filetype_), 71 )
   END SELECT

   do_orthoovp_ = .FALSE.
   IF ( PRESENT( do_orthoovp ) ) do_orthoovp_ = do_orthoovp
   !
   IF ( do_orthoovp_ ) CALL errore(subname,"orthogonalization not yet implemented",10)


!
!---------------------------------
! read from filein (CRYSTAL06/9 fmt)
!---------------------------------
!
   CALL crio_open_file( UNIT=iunit, FILENAME=filein, ACTION='read', IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'opening'//TRIM(filein), ABS(ierr) )
   !
   !
   CALL crio_open_section( "GEOMETRY", ACTION='read', IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'opening sec. GEOMETRY', ABS(ierr) )
   !
   CALL crio_read_periodicity( AVEC=dlatt, A_UNITS=a_units, BVEC=rlatt, B_UNITS=b_units, IERR=ierr)
   IF ( ierr/=0 ) CALL errore(subname, 'reading lattices', ABS(ierr) )
   !
   CALL crio_close_section( "GEOMETRY", ACTION='read', IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'closing sec. GEOMETRY', ABS(ierr) )
   !
   !
   ! convert units if the case
   !
   CALL change_case( a_units, 'lower' )
   CALL change_case( b_units, 'lower' )
   !
   !
   SELECT CASE( ADJUSTL(TRIM(a_units)) )
   CASE ( "b", "bohr", "au" )
      !
      ! do nothing
   CASE ( "ang", "angstrom" )
      !
      dlatt = dlatt / BOHR
      !
   CASE DEFAULT
      CALL errore( subname, 'unknown units for A: '//TRIM(a_units), 71)
   END SELECT
   !
   !
   SELECT CASE( ADJUSTL(TRIM(b_units)) )
   CASE ( "bohr^-1", "bohr-1", "au" )
      !
      ! do nothing
   CASE ( "ang-1", "angstrom-1", "ang^-1", "angstrom^-1" )
      !
      rlatt = rlatt * BOHR
      !
   CASE DEFAULT
      CALL errore( subname, 'unknown units for B: '//TRIM(b_units), 71)
   END SELECT

   !
   ! enter section METHOD
   !
   CALL crio_open_section( "METHOD", ACTION='read', IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'opening sec. METHOD', ABS(ierr) )

   !
   ! real-space lattice vectors
   !
   !
   CALL crio_read_direct_lattice( NRTOT=nrtot, IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'reading direct lattice dims', ABS(ierr) )
   !
   ALLOCATE( ivr(3, nrtot), vr(3, nrtot), STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'allocating ivr, vr', ABS(ierr) )
   !
   ALLOCATE( wr(nrtot), STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'allocating wr', ABS(ierr) )
   !
   CALL crio_read_direct_lattice( RVEC=vr, R_UNITS=r_units, IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'reading RVEC', ABS(ierr) )

   !
   ! units
   !
   CALL change_case( r_units, 'lower' )
   !
   SELECT CASE( ADJUSTL(TRIM(r_units)) )
   CASE ( "b", "bohr", "au" )
      !
      CALL cart2cry( vr, dlatt )
      !
   CASE ( "ang", "angstrom" )
      !
      vr = vr / BOHR
      CALL cart2cry( vr, dlatt )
      !
   CASE ( "cry", "crystal", "relative" )
      !
      ! do nothing
   CASE DEFAULT
      CALL errore( subname, 'unknown units for R: '//TRIM(r_units), 71)
   END SELECT
 
   !
   ! define IVR
   !
   DO ir = 1, nrtot
      !
      ivr(:, ir ) = NINT( vr(:, ir) )
      !
   ENDDO

   !
   ! nr is taken from ivr
   !
   nr ( 1 ) = MAXVAL( ivr(1,:) ) - MINVAL( ivr(1,:) ) +1
   nr ( 2 ) = MAXVAL( ivr(2,:) ) - MINVAL( ivr(2,:) ) +1
   nr ( 3 ) = MAXVAL( ivr(3,:) ) - MINVAL( ivr(3,:) ) +1

   !
   ! as a default, set wr to 1.0; to be checked
   !
   wr (1:nrtot) = ONE


   !
   ! read bz information
   !
   CALL crio_read_bz( NUM_K_POINTS=nkpts, NK=nk, K_UNITS=k_units, IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'reading bz dimensions', ABS(ierr) )
   !
   ! crystal does not allow for shifted k-meshes
   !
   shift(1:3) = 0

   !
   ! maybe we need some weird initialization
   ! about nkpts
   !
   ALLOCATE( vkpt_cry(3, nkpts), STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'allocating vkpt_cry', ABS(ierr) )
   ALLOCATE( vkpt(3, nkpts), STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'allocating vkpt', ABS(ierr) )
   !
   ALLOCATE( wk(nkpts), STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'allocating wk', ABS(ierr) )

   CALL crio_read_bz( XK=vkpt, WK=wk, IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'reading bz dimensions', ABS(ierr) )

   !
   ! units (we want kpt to be re-written in crystal units)
   !
   !
   CALL change_case( k_units, 'lower' )
   !
   SELECT CASE( ADJUSTL(TRIM(k_units)) )
   CASE ( "bohr^-1", "bohr-1", "au" )
      !
      vkpt_cry = vkpt
      CALL cart2cry( vkpt_cry, rlatt ) 
      !
   CASE ( "ang-1", "angstrom-1", "ang^-1", "angstrom^-1" )
      !
      vkpt     = vkpt * BOHR
      vkpt_cry = vkpt
      CALL cart2cry( vkpt_cry, rlatt ) 
      !
   CASE ( "cry", "crystal", "relative", "lattice" )
      !
      vkpt_cry = vkpt
      CALL cry2cart( vkpt, rlatt ) 
      !
   CASE DEFAULT
      CALL errore( subname, 'unknown units for kpts: '//TRIM(k_units), 71)
   END SELECT

   !
   ! check the normalization of the weights
   !
   norm   = SUM( wk )
   wk(:)  = wk(:) / norm

   !
   ! exit section METHOD, enter section OUTPUT_DATA
   !
   CALL crio_close_section( "METHOD", ACTION='read', IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'closing sec. METHOD', ABS(ierr) )
   !
   CALL crio_open_section( "OUTPUT_DATA", ACTION='read', IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'opening sec. OUTPUT_DATA', ABS(ierr) )
   
   !
   ! read electronic structure info
   !
   CALL crio_read_elec_structure( NUM_OF_ATOMIC_ORBITALS=dimwann, &
                                  NBND=nbnd, NSPIN=nspin, ENERGY_REF=efermi, &
                                  E_UNITS=e_units, IERR=ierr)
   IF ( ierr/=0 ) CALL errore(subname, 'reading electronic structure (I)', ABS(ierr) )

   ALLOCATE( eig(nbnd, nkpts, nspin), STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'allocating eig', ABS(ierr))
   !
   CALL crio_read_elec_structure( EIG=eig, IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'reading electronic structure (II)', ABS(ierr) )


   !
   ! efermi is converted to eV's
   !
   CALL change_case( e_units, 'lower' )
   !
   SELECT CASE( ADJUSTL(TRIM(e_units)) )
   CASE ( "ha", "hartree", "au" )
      !
      efermi = efermi * TWO * RYD 
      eig    = eig    * TWO * RYD
      !
   CASE ( "ry", "ryd", "rydberg" )
      !
      efermi = efermi * RYD 
      eig    = eig    * RYD
      !
   CASE ( "ev", "electronvolt" )
      !
      ! do nothing
   CASE DEFAULT
      CALL errore( subname, 'unknown units for efermi: '//TRIM(e_units), 72)
   END SELECT



   IF ( write_ham ) THEN
       !
       ! read Overlaps
       !
       CALL crio_read_matrix( "overlaps", DIM_BASIS=dimwann, NRTOT=auxdim2, IERR=ierr)
       IF ( ierr/=0 ) CALL errore(subname, 'reading ovp dimensions', ABS(ierr) )
       !
       IF ( auxdim2 /= nrtot ) CALL errore(subname, 'inconsistent dimensions', 72)
       !
       ALLOCATE( rovp(dimwann, dimwann, nrtot), STAT=ierr )
       IF ( ierr/=0 ) CALL errore(subname, 'allocating rovp', ABS(ierr) )
       ! 
       CALL crio_read_matrix( "overlaps", MATRIX=rovp, IERR=ierr )
       IF ( ierr/=0 ) CALL errore(subname, 'reading ovp matrix', ABS(ierr) )
    
    
       !
       ! read Hamiltonian 
       !
       CALL crio_read_matrix( "hamiltonian", UNITS=h_units, NSPIN=auxdim3, DIM_BASIS=auxdim1, &
                                             NRTOT=auxdim2, IERR=ierr)
       IF ( ierr/=0 ) CALL errore(subname, 'reading ham dimensions', ABS(ierr) )
       !
       IF ( auxdim1 /= dimwann ) CALL errore(subname, 'inconsistent dimwann in ham', 73)
       IF ( auxdim2 /= nrtot )   CALL errore(subname, 'inconsistent nrtot in ham', 74)
       IF ( auxdim3 /= nspin )   CALL errore(subname, 'inconsistent nspin in ham', 75)
       !
       !
       ALLOCATE( rham(dimwann, dimwann, nrtot, nspin), STAT=ierr )
       IF ( ierr/=0 ) CALL errore(subname, 'allocating rham', ABS(ierr) )
       !
       SELECT CASE ( nspin ) 
       CASE ( 1 )
          !
          CALL crio_read_matrix( "hamiltonian", MATRIX=rham(:,:,:,1), IERR=ierr )
          IF ( ierr/=0 ) CALL errore(subname, 'reading ham matrix', ABS(ierr) )
          !
       CASE ( 2 )
          !
          CALL crio_read_matrix( "hamiltonian", MATRIX_S=rham, IERR=ierr )
          IF ( ierr/=0 ) CALL errore(subname, 'reading ham matrix spin', ABS(ierr) )
          !
       CASE DEFAULT
          !
          CALL errore(subname, 'invalid spin value', 76 )
          !
       END SELECT
    
       !
       ! units are converted to eV's
       !
       CALL change_case( h_units, 'lower' )
       !
       SELECT CASE( ADJUSTL(TRIM(h_units)) )
       CASE ( "ha", "hartree", "au" )
          !
          rham = rham * TWO * RYD 
          !
       CASE ( "ry", "ryd", "rydberg" )
          !
          rham = rham * RYD 
          !
       CASE ( "ev", "electronvolt" )
          !
          ! do nothing
       CASE DEFAULT
          CALL errore( subname, 'unknown units for ham: '//TRIM(h_units), 71)
       END SELECT
    
       !
       ! take fermi energy into account
       !
       DO isp = 1, nspin
       DO ir  = 1, nrtot
           !
           rham(:,:,ir,isp) = rham(:,:,ir,isp) -efermi * rovp(:,:,ir)
           !
       ENDDO
       ENDDO
       !
   ENDIF


   !
   ! exit sec. OUTPUT_DATA and close the file
   !
   CALL crio_close_section( "OUTPUT_DATA", ACTION='read', IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'closing sec. OUTPUT_DATA', ABS(ierr) )
   !
   CALL crio_close_file( ACTION='read', IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'closing'//TRIM(filein), ABS(ierr))

!
!---------------------------------
! write to fileout (internal fmt)
!---------------------------------
!
   IF ( write_ham ) THEN
       !
       CALL iotk_open_write( ounit, FILE=TRIM(fileout), BINARY=binary )
       CALL iotk_write_begin( ounit, "HAMILTONIAN" )
       !
       !
       CALL iotk_write_attr( attr,"dimwann",dimwann,FIRST=.TRUE.)
       CALL iotk_write_attr( attr,"nkpts",nkpts)
       CALL iotk_write_attr( attr,"nspin",nspin)
       CALL iotk_write_attr( attr,"nk",nk)
       CALL iotk_write_attr( attr,"shift",shift)
       CALL iotk_write_attr( attr,"nrtot",nrtot)
       CALL iotk_write_attr( attr,"nr",nr)
       CALL iotk_write_attr( attr,"have_overlap", .TRUE. )
       CALL iotk_write_attr( attr,"fermi_energy", 0.0 )
       CALL iotk_write_empty( ounit,"DATA",ATTR=attr)
       !
       CALL iotk_write_attr( attr,"units","bohr",FIRST=.TRUE.)
       CALL iotk_write_dat( ounit,"DIRECT_LATTICE", dlatt, ATTR=attr, COLUMNS=3)
       !
       CALL iotk_write_attr( attr,"units","bohr^-1",FIRST=.TRUE.)
       CALL iotk_write_dat( ounit,"RECIPROCAL_LATTICE", rlatt, ATTR=attr, COLUMNS=3)
       !
       CALL iotk_write_attr( attr,"units","crystal",FIRST=.TRUE.)
       CALL iotk_write_dat( ounit,"VKPT", vkpt_cry, ATTR=attr, COLUMNS=3)
       CALL iotk_write_dat( ounit,"WK", wk)
       !
       CALL iotk_write_dat( ounit,"IVR", ivr, ATTR=attr, COLUMNS=3)
       CALL iotk_write_dat( ounit,"WR", wr)
       !
       !
       spin_loop: & 
       DO isp = 1, nspin
          !
          IF ( nspin == 2 ) THEN
              !
              CALL iotk_write_begin( ounit, "SPIN"//TRIM(iotk_index(isp)) )
              !
          ENDIF
          !
          !
          CALL iotk_write_begin( ounit,"RHAM")
          !
          DO ir = 1, nrtot
              !
              CALL iotk_write_dat( ounit,"VR"//TRIM(iotk_index(ir)), &
                                   CMPLX(rham( :, :, ir, isp), ZERO, dbl) )
              CALL iotk_write_dat( ounit,"OVERLAP"//TRIM(iotk_index(ir)), &
                                   CMPLX(rovp( :, :, ir), ZERO, dbl) )
              !
          ENDDO
          !
          CALL iotk_write_end( ounit,"RHAM")
          !
          IF ( nspin == 2 ) THEN
              !
              CALL iotk_write_end( ounit, "SPIN"//TRIM(iotk_index(isp)) )
              !
          ENDIF
          !
       ENDDO spin_loop
       !
       CALL iotk_write_end( ounit, "HAMILTONIAN" )
       CALL iotk_close_write( ounit )
       !
   ENDIF

   IF ( write_space ) THEN
       !
       CALL iotk_open_write( ounit, FILE=TRIM(fileout), BINARY=binary )
       !
       !
       CALL iotk_write_begin( ounit, "WINDOWS" )
       !
       !
       CALL iotk_write_attr( attr,"nbnd",nbnd,FIRST=.TRUE.)
       CALL iotk_write_attr( attr,"nkpts",nkpts)
       CALL iotk_write_attr( attr,"nspin",nspin)
       CALL iotk_write_attr( attr,"spin_component","none")
       CALL iotk_write_attr( attr,"efermi", 0.0 )
       CALL iotk_write_attr( attr,"dimwinx", dimwann )
       CALL iotk_write_empty( ounit,"DATA",ATTR=attr)
       !
       ALLOCATE( itmp(nkpts), STAT=ierr )
       IF ( ierr/=0 ) CALL errore(subname, 'allocating itmp', ABS(ierr))
       !
       itmp(:) = dimwann 
       CALL iotk_write_dat( ounit, "DIMWIN", itmp, COLUMNS=8 )
       itmp(:) = 1
       CALL iotk_write_dat( ounit, "IMIN", itmp, COLUMNS=8 )
       itmp(:) = dimwann
       CALL iotk_write_dat( ounit, "IMAX", itmp, COLUMNS=8 )
       !
       DO isp = 1, nspin
           !
           IF ( nspin == 2 ) THEN
               CALL iotk_write_begin( ounit, "SPIN"//TRIM(iotk_index(isp)) )
           ENDIF
           !
           CALL iotk_write_dat( ounit, "EIG", eig(:,:,isp), COLUMNS=4)
           !
           IF ( nspin == 2 ) THEN
               CALL iotk_write_end( ounit, "SPIN"//TRIM(iotk_index(isp)) )
           ENDIF
           !
       ENDDO
       !
       CALL iotk_write_end( ounit, "WINDOWS" )
       !
       !
       CALL iotk_write_begin( ounit, "SUBSPACE" )
       !
       CALL iotk_write_attr( attr,"dimwinx",dimwann,FIRST=.TRUE.)
       CALL iotk_write_attr( attr,"nkpts",nkpts)
       CALL iotk_write_attr( attr,"dimwann", dimwann)
       CALL iotk_write_empty( ounit,"DATA",ATTR=attr)
       !
       itmp(:) = dimwann 
       CALL iotk_write_dat( ounit, "DIMWIN", itmp, COLUMNS=8 )
       !
       CALL iotk_write_end( ounit, "SUBSPACE" )
       !
       !
       CALL iotk_close_write( ounit )
       !
       DEALLOCATE( itmp, STAT=ierr )
       IF ( ierr/=0 ) CALL errore(subname, 'deallocating itmp', ABS(ierr))
       !
   ENDIF

!
! local cleaning
!

   DEALLOCATE( vkpt_cry, wk, STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'deallocating vkpt_cry, wk', ABS(ierr) )
   DEALLOCATE( vkpt, STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'deallocating vkpt', ABS(ierr) )
   !
   DEALLOCATE( ivr, wr, STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'deallocating ivr, wr', ABS(ierr) )
   !
   DEALLOCATE( eig, STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'deallocating eig', ABS(ierr) )
   !
   IF( ALLOCATED( rham ) ) THEN
       DEALLOCATE( rham, STAT=ierr )
       IF ( ierr/=0 ) CALL errore(subname, 'deallocating rham', ABS(ierr) )
   ENDIF
   IF( ALLOCATED( rovp ) ) THEN
       DEALLOCATE( rovp, STAT=ierr )
       IF ( ierr/=0 ) CALL errore(subname, 'deallocating rovp', ABS(ierr) )
   ENDIF
   

   CALL log_pop( subname )
   CALL timing( subname, OPR='stop' )
   !
   RETURN
   !
END SUBROUTINE crystal_to_internal

!**********************************************************
   LOGICAL FUNCTION file_is_crystal( filename )
   !**********************************************************
   !
   IMPLICIT NONE
   !
   ! check for crystal fmt
   !
   CHARACTER(*)     :: filename
   !
   INTEGER          :: iunit
   INTEGER          :: ierr
   LOGICAL          :: lerror, lopnd
   CHARACTER(nstrx) :: prog
     !
     !
     CALL iotk_free_unit( iunit )
     !
     file_is_crystal = .FALSE.
     lerror = .FALSE.
     !
     CALL crio_open_file( iunit, FILENAME=TRIM(filename), ACTION='read', IERR=ierr )
     IF ( ierr/= 0 ) lerror = .TRUE.
     !
     CALL crio_read_header( CREATOR_NAME=prog, IERR=ierr)
     IF ( ierr/= 0 ) lerror = .TRUE.
     !
     IF ( TRIM(prog) /= "CRYSTAL06" ) lerror = .TRUE.
     !
     CALL crio_close_file( ACTION='read', IERR=ierr )
     IF ( ierr/= 0 ) lerror = .TRUE.

     IF ( lerror ) THEN
         !
         INQUIRE( iunit, OPENED=lopnd )
         IF( lopnd ) CLOSE( iunit )
         !
         RETURN
         !
     ENDIF
     !
     file_is_crystal = .TRUE.
     !
  END FUNCTION file_is_crystal

END MODULE crystal_tools_module
