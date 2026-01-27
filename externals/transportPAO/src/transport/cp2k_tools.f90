!
! Copyright (C) 2012 WanT Group, 2017 ERMES Group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License\'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
! 
! This module has been written together with
! Leopold Talirz and Carlo A. Pignedoli (to whom complaints have to be addressed)
!
!*********************************************
   MODULE cp2k_tools_module
!*********************************************
   !
   USE kinds,              ONLY : dbl
   USE constants,          ONLY : BOHR => bohr_radius_angs, ZERO, ONE, &
                                  CZERO, CONE, TWO, RYD, EPS_m8
   USE parameters,         ONLY : nstrx
   USE timing_module,      ONLY : timing
   USE log_module,         ONLY : log_push, log_pop
   USE parser_module,      ONLY : change_case, matches, field_count
   USE grids_module,       ONLY : grids_get_rgrid
   USE iotk_module
   USE util_module 
   !
   IMPLICIT NONE 
   PRIVATE
   SAVE

   !
   ! global variables of the module
   !

   !
   ! contains:
   ! SUBROUTINE  cp2k_tools_init( prefix_ )
   ! SUBROUTINE  cp2k_tools_get_dims( filename, [nkpts, dimbset] )
   ! SUBROUTINE  cp2k_tools_get_data( filename, dimbset, nkpts, nspin, ... )
   ! SUBROUTINE  cp2k_to_internal( filein, fileout, filetype [, do_orthoovp] )
   ! FUNCTION    file_is_cp2k( filein )
   !
   PUBLIC :: cp2k_tools_get_dims
   PUBLIC :: cp2k_tools_get_data
   PUBLIC :: cp2k_to_internal
   PUBLIC :: file_is_cp2k

CONTAINS


!**********************************************************
   SUBROUTINE cp2k_tools_get_dims( filename, nkpts, dimbset )
   !**********************************************************
   !
   ! get the dimensions of the problem
   ! need to have the module initialized
   !
   IMPLICIT NONE
   ! 
   CHARACTER(*),      INTENT(IN)  :: filename
   INTEGER, OPTIONAL, INTENT(OUT) :: nkpts, dimbset
   !
   CHARACTER(24) :: subname='cp2k_tools_get_dims'
   CHARACTER(256):: line 
   LOGICAL       :: ldone
   INTEGER       :: iunit, ierr
   INTEGER       :: num, ldim, dimbset_
   
   CALL timing( subname, OPR="start" )
   CALL log_push( subname )
   !
   CALL iotk_free_unit( iunit )

   !
   ! check kind compatibility
   !
   IF ( dbl /= KIND(1.0D0)) CALL errore(subname,'internal dbl kind incompatible',10)

   !
   ! get dimensions
   !
   OPEN( iunit, FILE=filename, STATUS='old', IOSTAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'opening '//TRIM(filename),ABS(ierr) )
   !
   ldone = .FALSE.
   ldim  = 0
   dimbset_ = 0
   !
   DO WHILE ( .NOT. ldone )
       !
       READ( iunit, "(A256)", IOSTAT=ierr ) line
       IF ( ierr/=0 ) CALL errore(subname,'reading line', 10)
       !
       CALL field_count( num, line )
       IF ( num < 5) CYCLE
       !
       READ ( line, *, IOSTAT=ierr) ldim
       IF ( ierr/=0 ) CALL errore(subname,'reading dims', 11)
       !
       IF ( ldim > dimbset_ ) THEN
          dimbset_ = ldim
       ELSE
          ldone = .TRUE.
       ENDIF
       !
   ENDDO
   !
   CLOSE ( iunit, IOSTAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'closing '//TRIM(filename),ABS(ierr) )
   !
   IF ( PRESENT( nkpts ) )      nkpts = 1
   IF ( PRESENT( dimbset ) )  dimbset = dimbset_
  
   !
   CALL log_pop( subname )
   CALL timing( subname, OPR="stop" )
   RETURN
   !
END SUBROUTINE cp2k_tools_get_dims


!**********************************************************
   SUBROUTINE cp2k_tools_get_data( filename, dimbset, nkpts, nspin, rham, rovp )
   !**********************************************************
   !
   ! read main data from cp2k files
   !
   IMPLICIT NONE
   ! 
   CHARACTER(*), INTENT(IN)   :: filename
   INTEGER,      INTENT(IN)   :: dimbset, nkpts, nspin
   REAL(dbl),    INTENT(OUT)  :: rham(dimbset,dimbset,nkpts,nspin)
   REAL(dbl),    INTENT(OUT)  :: rovp(dimbset,dimbset,nkpts,nspin)
   !
   CHARACTER(24) :: subname='cp2k_tools_get_data'
   LOGICAL       :: lfound
   INTEGER       :: iunit
   INTEGER       :: ik, isp, ierr
   

   CALL log_push( subname )
   !
   CALL iotk_free_unit( iunit )

   !
   ! check kind compatibility
   !
   IF ( dbl /= KIND(1.0D0))  CALL errore(subname,'internal dbl kind incompatible',10)
   IF ( nkpts /= 1 )         CALL errore(subname,"nkpts/=1: unexpected value",10)

   !
   ! get data from datafile
   !
   OPEN( iunit, FILE=filename, STATUS='old', IOSTAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'opening '//TRIM(filename),ABS(ierr) )

   !
   ! first we read the overlap matrix
   !
   lfound  = .FALSE.
   !
   CALL cp2k_search_tag( iunit, "OVERLAP MATRIX", lfound )
   !
   IF ( .NOT. lfound ) CALL errore(subname,'tag OVERLAP MATRIX not found',10)
       
   !
   ! read overlaps
   !
   CALL cp2k_read_matrix( iunit, dimbset, rovp(:,:,1,1) )
   !
   IF ( nspin == 2 )  rovp(:,:,1,2) = rovp(:,:,1,1)


   !
   ! read hamiltonian
   !
   IF ( nspin == 1 ) THEN
       !
       CALL cp2k_search_tag( iunit, "KOHN-SHAM MATRIX (SCF ENV 1)", lfound )
       !
       IF ( .NOT. lfound ) CALL errore(subname,'tag KOHN-SHAM MATRIX not found',10)
       !
       ! read rham
       !
       CALL cp2k_read_matrix( iunit, dimbset, rham(:,:,1,1) )
       !
   ELSEIF ( nspin == 2 ) THEN
       !
       CALL errore(subname,'nspin=2 not implemented',10)
       !
   ELSE
       CALL errore(subname,'nspin: unexpected value',ABS(nspin)+1)
   ENDIF
   !
   ! convert to eV
   rham = rham * TWO * RYD
   !
   CLOSE( iunit, IOSTAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'closing '//TRIM(filename),ABS(ierr) )

   !
   ! final checks
   !
   DO isp = 1, nspin 
   DO ik  = 1, nkpts
       !
       IF ( .NOT. mat_is_herm( dimbset, rovp(:,:,ik,isp), TOLL=EPS_m8 ) ) &
             CALL errore(subname,"rovp is not hermitean", 10)
       !
       IF ( .NOT. mat_is_herm( dimbset, rham(:,:,ik,isp), TOLL=EPS_m8 ) ) &
             CALL errore(subname,"rham is not hermitean", 10)
       !
   ENDDO
   ENDDO
   !
   !
   CALL log_pop( subname )
   RETURN
   !
END SUBROUTINE cp2k_tools_get_data


!**********************************************************
   SUBROUTINE cp2k_to_internal( filein, fileout, filetype, do_orthoovp )
   !**********************************************************
   !
   ! Convert the datafile written by the cp2k program to
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
   INTEGER                     :: ounit
   LOGICAL                     :: do_orthoovp_
   !
   CHARACTER(21)               :: subname="cp2k_to_internal"
   !
   CHARACTER(nstrx)            :: attr, filetype_
   INTEGER                     :: dimbset, nkpts, nk(3), shift(3), nrtot, nr(3)
   INTEGER                     :: nspin, nbnd
   INTEGER                     :: ierr, i, j, ir, isp
   !
   LOGICAL                     :: write_ham, write_space
   !
   REAL(dbl)                   :: norm
   REAL(dbl),      ALLOCATABLE :: vkpt(:,:), wk(:), wr(:), vr(:,:)
   INTEGER,        ALLOCATABLE :: ivr(:,:)
   REAL(dbl),      ALLOCATABLE :: rham(:,:,:,:)
   REAL(dbl),      ALLOCATABLE :: rovp(:,:,:,:)
   REAL(dbl),      ALLOCATABLE :: rtmp(:,:), raux(:,:)
   REAL(dbl),      ALLOCATABLE :: w(:), rovp_isq(:,:)

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
      !
      write_space = .TRUE.
      CALL errore(subname, 'not yet implemented', 10 )
      !
   CASE DEFAULT
      CALL errore(subname, 'invalid filetype: '//TRIM(filetype_), 71 )
   END SELECT

   !
   ! orthogonalization controlled by input
   !
   do_orthoovp_ = .FALSE.
   IF ( PRESENT( do_orthoovp ) ) do_orthoovp_ = do_orthoovp

!
!---------------------------------
! read from filein (CP2K fmt)
!---------------------------------
!

   CALL cp2k_tools_get_dims( filein, nkpts, dimbset )

   !
   ! assume spin unpolarized calculation
   !
   nbnd  = 0
   nspin = 1
   isp   = 1
   !
   !WRITE(0,*) "Reading dims"
   !WRITE(0,*) "    nkpts : ", nkpts
   !WRITE(0,*) "  dimbset : ", dimbset

   !
   ! allocate main quantities to be readed
   !
   ALLOCATE( vkpt(3, nkpts), STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'allocating vkpt', ABS(ierr) )
   !
   ALLOCATE( wk(nkpts), STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'allocating wk', ABS(ierr) )
  

   !
   ! simple data to be initialized
   !

   vkpt(:,1)  = 0.0_dbl 
   wk(1)      = 1.0_dbl
   nk(1:3)    = 1
   shift(1:3) = 0

   !
   ! check the normalization of the weights
   !
   norm   = SUM( wk )
   wk(:)  = wk(:) / norm


   !
   !
   ! real-space lattice vectors
   !
   nr(1:3) = nk(1:3)
   nrtot   = 1
   !
   ALLOCATE( ivr(3, nrtot), vr(3, nrtot), STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'allocating ivr, vr', ABS(ierr) )
   !
   ALLOCATE( wr(nrtot), STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'allocating wr', ABS(ierr) )
   !
   CALL grids_get_rgrid(nr, WR=wr, IVR=ivr )

   vr(:,:) = REAL( ivr, dbl)
   

   !
   ! define the Hamiltonian on the WF basis
   !
   IF ( write_ham ) THEN
       !
       ALLOCATE( rham(dimbset, dimbset, nrtot, nspin), STAT=ierr )
       IF ( ierr/=0 ) CALL errore(subname, 'allocating rham', ABS(ierr) )
       ALLOCATE( rovp(dimbset, dimbset, nrtot, nspin), STAT=ierr )
       IF ( ierr/=0 ) CALL errore(subname, 'allocating rovp', ABS(ierr) )
   
       !
       ! read and set the main dataset
       !
       CALL cp2k_tools_get_data( filein, dimbset, nkpts, nspin, rham, rovp )
       

       !
       ! if required, we lowding-orthogonalize the basis
       !
       IF ( do_orthoovp_ ) THEN
           !
           ALLOCATE( raux(dimbset,dimbset), rtmp(dimbset,dimbset), STAT=ierr )
           IF ( ierr/=0 ) CALL errore(subname, 'allocating raux-rtmp', ABS(ierr) )
           ALLOCATE( w(dimbset), rovp_isq(dimbset,dimbset), STAT=ierr )
           IF ( ierr/=0 ) CALL errore(subname, 'allocating w, rovp_isq', ABS(ierr) )
           !
           !
           IF ( nrtot /= 1 ) CALL errore(subname,"nrtot: unexpected value",10)
           IF ( nkpts /= 1 ) CALL errore(subname,"nkpts, unexpected value",10)
           !
           DO isp = 1, nspin
               !
               CALL mat_hdiag( raux, w(:), rovp(:,:,1,isp), dimbset)
               !              
               DO i = 1, dimbset
                   !
                   IF ( w(i) <= ZERO ) CALL errore(subname,'unexpected eig < = 0 ',i)
                   w(i) = ONE / SQRT( w(i) )
                   !
               ENDDO
               !
               DO j = 1, dimbset
               DO i = 1, dimbset
                   !
                   rtmp(i,j) = raux(i,j) * w(j)
                   !
               ENDDO
               ENDDO
               !
               CALL mat_mul( rovp_isq, rtmp, 'N', raux, 'T', dimbset, dimbset, dimbset)
               !
               !
               ! apply the basis change to the hamiltonian
               ! multiply rovp_isq (S^-1/2) to the right and the left of rham
               !
               CALL mat_mul( raux, rovp_isq, 'N', rham(:,:,1,isp), 'N', dimbset, dimbset, dimbset)
               CALL mat_mul( rham(:,:,1,isp), raux, 'N', rovp_isq, 'N', dimbset, dimbset, dimbset)
               !
               !
               DEALLOCATE( raux, rtmp, STAT=ierr)
               IF ( ierr/=0 ) CALL errore(subname,'deallocating raux, rtmp',ABS(ierr))
               DEALLOCATE( w, rovp_isq, STAT=ierr)
               IF ( ierr/=0 ) CALL errore(subname,'deallocating w, rovp_isq',ABS(ierr))
               !
           ENDDO
           !
       ENDIF
       !
   ENDIF


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
       CALL iotk_write_attr( attr,"dimwann",dimbset,FIRST=.TRUE.)
       CALL iotk_write_attr( attr,"nkpts",nkpts)
       CALL iotk_write_attr( attr,"nspin",nspin)
       CALL iotk_write_attr( attr,"nk",nk)
       CALL iotk_write_attr( attr,"shift",shift)
       CALL iotk_write_attr( attr,"nrtot",nrtot)
       CALL iotk_write_attr( attr,"nr",nr)
       CALL iotk_write_attr( attr,"have_overlap", .NOT. do_orthoovp_ )
       CALL iotk_write_attr( attr,"fermi_energy", 0.0_dbl )
       CALL iotk_write_empty( ounit,"DATA",ATTR=attr)
       !
       !CALL iotk_write_attr( attr,"units","bohr",FIRST=.TRUE.)
       !CALL iotk_write_dat( ounit,"DIRECT_LATTICE", avec, ATTR=attr, COLUMNS=3)
       !
       !CALL iotk_write_attr( attr,"units","bohr^-1",FIRST=.TRUE.)
       !CALL iotk_write_dat( ounit,"RECIPROCAL_LATTICE", bvec, ATTR=attr, COLUMNS=3)
       !
       CALL iotk_write_attr( attr,"units","crystal",FIRST=.TRUE.)
       CALL iotk_write_dat( ounit,"VKPT", vkpt, ATTR=attr, COLUMNS=3)
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
          CALL iotk_write_begin( ounit,"RHAM")
          !
          DO ir = 1, nrtot
              !
              CALL iotk_write_dat( ounit,"VR"//TRIM(iotk_index(ir)),&
                                   CMPLX( rham( :, :, ir, isp), ZERO, dbl) )
              !
              IF ( .NOT. do_orthoovp_ ) THEN 
                  CALL iotk_write_dat( ounit,"OVERLAP"//TRIM(iotk_index(ir)), &
                                       CMPLX( rovp( :, :, ir, isp), ZERO, dbl) )
              ENDIF
              !
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

!
! This is left for a future implementation
! (in case we really need it)
!
!   IF ( write_space ) THEN
!       !
!       CALL iotk_open_write( ounit, FILE=TRIM(fileout), BINARY=binary )
!       !
!       !
!       CALL iotk_write_begin( ounit, "WINDOWS" )
!       !
!       !
!       CALL iotk_write_attr( attr,"nbnd",nbnd,FIRST=.TRUE.)
!       CALL iotk_write_attr( attr,"nkpts",nkpts)
!       CALL iotk_write_attr( attr,"nspin",nspin)
!       CALL iotk_write_attr( attr,"spin_component","none")
!       CALL iotk_write_attr( attr,"efermi", 0.0_dbl )
!       CALL iotk_write_attr( attr,"dimwinx", dimwinx )
!       CALL iotk_write_empty( ounit,"DATA",ATTR=attr)
!       !
!       CALL iotk_write_dat( ounit, "DIMWIN", ndimwin, COLUMNS=8 )
!       CALL iotk_write_dat( ounit, "IMIN", imin, COLUMNS=8 )
!       CALL iotk_write_dat( ounit, "IMAX", imax, COLUMNS=8 )
!       !
!       DO isp = 1, nspin
!           !
!           IF ( nspin == 2 ) THEN
!               CALL iotk_write_begin( ounit, "SPIN"//TRIM(iotk_index(isp)) )
!           ENDIF
!           !
!           CALL iotk_write_dat( ounit, "EIG", eig(:,:,isp), COLUMNS=4)
!           !
!           IF ( nspin == 2 ) THEN
!               CALL iotk_write_end( ounit, "SPIN"//TRIM(iotk_index(isp)) )
!           ENDIF
!           !
!       ENDDO
!       !
!       CALL iotk_write_end( ounit, "WINDOWS" )
!       !
!       !
!       CALL iotk_write_begin( ounit, "SUBSPACE" )
!       !
!       CALL iotk_write_attr( attr,"dimwinx",dimwinx,FIRST=.TRUE.)
!       CALL iotk_write_attr( attr,"nkpts",nkpts)
!       CALL iotk_write_attr( attr,"dimwann", dimbset)
!       CALL iotk_write_empty( ounit,"DATA",ATTR=attr)
!       !
!       CALL iotk_write_dat( ounit, "DIMWIN", ndimwin, COLUMNS=8 )
!       CALL iotk_write_dat( ounit, "WAN_EIGENVALUES", eig_wan, COLUMNS=8 )
!       !
!       CALL iotk_write_end( ounit, "SUBSPACE" )
!       !
!       !
!       CALL iotk_close_write( ounit )
!       !
!   ENDIF

!
! local cleaning
!

   DEALLOCATE( vkpt, wk, STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'deallocating vkpt, wk', ABS(ierr) )
   !
   DEALLOCATE( ivr, vr, wr, STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'deallocating ivr, wr', ABS(ierr) )
   !
   IF( ALLOCATED( rham ) ) THEN
       DEALLOCATE( rham, STAT=ierr )
       IF ( ierr/=0 ) CALL errore(subname, 'deallocating rham', ABS(ierr) )
   ENDIF
   !
   IF( ALLOCATED( rovp ) ) THEN
       DEALLOCATE( rovp, STAT=ierr )
       IF ( ierr/=0 ) CALL errore(subname, 'deallocating rovp', ABS(ierr) )
   ENDIF
   
   CALL log_pop( subname )
   CALL timing( subname, OPR='stop' )
   !
   RETURN
   !
END SUBROUTINE cp2k_to_internal


!**********************************************************
   LOGICAL FUNCTION file_is_cp2k( filename )
   !**********************************************************
   !
   IMPLICIT NONE
   !
   ! Check for cp2k fmt
   ! To do this, we check that the main datafile
   ! contains as a first (non-empty) line "OVERLAP MATRIX"
   !
   CHARACTER(*), INTENT(IN) :: filename
   !
   CHARACTER(12)    :: subname='file_is_cp2k'
   CHARACTER(256)   :: str
   INTEGER          :: iunit, ierr
   LOGICAL          :: lerror, lexist, lfound
     !
     !
     CALL iotk_free_unit( iunit )
     !
     file_is_cp2k = .FALSE.
     lerror = .FALSE.
     !
     INQUIRE( FILE=filename, EXIST=lexist ) 
     IF ( .NOT. lexist ) lerror = .TRUE.
     ! 
     OPEN ( iunit, FILE=filename, IOSTAT=ierr )
     IF ( ierr/=0 ) CALL errore(subname,'opening '//TRIM(filename), ABS(ierr) )
     !
     lfound = .FALSE.
 
     DO WHILE ( .NOT. lfound )
         !
         READ( iunit, fmt="(A256)", IOSTAT=ierr ) str 
         !
         IF ( ierr/=0 ) THEN
            file_is_cp2k = .FALSE.
            RETURN
            !CALL errore(subname,'reading line', ABS(ierr) )
         ENDIF
         !
         IF ( LEN_TRIM( str ) /= 0 ) lfound = .TRUE.
         !           
     ENDDO
     !
     CLOSE ( iunit, IOSTAT=ierr )
     IF ( ierr/=0 ) CALL errore(subname,'closing '//TRIM(filename), ABS(ierr) )
     !
     IF ( .NOT. lfound .OR. lerror ) THEN
         file_is_cp2k = .FALSE.
         RETURN
     ENDIF
     !
     IF ( matches ("OVERLAP MATRIX", str) ) THEN
         file_is_cp2k = .TRUE.
         RETURN
     ENDIF
     !
  END FUNCTION file_is_cp2k


!**********************************************************
   SUBROUTINE cp2k_search_tag( iunit, tag, lfound )
   !**********************************************************
   !
   IMPLICIT NONE
     !
     INTEGER,      INTENT(IN)  :: iunit
     CHARACTER(*), INTENT(IN)  :: tag
     LOGICAL,      INTENT(OUT) :: lfound
     !
     INTEGER        :: ierr
     CHARACTER(256) :: line
     CHARACTER(15)  :: subname="cp2k_search_tag"
     !
     !
     lfound  = .FALSE.
     DO WHILE ( .NOT. lfound )
         !
         READ( iunit, "(A256)", IOSTAT=ierr ) line
         IF ( ierr/=0 ) CALL errore(subname,'reading line', ABS(ierr))
         !
         IF ( matches(TRIM(tag), line) ) THEN
             lfound = .TRUE.
             EXIT
         ENDIF
         !
     ENDDO
     !
     RETURN
     !
  END SUBROUTINE cp2k_search_tag


!**********************************************************
   SUBROUTINE cp2k_read_matrix( iunit, dimbset, rmat )
   !**********************************************************
   !
   IMPLICIT NONE
     !
     INTEGER,      INTENT(IN)  :: iunit
     INTEGER,      INTENT(IN)  :: dimbset
     REAL(dbl),    INTENT(OUT) :: rmat(dimbset, dimbset)
     !
     CHARACTER(16)  :: subname='cp2k_read_matrix'
     CHARACTER(256) :: line
     CHARACTER(20)  :: adum
     !
     LOGICAL    :: ldone
     INTEGER    :: icols(4), num
     INTEGER    :: icol_s, icol_e, irow
     INTEGER    :: ierr, idum
     REAL(dbl)  :: rtmp(4)

     !
     ldone  = .FALSE.
     icols  = 0
     icol_s = 0
     icol_e = 0
     irow   = 0
     !
     matrix_loop:& 
     DO WHILE ( .NOT. ldone )
         !
         READ( iunit, "(A256)", IOSTAT=ierr ) line
         IF ( ierr/=0 ) CALL errore(subname,'reading line', 10)
         !
         CALL field_count( num, line )
         !
         IF ( num >= 1 .AND. num <= 4 ) THEN
            !
            icols = 0
            !
            READ( line, *, IOSTAT=ierr ) icols(1:num)
            IF ( ierr/=0 ) CALL errore(subname,"reading icols", ABS(ierr) )
            !
            irow   = 0
            icol_s = MINVAL( icols(1:num) )
            icol_e = MAXVAL( icols(1:num) )
            !
         ELSEIF ( num > 4 .AND. num <= 8 ) THEN
            !
            READ( line, *, IOSTAT=ierr ) irow, idum, adum, adum, rtmp(1:num-4)
            IF ( ierr/=0 ) CALL errore(subname,"reading row-line",ABS(ierr))
            !
            rmat( icol_s:icol_e , irow ) = rtmp( 1: icol_e-icol_s+1)
            !
         ELSE
            !
            CYCLE
            !
         ENDIF
         !
         IF ( irow == dimbset .AND. icol_e == dimbset ) THEN
             !
             ldone = .TRUE.
             EXIT matrix_loop
             !
         ENDIF
         !
     ENDDO matrix_loop
     !
     RETURN
     !
  END SUBROUTINE cp2k_read_matrix

END MODULE cp2k_tools_module

