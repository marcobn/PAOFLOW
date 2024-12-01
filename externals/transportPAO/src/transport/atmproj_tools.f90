!
! Copyright (C) 2012 WanT Group, 2017 ERMES Group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License\'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!*********************************************
   MODULE atmproj_tools_module
!*********************************************
   !
   USE kinds,               ONLY : dbl
   USE constants,           ONLY : BOHR => bohr_radius_angs, ZERO, ONE, TWO, &
                                   RYD, EPS_m8, TPI, CZERO !Luis 3
   USE parameters,          ONLY : nstrx
   USE timing_module,       ONLY : timing
   USE log_module,          ONLY : log_push, log_pop
   USE io_global_module,    ONLY : stdout
   USE converters_module,   ONLY : cart2cry, cry2cart
   USE parser_module,       ONLY : change_case
   USE util_module,         ONLY : mat_is_herm, mat_mul
   USE grids_module,        ONLY : grids_get_rgrid
   USE files_module,        ONLY : file_exist
   USE pseudo_types_module, ONLY : pseudo_upf 
   USE iotk_module
   USE qexml_module
   !
   IMPLICIT NONE 
   PRIVATE
   SAVE
   
   ! global variables of the module
   !
   CHARACTER(nstrx)   :: savedir
   CHARACTER(nstrx)   :: file_proj
   CHARACTER(nstrx)   :: file_data
   !
   LOGICAL            :: init = .FALSE.
   !
   ! parameters for the reconstruction 
   ! of the Hamiltonian
   !
   REAL(dbl)          :: atmproj_sh = 10.0d0
   REAL(dbl)          :: atmproj_thr = 0.0d0    ! 0.9d0
   INTEGER            :: atmproj_nbnd = 0
   CHARACTER(256)     :: spin_component = "all"
   LOGICAL            :: atmproj_do_norm = .FALSE. !Luis 6 

   ! contains:
   ! SUBROUTINE  atmproj_to_internal( filein, fileham, filespace, filewan, do_orthoovp )
   ! FUNCTION    file_is_atmproj( filein )
   ! SUBROUTINE  atmproj_get_natomwfc( nsp, psfile, natomwfc )
   ! FUNCTION    atmproj_get_index( i, ia, natomwfc(:) )
   !
   PUBLIC :: atmproj_to_internal
   PUBLIC :: file_is_atmproj
   PUBLIC :: atmproj_get_index
   PUBLIC :: atmproj_get_natomwfc
   !
   PUBLIC :: atmproj_sh
   PUBLIC :: atmproj_thr
   PUBLIC :: atmproj_nbnd
   PUBLIC :: atmproj_do_norm !Luis 6
   PUBLIC :: spin_component

CONTAINS


!**********************************************************
   SUBROUTINE atmproj_tools_init( file_proj_, ierr )
   !**********************************************************
   !
   ! define module global variables
   !
   IMPLICIT NONE
   CHARACTER(*),   INTENT(IN)  :: file_proj_
   INTEGER,        INTENT(OUT) :: ierr
   !
   CHARACTER(18)  :: subname="atmproj_tools_init"
   INTEGER        :: ilen
   !
   file_proj = TRIM( file_proj_ )
   !
   IF ( .NOT. file_exist( file_proj ) ) &
        CALL errore(subname,"file proj not found: "//TRIM(file_proj), 10 )

   ierr = 1
    
   !
   ! define save_dir
   !
   savedir  = ' '
   !
   ilen = LEN_TRIM( file_proj_ )
   IF ( ilen <= 14 ) RETURN
   
   !
   IF ( file_proj_(ilen-14:ilen) == "atomic_proj.xml" .OR. &
        file_proj_(ilen-14:ilen) == "atomic_proj.dat" ) THEN
       !
       savedir = file_proj_(1:ilen-15)
       !
   ENDIF
   !
   IF ( LEN_TRIM(savedir) == 0 ) RETURN
   !
   file_data = TRIM(savedir) // "/data-file.xml"
   !
   IF ( .NOT. file_exist( file_data ) ) RETURN
   !

   !WRITE(0,*) "file_proj: ", TRIM(file_proj)
   !WRITE(0,*) "file_data: ", TRIM(file_data)
   !WRITE(0,*) "savedir:   ", TRIM(savedir)

   ierr     =  0
   init     = .TRUE.
   !
END SUBROUTINE atmproj_tools_init


!**********************************************************
   SUBROUTINE atmproj_to_internal( filein, fileham, filespace, filewan, do_orthoovp )
   !**********************************************************
   !
   ! Convert the datafile written by the projwfc program (QE suite) to
   ! the internal representation.
   !
   ! 3 files are creates: fileham, filespace, filewan
   !
   USE util_module
   USE constants, ONLY : rytoev
   USE T_input_parameters_module,ONLY : surface,efermi_bulk 
!   USE ions_module, ONLY : nat, atm_symb, tau, ityp, nsp
   !
   IMPLICIT NONE

   LOGICAL, PARAMETER :: binary = .FALSE.
   
   !
   ! input variables
   !
   CHARACTER(*), INTENT(IN) :: filein
   CHARACTER(*), OPTIONAL, INTENT(IN) :: fileham, filespace, filewan
   LOGICAL,      OPTIONAL, INTENT(IN) :: do_orthoovp
   
   !
   ! local variables
   !
   CHARACTER(19)     :: subname="atmproj_to_internal"
   INTEGER           :: iunit, ounit
   LOGICAL           :: do_orthoovp_
   INTEGER           :: atmproj_nbnd_
   LOGICAL           :: atmproj_do_norm_                               !Luis 6
   !
   CHARACTER(nstrx)  :: attr, energy_units
   CHARACTER(nstrx)  :: filetype_
   LOGICAL           :: write_ham, write_space, write_loc
   LOGICAL           :: spin_noncollinear                              !Luis 2 
   REAL(dbl)         :: avec(3,3), bvec(3,3), norm, alat, efermi, nelec
   REAL(dbl)         :: proj_wgt, hr, hi
   INTEGER           :: dimwann, natomwfc, nkpts, nspin, nbnd
   INTEGER           :: nspin_                                         !Luis 2
   INTEGER           :: nk(3), shift(3), nrtot, nr(3)
   INTEGER           :: i, j, ir, ik, ib, isp
   INTEGER           :: ierr
   !
   INTEGER,        ALLOCATABLE :: ivr(:,:), itmp(:)
   REAL(dbl),      ALLOCATABLE :: vkpt_cry(:,:), vkpt(:,:), wk(:), wr(:), vr(:,:)
   REAL(dbl),      ALLOCATABLE :: eig(:,:,:)
   REAL(dbl),      ALLOCATABLE :: rtmp(:,:)
   COMPLEX(dbl),   ALLOCATABLE :: rham(:,:,:,:), kham(:,:,:)
   COMPLEX(dbl),   ALLOCATABLE :: hamk2(:,:)
   COMPLEX(dbl),   ALLOCATABLE :: proj(:,:,:,:)
   COMPLEX(dbl),   ALLOCATABLE :: kovp(:,:,:,:), rovp(:,:,:,:)
   COMPLEX(dbl),   ALLOCATABLE :: cu_tmp(:,:,:), eamp_tmp(:,:)
   !
   COMPLEX(dbl),   ALLOCATABLE :: zaux(:,:), ztmp(:,:)
   COMPLEX(dbl),   ALLOCATABLE :: kovp_sq(:,:)
   REAL(dbl),      ALLOCATABLE :: w(:)


   !Luis 3: Changes related to Sohrab's shifting scheme 
   !PA   = A^dagger * A 
   !I_PA = inv(PA)
   COMPLEX(dbl),   ALLOCATABLE :: A(:,:),PA(:,:), IPA(:,:), kham_aux(:,:), E(:,:)
   INTEGER           :: shifting_scheme = 1

   !Luis 4: changes related to new TB scheme
   INTEGER           :: ncols, icounter
   INTEGER,        ALLOCATABLE :: mask_indx(:)

   !Luis 5: Print projectability
   COMPLEX(dbl),   ALLOCATABLE :: ztmp1(:)

   !Luis 6: Normalization of the trial vectors
  
!#if defined __WRITE_ASCIIHAM
   CHARACTER(100)    :: kham_file
   INTEGER           :: iw,jw

   REAL(dbl), ALLOCATABLE :: tau(:,:)
   INTEGER,   ALLOCATABLE :: ityp(:)
   INTEGER    :: nat, nsp

   REAL(dbl) :: eps=1.d-6
   INTEGER :: hash, iunhamilt, iunsystem, ios
   CHARACTER(nstrx) :: title
   CHARACTER(LEN=9)  :: cdate, ctime
   CHARACTER(3),     ALLOCATABLE :: atm_symb(:)

   logical :: ions

   real(dbl), external :: cclock

!#endif
   
   !Luis 3
#if defined __SHIFT_TEST
   shifting_scheme = 2
   WRITE( stdout, "(2x, ' ATMPROJ uses the experimental shifting: ')")
#endif
   !
!------------------------------
! main body
!------------------------------
!
   CALL timing( subname, OPR='start' )
   CALL log_push( subname )


   !
   ! re-initialize all the times
   !
   CALL atmproj_tools_init( filein, ierr )
   IF ( ierr/=0 ) CALL errore(subname,'initializing atmproj',10)

   !
   ! search for units indipendently of io_module
   !
   CALL iotk_free_unit( iunit )
   CALL iotk_free_unit( ounit )

   !
   ! what files are to be written
   !
   write_ham = .FALSE.
   write_space = .FALSE.
   write_loc = .FALSE.
   IF ( PRESENT(fileham) )    write_ham = .TRUE.
   IF ( PRESENT(filespace) )  write_space = .TRUE.
   IF ( PRESENT(filewan) )    write_loc = .TRUE.

   !
   ! orthogonalization controlled by input
   ! NOTE: states are non-orthogonal by default
   !
   do_orthoovp_ = .FALSE.
   IF ( PRESENT( do_orthoovp ) ) do_orthoovp_ = do_orthoovp
   

!
!---------------------------------
! read data from filein (by projwfc, QE suite)
!---------------------------------
!
 
   !
   ! get lattice information
   !
   !CALL qexml_init( iunit, DIR=savedir)
   ! MBN
   CALL qexml_openfile( file_data, "read", IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname,"opening "//TRIM(file_data), ABS(ierr) )
   ! 
   CALL qexml_read_cell( ALAT=alat, A1=avec(:,1), A2=avec(:,2), A3=avec(:,3), &
                                    B1=bvec(:,1), B2=bvec(:,2), B3=bvec(:,3), IERR=ierr )
   CALL qexml_read_ions( NAT=nat, NSP=nsp, IERR=ierr)
   ALLOCATE(tau(3,nat))
   ALLOCATE(atm_symb(nsp))
   ALLOCATE(ityp(nat))
   CALL qexml_read_ions( ATM=atm_symb, ITYP=ityp, &
                                  TAU=tau,  IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname,"reading avec, bvec", ABS(ierr) )
   !
   CALL qexml_closefile( "read", IERR=ierr )
   IF ( ierr/=0 ) CALL errore(subname,"closing "//TRIM(file_data), ABS(ierr) )

   !
   ! bvec is in 2pi/a units
   ! convert it to bohr^-1
   !
   bvec = bvec * TPI / alat


   !
   ! reading dimensions
   ! and small data
   !

   write(*,*) 'Begins atmproj_read_ext' !Luis 2
   CALL atmproj_read_ext( filein, nbnd, nkpts, nspin, natomwfc, &
                          nelec, efermi, energy_units, IERR=ierr)
   write(*,*) 'Ends atmproj_read_ext' !Luis 2

   IF ( ierr/=0 ) CALL errore(subname, "reading dimensions I", ABS(ierr))

   dimwann = natomwfc
   !
   atmproj_nbnd_ = nbnd
   IF ( atmproj_nbnd > 0 ) atmproj_nbnd_ = MIN(atmproj_nbnd, nbnd)
  
   atmproj_do_norm_ = atmproj_do_norm  !Luis 6

   ! Marcio - surface bandstructure - aligning the fermi level to the bulk one.
   if (surface) then
     efermi = efermi_bulk
   end if


   !Luis 2 begin --> 
   WRITE( stdout, "(2x, ' Dimensions found in atomic_proj.{dat,xml}: ')")
   WRITE( stdout, "(2x, '   nbnd     :  ',i5 )") nbnd
   WRITE( stdout, "(2x, '   nkpts    :  ',i5 )") nkpts
   WRITE( stdout, "(2x, '   nspin    :  ',i5 )") nspin
   WRITE( stdout, "(2x, '   natomwfc :  ',i5 )") natomwfc
   WRITE( stdout, "(2x, '   nelec    :  ',f12.6)") nelec
   WRITE( stdout, "(2x, '   efermi   :  ',f12.6 )") efermi 
   WRITE( stdout, "(2x, '   energy_units :  ',a10 )") energy_units 
   WRITE( stdout, "()" )
   IF ( nspin == 4 ) THEN
      nspin_ = 1
      spin_noncollinear = .true.
   ELSE
      spin_noncollinear = .false.
      nspin_ = nspin
   END IF
   !Luis 2 end   <--

   !
   ! quick report
   !
   WRITE( stdout, "(2x, ' ATMPROJ conversion to be done using: ')")
   WRITE( stdout, "(2x, '   atmproj_nbnd :  ',i5 )") atmproj_nbnd_
   WRITE( stdout, "(2x, '   atmproj_thr  :  ',f12.6 )") atmproj_thr
   WRITE( stdout, "(2x, '   atmproj_sh   :  ',f12.6 )") atmproj_sh
   WRITE( stdout, "(2x, '   atmproj_do_norm:  ',L )") atmproj_do_norm !Luis 6
   WRITE( stdout, "()" )

   !
   ! allocations
   !
   ALLOCATE( vkpt(3,nkpts), wk(nkpts), STAT=ierr )
   IF (ierr/=0) CALL errore(subname, 'allocating vkpt, wk', ABS(ierr))
   ALLOCATE( vkpt_cry(3, nkpts), STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'allocating vkpt_cry', ABS(ierr) )
   !
   !ALLOCATE( eig(nbnd,nkpts,nspin), STAT=ierr )                       !Luis 2
   ALLOCATE( eig(nbnd,nkpts,nspin_), STAT=ierr )                       !Luis 2
   IF (ierr/=0) CALL errore(subname, 'allocating eig', ABS(ierr))
   !
   !ALLOCATE( proj(natomwfc,nbnd,nkpts,nspin), STAT=ierr )             !Luis 2
   ALLOCATE( proj(natomwfc,nbnd,nkpts,nspin_), STAT=ierr )             !Luis 2
   IF (ierr/=0) CALL errore(subname, 'allocating proj', ABS(ierr))

   !
   ! read-in massive data
   !
   write(*,*) 'Begins atmproj_read_ext --massive data' !Luis 2
   IF ( do_orthoovp_ ) THEN
       !
       ALLOCATE( kovp(1,1,1,1), STAT=ierr )
       IF (ierr/=0) CALL errore(subname, 'allocating kovp I', ABS(ierr))
       !
       write(*,*) 'Using an orthogonal basis. do_orthoovp=.true.' !Luis 2
       CALL atmproj_read_ext( filein, VKPT=vkpt, WK=wk, EIG=eig, PROJ=proj, IERR=ierr )
       !
   ELSE
       !
       ALLOCATE( kovp(natomwfc,natomwfc,nkpts,nspin), STAT=ierr )
       IF (ierr/=0) CALL errore(subname, 'allocating kovp II', ABS(ierr))
       !
       ! reading  proj(i,b)  = < phi^at_i | evc_b >
       !
       CALL atmproj_read_ext( filein, VKPT=vkpt, WK=wk, EIG=eig, PROJ=proj, KOVP=kovp, IERR=ierr )
       !
   ENDIF
   write(*,*) 'Ends atmproj_read_ext --massive data' !Luis 2
   !
   IF ( ierr/=0 ) CALL errore(subname, "reading data II", ABS(ierr))


   !
   ! units (first we convert to bohr^-1, 
   ! then we want vkpt to be re-written in crystal units)
   !
   vkpt = vkpt * TPI / alat
   !
   vkpt_cry = vkpt
   CALL cart2cry( vkpt_cry, bvec ) 
   !
   CALL get_monkpack( nk, shift, nkpts, vkpt_cry, 'CRYSTAL', bvec, ierr)
   nk(1)=1
   nk(2)=1
   nk(3)=4
   !IF ( ierr/=0 ) CALL errore(subname,'kpt grid not Monkhorst-Pack',ABS(ierr))

   !
   ! check the normalization of the weights
   !
   norm   = SUM( wk )
   wk(:)  = wk(:) / norm


   !
   ! kpts and real-space lattice vectors
   !
   nr(1:3) = nk(1:3)
   !
   ! get the grid dimension
   !
   CALL grids_get_rgrid(nr, NRTOT=nrtot )
   !
   ALLOCATE( ivr(3, nrtot), vr(3, nrtot), STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'allocating ivr, vr', ABS(ierr) )
   ALLOCATE( wr(nrtot), STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'allocating wr', ABS(ierr) )
   !
   CALL grids_get_rgrid(nr, WR=wr, IVR=ivr )
   !
   vr(:,:) = REAL( ivr, dbl)
   CALL cry2cart( vr, avec)
 
   !
   ! efermi and eigs are converted to eV's
   !
   CALL change_case( energy_units, 'lower' )
   !
   SELECT CASE( ADJUSTL(TRIM(energy_units)) )
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
      CALL errore( subname, 'unknown units for efermi: '//TRIM(energy_units), 72)
   END SELECT

   !
   ! shifting
   !
   ! apply the energy shift,
   ! meant to set the zero of the energy scale (where we may have
   ! spurious 0 eigenvalues) far from any physical energy region of interest
   !
   !eig    = eig    -atmproj_sh                                        !Luis
   !efermi = efermi -atmproj_sh                                        !Luis
   eig    = eig  -efermi                                               !Luis


   ! 
   ! build the Hamiltonian in real space
   ! 
   write_ham=.TRUE.
   IF ( write_ham ) THEN

       !ALLOCATE( rham(dimwann, dimwann, nrtot, nspin), STAT=ierr )    !Luis 2
       ALLOCATE( rham(dimwann, dimwann, nrtot, nspin_), STAT=ierr )    !Luis 2
       IF ( ierr/=0 ) CALL errore(subname, 'allocating rham', ABS(ierr) )
       ALLOCATE( kham(dimwann, dimwann, nkpts), STAT=ierr )
       IF ( ierr/=0 ) CALL errore(subname, 'allocating kham', ABS(ierr) )
       !
       IF ( .NOT. do_orthoovp_ ) THEN
           ALLOCATE( rovp(natomwfc,natomwfc,nrtot,nspin), STAT=ierr )
           IF (ierr/=0) CALL errore(subname, 'allocating rovp', ABS(ierr))
       ENDIF
       
          
       !DO isp = 1, nspin                                              !Luis 2
       DO isp = 1, nspin_                                              !Luis 2

           IF ( TRIM(spin_component) == "up"   .AND. isp == 2 ) CYCLE
           IF ( TRIM(spin_component) == "down" .AND. isp == 1 ) CYCLE
           IF ( TRIM(spin_component) == "dw"   .AND. isp == 1 ) CYCLE


           !
           ! build kham
           !
           kpt_loop:&
           DO ik = 1, nkpts
               !
               kham(:,:,ik) = ZERO
               !
               ALLOCATE( ztmp(natomwfc,nbnd), STAT=ierr)
               IF ( ierr/=0 ) CALL errore(subname,'allocating ztmp', ABS(ierr) )
               !
               ! ztmp(i, b) = < phi^at_i | evc_b >
               !
               ztmp(1:natomwfc,1:nbnd) = proj(1:natomwfc,1:nbnd,ik,isp) 
       
               !Luis 3       
               IF ( shifting_scheme .EQ. 2 ) THEN
                   !Luis 4
                   !Finds the dimensions
                   ncols = 0
                   DO ib = 1, nbnd
                      if (eig(ib,ik,isp) < atmproj_sh) then
                         ncols=ncols+1
                      endif
                   ENDDO
                   IF (ncols == 0) THEN
                      WRITE(*,*) 'No eigenvalues are below shifting at ik=',ik,'. Stopping ...'
                      STOP
                   ENDIF

                   ALLOCATE( mask_indx(ncols), STAT=ierr )
                   IF ( ierr/=0 ) CALL errore(subname, 'allocating mask_indx', ABS(ierr))

                   ALLOCATE( A(natomwfc, ncols), STAT=ierr )
                   IF ( ierr/=0 ) CALL errore(subname, 'allocating Space-A Projector', ABS(ierr))
                   ALLOCATE( PA(ncols, ncols), STAT=ierr )
                   IF ( ierr/=0 ) CALL errore(subname, 'allocating Space-A Projector', ABS(ierr))
                   ALLOCATE( IPA(ncols, ncols), STAT=ierr )
                   IF ( ierr/=0 ) CALL errore(subname, 'allocating inv(P_A)', ABS(ierr))
                   ALLOCATE( E(ncols, ncols), STAT=ierr )
                   IF ( ierr/=0 ) CALL errore(subname, 'allocating E', ABS(ierr))
                   ALLOCATE( kham_aux(ncols, natomwfc), STAT=ierr )
                   IF ( ierr/=0 ) CALL errore(subname, 'allocating kham_aux', ABS(ierr))
                   
                   E  = CZERO 
                   kham_aux = CZERO
                   icounter = 1
                   DO ib = 1, nbnd
                      if (eig(ib,ik,isp) < atmproj_sh) then
                         E(icounter,icounter) = eig(ib,ik,isp) 
                         mask_indx(icounter)  = ib
                         icounter = icounter + 1
                      endif
                   ENDDO

                   A = proj(1:natomwfc,mask_indx,ik,isp) 
                    
                   !Luis 6 
                   DO ib =1, ncols
                      if (atmproj_do_norm) then
                         proj_wgt = REAL(DOT_PRODUCT(A(:,ib),A(:,ib)))
                         A(:,ib) = A(:,ib)/SQRT(proj_wgt)
                      endif
                   ENDDO

                   PA = ZERO
                   IPA= ZERO
                   !PA = A' * A
                   CALL mat_mul( PA, A, 'C', A, 'N', ncols, ncols, natomwfc)
                   CALL mat_inv( ncols, PA, IPA)
               
                   !HKS_aux = (E - kappa*IPA)*A'
                   CALL mat_mul( kham_aux, E -atmproj_sh*IPA, 'N', A, 'C', ncols, natomwfc, ncols)
                   !CALL mat_mul( kham_aux, E, 'N', A, 'C', ncols, natomwfc, ncols)
                   !HKS = A*HKS_aux = A*(E - kappa*IPA)*A'
                   CALL mat_mul( kham(:,:,ik), A, 'N', kham_aux, 'N', natomwfc, natomwfc, ncols)
                   do i=1,natomwfc
                      do j=1,i-1 
                         kham(i,j,ik) = conjg(kham(j,i,ik))
                      enddo
                   enddo
                   DEALLOCATE( mask_indx )
                   DEALLOCATE( A )
                   DEALLOCATE( PA )
                   DEALLOCATE( IPA )
                   DEALLOCATE( E )
                   DEALLOCATE( kham_aux )

               ELSE
               !
               ibnd_loop:&
               DO ib = 1, atmproj_nbnd_
                   !
                   ! filtering
                   ! Note: - This is one way of doing the filtering.
                   !       It filters within the atmproj_nbnd bands.
                   !       Useful if used with atmproj_bnd == Inf (i.e. QE's nbnd)
                   !       so that the filter is controlled only by atmproj_thr
                   !
                   !       - Another way is to set atmproj_thr<=0 
                   !       and then the filtering is controlled only by atmproj_nbnd
                   IF ( atmproj_thr > 0.0d0 ) THEN
                       !
                       proj_wgt = REAL( DOT_PRODUCT( ztmp(:,ib), ztmp(:,ib) ) )
                       !IF ( proj_wgt < atmproj_thr ) CYCLE ibnd_loop  !Luis 4
                       IF ( eig(ib,ik,isp) > atmproj_sh ) CYCLE ibnd_loop 
                                                                       !Luis 4

                       !
                   ENDIF


                   !Luis 6 
                   if (atmproj_do_norm) then
                      ztmp(:,ib) = ztmp(:,ib)/SQRT(proj_wgt)
                   endif
                   !
                   !
                   DO j = 1, dimwann
                   DO i = 1, dimwann
                       !
                       kham(i,j,ik) = kham(i,j,ik) + &
                       !                         ( ztmp(i,ib) ) * eig(ib,ik,isp) * CONJG( ztmp(j,ib) )
                                                                       !Luis
                                                ( ztmp(i,ib) ) * (eig(ib,ik,isp)-atmproj_sh) * CONJG( ztmp(j,ib) )
                                                                       !Luis
                       !
                   ENDDO
                   ENDDO
                   !
               ENDDO ibnd_loop

               ENDIF
               !
               IF ( .NOT. mat_is_herm( dimwann, kham(:,:,ik), TOLL=EPS_m8 ) ) &
                   CALL errore(subname,'kham not hermitian',10)
               !
               DEALLOCATE( ztmp, STAT=ierr)
               IF ( ierr/=0 ) CALL errore(subname,'deallocating ztmp', ABS(ierr) )


               !
               ! overlaps
               ! projections are read orthogonals, if non-orthogonality
               ! is required, we multiply by S^1/2
               !
               IF ( .NOT. do_orthoovp_ ) THEN
                   !
                   ALLOCATE( zaux(dimwann,dimwann), ztmp(dimwann,dimwann), STAT=ierr )
                   IF ( ierr/=0 ) CALL errore(subname, 'allocating raux-rtmp', ABS(ierr) )
                   ALLOCATE( w(dimwann), kovp_sq(dimwann,dimwann), STAT=ierr )
                   IF ( ierr/=0 ) CALL errore(subname, 'allocating w, kovp_sq', ABS(ierr) )
                   !
                   CALL mat_hdiag( zaux, w(:), kovp(:,:,ik,isp), dimwann)
                   !              
                   DO i = 1, dimwann
                       !
                       IF ( w(i) <= ZERO ) CALL errore(subname,'unexpected eig < = 0 ',i)
                       w(i) = SQRT( w(i) )
                       !
                   ENDDO
                   !
                   DO j = 1, dimwann
                   DO i = 1, dimwann
                       !
                       ztmp(i,j) = zaux(i,j) * w(j)
                       !
                   ENDDO
                   ENDDO
                   !
                   CALL mat_mul( kovp_sq, ztmp, 'N', zaux, 'C', dimwann, dimwann, dimwann)
                   !
                   IF ( .NOT. mat_is_herm( dimwann, kovp_sq, TOLL=EPS_m8 ) ) &
                       CALL errore(subname,'kovp_sq not hermitean',10)

                   !
                   ! apply the basis change to the Hamiltonian
                   ! multiply kovp_sq (S^1/2) to the right and the left of kham
                   !
                   CALL mat_mul( zaux, kovp_sq,      'N', kham(:,:,ik), 'N', dimwann, dimwann, dimwann)
                   CALL mat_mul( kham(:,:,ik), zaux, 'N', kovp_sq,      'N', dimwann, dimwann, dimwann)
                   !
                   !
                   DEALLOCATE( zaux, ztmp, STAT=ierr)
                   IF ( ierr/=0 ) CALL errore(subname,'deallocating zaux, ztmp',ABS(ierr))
                   DEALLOCATE( w, kovp_sq, STAT=ierr)
                   IF ( ierr/=0 ) CALL errore(subname,'deallocating w, kovp_sq',ABS(ierr))
                   !
               ENDIF
               
               !
               ! fermi energy is taken into accout
               ! The energy shift is performed on the final matrix
               ! and not on the DFT eigenvalues            
               ! as     eig(:,:,:) = eig(:,:,:) -efermi
               ! because the atomic basis is typically very large 
               ! and would require a lot of bands to be described
               !
               IF ( .NOT. do_orthoovp_ ) THEN
                   !
                   DO j = 1, dimwann
                   DO i = 1, dimwann
                       !kham(i,j,ik) = kham(i,j,ik) -efermi * kovp(i,j,ik,isp)
                                                                       !Luis
                       kham(i,j,ik) = kham(i,j,ik) + atmproj_sh * kovp(i,j,ik,isp)
                                                                       !Luis
                   ENDDO
                   ENDDO
                   !
               ELSE
                   !
                   DO i = 1, dimwann
                       !kham(i,i,ik) = kham(i,i,ik) -efermi            !Luis
                       kham(i,i,ik) = kham(i,i,ik) + atmproj_sh        !Luis
                   ENDDO
                   !
               ENDIF
               !
           ENDDO kpt_loop

!#if defined __WRITE_ASCIIHAM
           !Luis 5 
           !print projectabilities
           write(*,*) 'Prints projectabilities'
           if (isp == 1 .and. nspin_==1) kham_file = "projectability.txt"
           if (isp == 1 .and. nspin_==2) kham_file = "projectability_up.txt"
           if (isp == 2) kham_file = "projectability_dn.txt"
           OPEN (unit = 14, file = trim(kham_file))
           ALLOCATE( ztmp1(natomwfc), STAT=ierr)
           IF ( ierr/=0 ) CALL errore(subname,'allocating ztmp1', ABS(ierr) )

           DO ik = 1, nkpts
               DO ib = 1, nbnd
                  ztmp1(1:natomwfc) = proj(1:natomwfc,ib,ik,isp) 
                  WRITE(14,"(2f20.13)") eig(ib,ik,isp),REAL(DOT_PRODUCT(ztmp1,ztmp1))
               ENDDO
           ENDDO 
           CLOSE(14)
           DEALLOCATE(ztmp1)


           !print Hamiltonians 
           if (isp == 1 .and. nspin_==1) kham_file = "kham.txt"
           if (isp == 1 .and. nspin_==2) kham_file = "kham_up.txt"
           if (isp == 2) kham_file = "kham_down.txt"

           IF (isp ==1) THEN !
              OPEN (unit = 14, file = "k.txt")
              DO ik =1, nkpts
                 WRITE(14,"(3f20.13)") vkpt_cry(1,ik), vkpt_cry(2,ik), vkpt_cry(3,ik)
              ENDDO
              CLOSE(14)

              OPEN (unit = 14, file = "wk.txt")
              DO ik =1, nkpts
                 WRITE(14,"(f20.13)") wk(ik)
              ENDDO
              CLOSE(14)

              OPEN (unit = 14, file = "kovp.txt")
              DO ik =1, nkpts
                  DO iw=1,dimwann
                     DO jw=1,dimwann
                        WRITE(14,"(2f20.13)") real(kovp(iw,jw,ik,isp)),aimag(kovp(iw,jw,ik,isp))
                     ENDDO
                  ENDDO
              ENDDO
              CLOSE(14)
           ENDIF

           OPEN (unit = 14, file = trim(kham_file))
           DO ik =1, nkpts
               DO iw=1,dimwann
                  DO jw=1,dimwann
                     WRITE(14,"(2f20.13)") real(kham(iw,jw,ik)),aimag(kham(iw,jw,ik))
                  ENDDO
               ENDDO
           ENDDO
           CLOSE(14)

  !print Hamiltonians in AMULET format (MBN)

  iunhamilt = 14
  title = 'AMULET'
  hash = cclock()
  OPEN (iunhamilt, file = 'hamilt.am', status = 'unknown', form = 'formatted',err = 300, iostat = ios)
300 CALL errore ('HMLT', 'Opening hamilt', abs (ios) )

  CALL date_and_tim(cdate,ctime)
  write(iunsystem,'(a30,2a10/)') '# This file was generated on: ', cdate,ctime
  IF( trim(title) .NE. '' ) write(iunhamilt,'(a2,a80/)') '# ', title

  WRITE(iunhamilt,'(a5)') '&hash'
  WRITE(iunhamilt,*) hash
  WRITE(iunhamilt,*)

  WRITE(iunhamilt,'(a6)') '&nspin'
  WRITE(iunhamilt,'(i1)') nspin
  WRITE(iunhamilt,*)

  WRITE(iunhamilt,'(a4)') '&nkp'
  WRITE(iunhamilt,'(i5)') nkpts/nspin
  WRITE(iunhamilt,*)

  WRITE(iunhamilt,'(a4)') '&dim'
  WRITE(iunhamilt,'(i3)') dimwann
  WRITE(iunhamilt,*)

  WRITE(iunhamilt,'(a8)') '&kpoints'
  DO ik=1, nkpts/nspin
    WRITE(iunhamilt,'(f15.12,3f9.5)') wk(ik), vkpt_cry(:,ik)
  END DO
  WRITE(iunhamilt,*)

  WRITE(iunhamilt,'(a12)') '&hamiltonian'

  ALLOCATE(hamk2(dimwann,dimwann))

  DO ik = 1, nkpts

     hamk2 = kham(:,:,ik) !* rytoev

     DO i=1, dimwann
        DO j=i, dimwann

           hr = abs(dreal(hamk2(i,j)))
           hi = abs(dimag(hamk2(i,j)))
           IF((hr>=eps).and.(hi>=eps)) WRITE(iunhamilt,'(2f13.8)') dreal(hamk2(i,j)), aimag(hamk2(i,j))
           IF ((hr<eps).and.(hi>=eps)) WRITE(iunhamilt,'(f3.0,f13.8)') 0., aimag(hamk2(i,j))
           IF ((hr>=eps).and.(hi<eps)) WRITE(iunhamilt,'(f13.8,f3.0)') dreal(hamk2(i,j)), 0.
           IF ((hr<eps).and.(hi<eps)) WRITE(iunhamilt,'(2f3.0)') 0., 0.

        ENDDO
     ENDDO

  ENDDO
  DEALLOCATE(hamk2)
  CLOSE(iunhamilt)

  !print system data in AMULET format (MBN)
  iunsystem=14

!
! Getting DFT data
!
  OPEN (iunsystem, file = 'system.am', status = 'unknown', form = 'formatted',err = 301, iostat = ios)
301 CALL errore ('HMLT', 'Opening system.am', abs (ios) )

  CALL date_and_tim(cdate,ctime)
  write(iunsystem,'(a30,2a10/)') '# This file was generated on: ', cdate,ctime
  IF( trim(title) .NE. '' ) write(iunsystem,'(a2,a80/)') '# ', title

  WRITE(iunsystem,'(a5)') '&hash'
  WRITE(iunsystem,*) hash
  WRITE(iunsystem,*)

  WRITE(iunsystem,'(a5)') '&cell'
  WRITE(iunsystem,'(f12.9)') alat
  DO i=1,3
    WRITE(iunsystem,'(3f9.5)') avec(:,i)/alat
  END DO

  WRITE(iunsystem,'(a6)') '&atoms'
  WRITE(iunsystem,'(i5)') nat
  DO i=1, nat
    WRITE(iunsystem,'(a4,1x,3f9.5)') atm_symb(ityp(i)), tau(:,i)/alat
  END DO
  WRITE(iunsystem,*)

  WRITE(iunsystem,'(a6)') '&nelec'
  WRITE(iunsystem,'(f7.2)') nelec
  WRITE(iunsystem,*)

  WRITE(iunsystem,'(a7)') '&efermi'
  WRITE(iunsystem,'(f8.4)') 0.0 ! Everything has been already shifted.
  WRITE(iunsystem,*)

  WRITE(iunsystem,'(a20)') '# Basis description:'
  WRITE(iunsystem,'(a14)') '# dim, nblocks'
  WRITE(iunsystem,'(a74)') '# atom_sym, atom_num, l_sym, block_dim, block_start, orbitals(1:block_dim)'
  WRITE(iunsystem,'(a6)') '&basis'
!  WRITE(iunsystem,'(i2,i4)') dimwann, nblocks
!  DO i=1, nblocks
!    WRITE(iunsystem,'(a3,i3,a2,i2,i4,4x,7i2)') atm(ityp(block_atom(i))),block_atom(i), l_symb(block_l(i)+1), &
!                       block_dim(i), block_start(i), (orbitals(wan_in(j,1)%ing(1)%m,block_l(i)+1), &
!                        j=block_wannier(i,1), block_wannier(i,block_dim(i)) )
!  END DO

  !WRITE(iunsystem,*)

  CLOSE(iunsystem)

!#endif


           ! 
           ! convert to real space
           !
           DO ir = 1, nrtot
               !
               CALL compute_rham( dimwann, vr(:,ir), rham(:,:,ir,isp), &
                                  nkpts, vkpt, wk, kham )
               !
               IF ( .NOT. do_orthoovp_ ) THEN
                   !
                    CALL compute_rham( dimwann, vr(:,ir), rovp(:,:,ir,isp), &
                                       nkpts, vkpt, wk, kovp(:,:,:,isp) )
                   !
               ENDIF
               !
           ENDDO
           !
       ENDDO 
 
       ! 
       DEALLOCATE( kham, STAT=ierr )
       IF ( ierr/=0 ) CALL errore(subname, 'deallocating kham', ABS(ierr) )
       !
   ENDIF
   !
   IF ( ALLOCATED( kovp ) ) THEN
       DEALLOCATE( kovp, STAT=ierr)
       IF ( ierr/=0 ) CALL errore(subname, 'deallocating kovp', ABS(ierr) )
   ENDIF


!
!---------------------------------
! write to fileout (internal fmt)
!---------------------------------
!
   IF ( write_ham ) THEN
       !
       CALL iotk_open_write( ounit, FILE=TRIM(fileham), BINARY=binary )
       CALL iotk_write_begin( ounit, "HAMILTONIAN" )
       !
       !
       CALL iotk_write_attr( attr,"dimwann",dimwann,FIRST=.TRUE.)
       CALL iotk_write_attr( attr,"nkpts",nkpts)
       CALL iotk_write_attr( attr,"nspin",nspin)
       CALL iotk_write_attr( attr,"spin_component",TRIM(spin_component))
       CALL iotk_write_attr( attr,"nk",nk)
       CALL iotk_write_attr( attr,"shift",shift)
       CALL iotk_write_attr( attr,"nrtot",nrtot)
       CALL iotk_write_attr( attr,"nr",nr)
       CALL iotk_write_attr( attr,"have_overlap", .NOT. do_orthoovp_ )
       CALL iotk_write_attr( attr,"fermi_energy", 0.0_dbl )
       CALL iotk_write_empty( ounit,"DATA",ATTR=attr)
       !
       CALL iotk_write_attr( attr,"units","bohr",FIRST=.TRUE.)
       CALL iotk_write_dat( ounit,"DIRECT_LATTICE", avec, ATTR=attr, COLUMNS=3)
       !
       CALL iotk_write_attr( attr,"units","bohr^-1",FIRST=.TRUE.)
       CALL iotk_write_dat( ounit,"RECIPROCAL_LATTICE", bvec, ATTR=attr, COLUMNS=3)
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
          IF ( TRIM(spin_component) == "up"   .AND. isp == 2 ) CYCLE
          IF ( TRIM(spin_component) == "down" .AND. isp == 1 ) CYCLE
          IF ( TRIM(spin_component) == "dw"   .AND. isp == 1 ) CYCLE
          !
          IF ( TRIM(spin_component) == "all" .AND. nspin == 2 ) THEN
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
              CALL iotk_write_dat( ounit,"VR"//TRIM(iotk_index(ir)), rham(:,:, ir, isp) )
              !
              IF ( .NOT. do_orthoovp_ ) THEN
                  CALL iotk_write_dat( ounit,"OVERLAP"//TRIM(iotk_index(ir)), &
                                       rovp( :, :, ir, isp) )
              ENDIF
              !
              !
          ENDDO
          !
          CALL iotk_write_end( ounit,"RHAM")
          !
          IF ( nspin == 2 .AND. TRIM(spin_component) == "all" ) THEN
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
       CALL iotk_open_write( ounit, FILE=TRIM(filespace), BINARY=binary )
       !
       !
       CALL iotk_write_begin( ounit, "WINDOWS" )
       !
       !
       CALL iotk_write_attr( attr,"nbnd",atmproj_nbnd_,FIRST=.TRUE.)
       CALL iotk_write_attr( attr,"nkpts",nkpts)
       CALL iotk_write_attr( attr,"nspin",nspin)
       CALL iotk_write_attr( attr,"spin_component",TRIM(spin_component))
       CALL iotk_write_attr( attr,"efermi", 0.0_dbl )
       CALL iotk_write_attr( attr,"dimwinx", atmproj_nbnd_ )
       CALL iotk_write_empty( ounit,"DATA",ATTR=attr)
       !
       ALLOCATE( itmp(nkpts), STAT=ierr )
       IF ( ierr/=0 ) CALL errore(subname, 'allocating itmp', ABS(ierr))
       ALLOCATE( eamp_tmp(atmproj_nbnd_,dimwann), STAT=ierr )
       IF ( ierr/=0 ) CALL errore(subname, 'allocating eamp_tmp', ABS(ierr))
       !
       itmp(:) = atmproj_nbnd_
       CALL iotk_write_dat( ounit, "DIMWIN", itmp, COLUMNS=8 )
       itmp(:) = 1
       CALL iotk_write_dat( ounit, "IMIN", itmp, COLUMNS=8 )
       itmp(:) = atmproj_nbnd_
       CALL iotk_write_dat( ounit, "IMAX", itmp, COLUMNS=8 )
       !
       DO isp = 1, nspin
           !
           IF ( nspin == 2 ) THEN
               CALL iotk_write_begin( ounit, "SPIN"//TRIM(iotk_index(isp)) )
           ENDIF
           !
           CALL iotk_write_dat( ounit, "EIG", eig(1:atmproj_nbnd_,:,isp), COLUMNS=4)
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
       CALL iotk_write_attr( attr,"dimwinx",atmproj_nbnd_,FIRST=.TRUE.)
       CALL iotk_write_attr( attr,"nkpts",nkpts)
       CALL iotk_write_attr( attr,"dimwann", dimwann)
       CALL iotk_write_empty( ounit,"DATA",ATTR=attr)
       !
       itmp(:) = atmproj_nbnd_
       CALL iotk_write_dat( ounit, "DIMWIN", itmp, COLUMNS=8 )
       !
       DO isp = 1, nspin
           !
           IF ( nspin == 2 ) THEN
               CALL iotk_write_begin( ounit, "SPIN"//TRIM(iotk_index(isp)) )
           ENDIF
           !
           DO ik = 1, nkpts
               !
               eamp_tmp(1:atmproj_nbnd_, 1:dimwann) = &
                        CONJG(TRANSPOSE(proj(1:dimwann,1:atmproj_nbnd_,ik,isp) )) 
               !
               DO ib = 1, atmproj_nbnd_
                   proj_wgt = REAL( DOT_PRODUCT( proj(:,ib,ik,isp ), proj(:,ib,ik,isp ) ) )
                   IF ( proj_wgt < atmproj_thr ) eamp_tmp( ib, :) = 0.0d0
               ENDDO
               !
               CALL iotk_write_dat( ounit, "EAMP"//TRIM(iotk_index(ik)), eamp_tmp )
               !
           ENDDO
           !
           IF ( nspin == 2 ) THEN
               CALL iotk_write_end( ounit, "SPIN"//TRIM(iotk_index(isp)) )
           ENDIF
           !
       ENDDO
       !
       CALL iotk_write_end( ounit, "SUBSPACE" )
       !
       !
       CALL iotk_close_write( ounit )
       !
       DEALLOCATE( itmp, eamp_tmp, STAT=ierr )
       IF ( ierr/=0 ) CALL errore(subname, 'deallocating itmp, eamp', ABS(ierr))
       !
   ENDIF


   IF ( write_loc ) THEN
       !
       CALL iotk_open_write( ounit, FILE=TRIM(filewan), BINARY=binary )
       !
       CALL iotk_write_begin( ounit, "WANNIER_LOCALIZATION" )
       !
       CALL iotk_write_attr( attr,"dimwann",dimwann,FIRST=.TRUE.)
       CALL iotk_write_attr( attr,"nkpts",nkpts)
       CALL iotk_write_empty( ounit,"DATA",ATTR=attr)
       !
       CALL iotk_write_attr( attr,"Omega_I",0.0d0,FIRST=.TRUE.)
       CALL iotk_write_attr( attr,"Omega_D",0.0d0)
       CALL iotk_write_attr( attr,"Omega_OD",0.0d0)
       CALL iotk_write_attr( attr,"Omega_tot",0.0d0)
       CALL iotk_write_empty( ounit,"SPREADS",ATTR=attr)
       !
       ALLOCATE( rtmp(3,dimwann), STAT=ierr )
       IF ( ierr/=0 ) CALL errore(subname, 'allocating rtmp', ABS(ierr))
       ALLOCATE( cu_tmp(dimwann,dimwann,nkpts), STAT=ierr )
       IF ( ierr/=0 ) CALL errore(subname, 'allocating cu_tmp', ABS(ierr))
       !
       cu_tmp(:,:,:) = 0.0d0
       !
       DO ik = 1, nkpts
       DO i  = 1, dimwann
           cu_tmp(i,i,ik) = 1.0d0
       ENDDO
       ENDDO
       !
       rtmp(:,:)=0.0d0
       !
       CALL iotk_write_dat(ounit,"CU",cu_tmp)
       CALL iotk_write_dat(ounit,"RAVE",rtmp,COLUMNS=3)
       CALL iotk_write_dat(ounit,"RAVE2",rtmp(1,:))
       CALL iotk_write_dat(ounit,"R2AVE",rtmp(1,:))
       !
       DEALLOCATE( rtmp, cu_tmp, STAT=ierr )
       IF ( ierr/=0 ) CALL errore(subname, 'deallocating rtmp, cu_tmp', ABS(ierr))
       !
       CALL iotk_write_end( ounit, "WANNIER_LOCALIZATION" )
       !
       CALL iotk_close_write( ounit )
       !
   ENDIF




!
! local cleaning
!

   DEALLOCATE( proj, STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'deallocating proj', ABS(ierr) )
   !
   DEALLOCATE( vkpt, wk, STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'deallocating vkpt, wk', ABS(ierr) )
   DEALLOCATE( vkpt_cry, STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'deallocating vkpt_cry', ABS(ierr) )
   !
   DEALLOCATE( ivr, wr, STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'deallocating ivr, wr', ABS(ierr) )
   DEALLOCATE( vr, STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'deallocating vr', ABS(ierr) )
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
END SUBROUTINE atmproj_to_internal


!**********************************************************
   LOGICAL FUNCTION file_is_atmproj( filename )
   !**********************************************************
   !
   IMPLICIT NONE
     !
     ! check for atmproj fmt
     !
     CHARACTER(*)     :: filename
     !
     INTEGER          :: nbnd, nkpts, nspin, natomwfc
     INTEGER          :: ierr
     LOGICAL          :: lerror
    

     file_is_atmproj = .FALSE.
     lerror = .FALSE.
     !
     IF ( .NOT. init ) THEN 
         !
         CALL atmproj_tools_init( filename, ierr )
         IF ( ierr/=0 ) lerror = .TRUE. 
         !
     ENDIF
     !
     IF ( lerror ) RETURN
     !
     CALL atmproj_read_ext( filename, NBND=nbnd, NKPT=nkpts, &
                            NSPIN=nspin, NATOMWFC=natomwfc,  IERR=ierr )
     IF ( ierr/=0 ) lerror = .TRUE.

     IF ( lerror ) RETURN
     !
     file_is_atmproj = .TRUE.
     !
  END FUNCTION file_is_atmproj


!*************************************************
SUBROUTINE atmproj_read_ext ( filein, nbnd, nkpt, nspin, natomwfc, nelec, &
                              efermi, energy_units, vkpt, wk, eig, proj, kovp, ierr )
   !*************************************************
   !
   USE kinds,          ONLY : dbl 
   USE iotk_module
   !
   IMPLICIT NONE
   !
   CHARACTER(*),           INTENT(IN)   :: filein
   INTEGER,      OPTIONAL, INTENT(OUT)  :: nbnd, nkpt, nspin, natomwfc
   REAL(dbl),    OPTIONAL, INTENT(OUT)  :: nelec, efermi
   CHARACTER(*), OPTIONAL, INTENT(OUT)  :: energy_units
   REAL(dbl),    OPTIONAL, INTENT(OUT)  :: vkpt(:,:), wk(:), eig(:,:,:)
   COMPLEX(dbl), OPTIONAL, INTENT(OUT)  :: proj(:,:,:,:)
   COMPLEX(dbl), OPTIONAL, INTENT(OUT)  :: kovp(:,:,:,:)
   INTEGER,                INTENT(OUT)  :: ierr
   !
   !
   CHARACTER(256)    :: attr, str
   INTEGER           :: iunit
   INTEGER           :: ik, isp, ias
   !
   INTEGER           :: nbnd_, nkpt_, nspin_, natomwfc_ 
   REAL(dbl)         :: nelec_, efermi_
   CHARACTER(20)     :: energy_units_
   !
   COMPLEX(dbl), ALLOCATABLE :: ztmp(:,:)


   CALL iotk_free_unit( iunit )
   ierr = 0

   CALL iotk_open_read( iunit, FILE=TRIM(filein), IERR=ierr )
   IF ( ierr/=0 ) RETURN
   !
   !
   CALL iotk_scan_begin( iunit, "HEADER", IERR=ierr) 
   IF ( ierr/=0 ) RETURN
   !
   CALL iotk_scan_dat( iunit, "NUMBER_OF_BANDS", nbnd_, IERR=ierr) 
   IF ( ierr/=0 ) RETURN
   CALL iotk_scan_dat( iunit, "NUMBER_OF_K-POINTS", nkpt_, IERR=ierr) 
   IF ( ierr/=0 ) RETURN
   CALL iotk_scan_dat( iunit, "NUMBER_OF_SPIN_COMPONENTS", nspin_, IERR=ierr) 
   IF ( ierr/=0 ) RETURN
   CALL iotk_scan_dat( iunit, "NUMBER_OF_ATOMIC_WFC", natomwfc_, IERR=ierr) 
   IF ( ierr/=0 ) RETURN
   CALL iotk_scan_dat( iunit, "NUMBER_OF_ELECTRONS", nelec_, IERR=ierr) 
   IF ( ierr/=0 ) RETURN
   !
   CALL iotk_scan_empty( iunit, "UNITS_FOR_ENERGY", ATTR=attr, IERR=ierr) 
   IF ( ierr/=0 ) RETURN
   CALL iotk_scan_attr( attr, "UNITS", energy_units_, IERR=ierr) 
   IF ( ierr/=0 ) RETURN
   !
   CALL iotk_scan_dat( iunit, "FERMI_ENERGY", efermi_, IERR=ierr) 
   IF ( ierr/=0 ) RETURN
   !
   CALL iotk_scan_end( iunit, "HEADER", IERR=ierr) 
   IF ( ierr/=0 ) RETURN
   
   ! 
   ! reading kpoints and weights 
   ! 
   IF ( PRESENT( vkpt ) ) THEN
       !
       CALL iotk_scan_dat( iunit, "K-POINTS", vkpt(:,:), IERR=ierr )
       IF ( ierr/=0 ) RETURN
       !
   ENDIF
   !
   IF ( PRESENT (wk) ) THEN
       !
       CALL iotk_scan_dat( iunit, "WEIGHT_OF_K-POINTS", wk(:), IERR=ierr )
       IF ( ierr/=0 ) RETURN
       !
   ENDIF
   
   ! 
   ! reading eigenvalues
   ! 

   !Luis 2 begin -->
   IF ( nspin_ == 4 ) THEN
      nspin_ = 1
   END IF
   !Luis 2 end   <--

   write(*,*) 'Begins reading eigenvalues' !Luis 2
   IF ( PRESENT( eig ) ) THEN
       ! 
       CALL iotk_scan_begin( iunit, "EIGENVALUES", IERR=ierr )
       IF ( ierr/=0 ) RETURN
       !
       !
       DO ik = 1, nkpt_
           !
           CALL iotk_scan_begin( iunit, "K-POINT"//TRIM(iotk_index(ik)), IERR=ierr )
           IF ( ierr/=0 ) RETURN
           !
           IF ( nspin_ == 1 ) THEN
                !
                isp = 1
                !
                CALL iotk_scan_dat(iunit, "EIG" , eig(:,ik, isp ), IERR=ierr)
                IF ( ierr /= 0 ) RETURN
                !
           ELSE
                !
                DO isp=1,nspin_
                   !
                   str = "EIG"//TRIM(iotk_index(isp))
                   !
                   CALL iotk_scan_dat(iunit, TRIM(str) , eig(:,ik,isp), IERR=ierr)
                   IF ( ierr /= 0 ) RETURN
                   !
                ENDDO
                !
           ENDIF       
           !
           !
           CALL iotk_scan_end( iunit, "K-POINT"//TRIM(iotk_index(ik)), IERR=ierr )
           IF ( ierr/=0 ) RETURN
           !
       ENDDO
       !
       !
       CALL iotk_scan_end( iunit, "EIGENVALUES", IERR=ierr )
       IF ( ierr/=0 ) RETURN
       !
   ENDIF
   write(*,*) 'Finished reading eigenvalues' !Luis 2


   write(*,*) 'Begins reading projections' !Luis 2
   ! 
   ! reading projections
   ! 
   IF ( PRESENT( proj ) ) THEN
       !
       ALLOCATE( ztmp(nbnd_, natomwfc_) )
       !
       CALL iotk_scan_begin( iunit, "PROJECTIONS", IERR=ierr )
       IF ( ierr/=0 ) RETURN
       !
       !
       DO ik = 1, nkpt_
           !
           !
           CALL iotk_scan_begin( iunit, "K-POINT"//TRIM(iotk_index(ik)), IERR=ierr )
           IF ( ierr/=0 ) RETURN
           !
           DO isp = 1, nspin_
               !
               IF ( nspin_ == 2 ) THEN
                   !
                   CALL iotk_scan_begin( iunit, "SPIN"//TRIM(iotk_index(isp)), IERR=ierr )
                   IF ( ierr/=0 ) RETURN
                   !
               ENDIF
               !
               DO ias = 1, natomwfc_
                   !
                   str= "ATMWFC"//TRIM( iotk_index( ias ) )
                   !
                   CALL iotk_scan_dat(iunit, TRIM(str) , ztmp( :, ias ), IERR=ierr)
                   IF ( ierr /= 0 ) RETURN
                   !
               ENDDO
               !
               proj( 1:natomwfc_, 1:nbnd_, ik, isp ) = TRANSPOSE( ztmp(1:nbnd_,1:natomwfc_) ) 
               !
               !
               IF ( nspin_ == 2 ) THEN
                   !
                   CALL iotk_scan_end( iunit, "SPIN"//TRIM(iotk_index(isp)), IERR=ierr )
                   IF ( ierr/=0 ) RETURN
                   !
               ENDIF
               !
           ENDDO
           !
           !
           CALL iotk_scan_end( iunit, "K-POINT"//TRIM(iotk_index(ik)), IERR=ierr )
           IF ( ierr/=0 ) RETURN
           !
           !
       ENDDO
       !
       DEALLOCATE( ztmp )
       !
       CALL iotk_scan_end( iunit, "PROJECTIONS", IERR=ierr )
       IF ( ierr/=0 ) RETURN
       !
   ENDIF
   write(*,*) 'Ends reading projections' !Luis 2

   ! 
   ! reading overlaps
   ! 
   IF ( PRESENT( kovp ) ) THEN
       !
       CALL iotk_scan_begin( iunit, "OVERLAPS", IERR=ierr )
       !IF ( ierr/=0 ) RETURN                                          !Luis

       IF ( ierr/=0 ) THEN                                             !Luis
         write(*,*) 'OVERLAPS data not found in file. Crashing ...'    !Luis
         RETURN                                                        !Luis
       ENDIF                                                           !Luis
       !
       !
       DO ik = 1, nkpt_
           !
           !
           CALL iotk_scan_begin( iunit, "K-POINT"//TRIM(iotk_index(ik)), IERR=ierr )
           IF ( ierr/=0 ) RETURN
           !
           DO isp = 1, nspin_
               !
               CALL iotk_scan_dat(iunit, "OVERLAP"//TRIM(iotk_index(isp)), kovp( :, :, ik, isp ), IERR=ierr)
               IF ( ierr/=0 ) RETURN
               !
           ENDDO
           !
           CALL iotk_scan_end( iunit, "K-POINT"//TRIM(iotk_index(ik)), IERR=ierr )
           IF ( ierr/=0 ) RETURN
           !
       ENDDO
       !
       CALL iotk_scan_end( iunit, "OVERLAPS", IERR=ierr )
       IF ( ierr/=0 ) RETURN
       !
   ENDIF
   !
   CALL iotk_close_read( iunit, IERR=ierr )
   IF ( ierr/=0 ) RETURN
   !
   !
   IF ( PRESENT( nbnd ) )         nbnd = nbnd_
   IF ( PRESENT( nkpt ) )         nkpt = nkpt_
   IF ( PRESENT( nspin ) )        nspin = nspin_
   IF ( PRESENT( natomwfc ) )     natomwfc = natomwfc_
   IF ( PRESENT( nelec ) )        nelec = nelec_
   IF ( PRESENT( efermi ) )       efermi = efermi_
   IF ( PRESENT( energy_units ) ) energy_units = TRIM(energy_units_)
   !
   RETURN
   !
END SUBROUTINE atmproj_read_ext


!************************************************************
INTEGER FUNCTION atmproj_get_index( i, ia, ityp, natomwfc )
   !************************************************************
   !
   IMPLICIT NONE
   !
   INTEGER       :: i, ia, ityp(*), natomwfc(*)
   !
   INTEGER       :: ind, iatm, nt
   CHARACTER(17) :: subname="atmproj_get_index"
   !
   IF ( i > natomwfc( ityp(ia)) ) CALL errore(subname,"invalid i",i)
   !
   ind = i
   DO iatm = 1, ia-1
       !
       nt = ityp(iatm)
       ind = ind + natomwfc(nt)
       !
   ENDDO
   !
   atmproj_get_index = ind
   !
END FUNCTION atmproj_get_index


!************************************************************
SUBROUTINE  atmproj_get_natomwfc( nsp, psfile, natomwfc )
   !************************************************************
   !
   IMPLICIT NONE
   !
   INTEGER,           INTENT(IN)  :: nsp
   TYPE(pseudo_upf),  INTENT(IN)  :: psfile(nsp)
   INTEGER,           INTENT(OUT) :: natomwfc(nsp)
   !
   INTEGER :: nt, nb, il
   !
   DO nt = 1, nsp
       !
       natomwfc(nt) = 0
       DO nb = 1, psfile(nt)%nwfc
           il = psfile(nt)%lchi(nb)
           IF ( psfile(nt)%oc(nb) >= 0.0d0 ) natomwfc(nt) = natomwfc(nt) + 2 * il + 1
       ENDDO
       !
   ENDDO
   !
END SUBROUTINE atmproj_get_natomwfc


END MODULE atmproj_tools_module
