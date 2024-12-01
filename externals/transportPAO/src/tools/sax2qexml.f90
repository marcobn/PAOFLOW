!
! Copyright (C) 2011 Andrea Ferretti
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!=====================================================
PROGRAM sax2qexml
   !=====================================================
   !
   ! Convert STATES and GW_QP from sax fmt into a
   ! proper qexml  prefix.save directory.
   !
   ! the qexml $prefix.save cirectory can be copied
   ! into      $prefix_$calc.save   (calc=GW,HF)
   !
   ! eigenval.xml, evc.dat 
   ! 
   ! are reformatted according to the file states
   ! or GW_QP + the existing .save directory
   !
   ! textual input is suggested
   !
   USE version_module,       ONLY : version_number
   USE kinds,                ONLY : dbl
   USE parameters,           ONLY : nstrx
   USE io_module,            ONLY : stdout, stdin
   USE io_module,            ONLY : work_dir, prefix, label => postfix
   USE control_module,       ONLY : verbosity, debug_level, use_debug_mode
   USE timing_module,        ONLY : timing
   USE log_module,           ONLY : log_push, log_pop
   USE parser_module,        ONLY : log2char, change_case
   !
   IMPLICIT NONE

   !
   ! input variables
   !
   CHARACTER(nstrx)  :: qp_file, qp_states

   NAMELIST /INPUT/  prefix, work_dir, label, qp_file, qp_states

   !   
   ! end of declariations
   !   

!
!------------------------------
! main body
!------------------------------
!
      CALL startup(version_number,'sax2qexml')

      !   
      ! read input
      !   
      CALL sax2qexml_input( )

      !
      ! do the main task 
      ! 
      CALL do_sax2qexml( prefix, work_dir, label, qp_file, qp_states )

      !
      ! finalize
      !
      CALL shutdown( 'sax2qexml' )

CONTAINS


!********************************************************
   SUBROUTINE sax2qexml_input()
   !********************************************************
   !
   ! Read INPUT namelist from stdin
   !
   USE mp,            ONLY : mp_bcast
   USE io_module,     ONLY : io_init, ionode, ionode_id
   !
   IMPLICIT NONE

      CHARACTER(15)    :: subname = 'sax2qexml_input'
      INTEGER          :: ierr
      !
      ! end of declarations
      !

      CALL timing( subname, OPR='start' )

      !
      ! init input namelist
      !
      prefix                      = ' '
      work_dir                    = ' '
      label                       = ' '
      qp_file                     = ' '
      qp_states                   = ' '

      CALL input_from_file ( stdin )
      !
      IF (ionode) READ(stdin, INPUT, IOSTAT=ierr)
      !
      CALL mp_bcast( ierr, ionode_id )
      IF ( ierr /= 0 )  CALL errore(subname,'Unable to read namelist INPUT',ABS(ierr))

      !
      ! broadcast
      !
      CALL mp_bcast( prefix,            ionode_id )
      CALL mp_bcast( work_dir,          ionode_id )
      CALL mp_bcast( label,             ionode_id )
      CALL mp_bcast( qp_file,           ionode_id )
      CALL mp_bcast( qp_states,         ionode_id )

      !
      ! summary
      !
      IF ( ionode ) THEN
          !   
          CALL write_header( stdout, "INPUT Summary" )
          !   
          WRITE( stdout, "(   7x,'              work dir :',5x,   a)") TRIM(work_dir)
          WRITE( stdout, "(   7x,'                prefix :',5x,   a)") TRIM(prefix)
          WRITE( stdout, "(   7x,'                 label :',5x,   a)") TRIM(label)
          WRITE( stdout, "(   7x,'               qp_file :',5x,   a)") TRIM(qp_file)
          WRITE( stdout, "(   7x,'             qp_states :',5x,   a)") TRIM(qp_states)
          !
      ENDIF
      !
      CALL timing(subname,OPR='stop')
      !
      RETURN
      !
   END SUBROUTINE sax2qexml_input


!************************************************************************
   SUBROUTINE do_sax2qexml( prefix, work_dir, label, qp_file, qp_states )
   !************************************************************************
   !
   ! performs the main task
   !
   USE kinds
   USE constants,            ONLY : ZERO, CZERO, TWO
   USE parameters,           ONLY : nstrx
   USE io_module,            ONLY : ionode, ionode_id, &
                                    dft_unit, sax_unit => aux1_unit, &
                                    qexml_unit => aux2_unit
   USE mp,                   ONLY : mp_bcast
   USE util_module,          ONLY : mat_mul
   USE ggrids_module,        ONLY : ggrids_gv_indexes
   USE qexml_module
   USE iotk_module
   !
   IMPLICIT NONE
      !
      ! input vars
      !
      CHARACTER(*),  INTENT(IN) :: prefix, work_dir
      CHARACTER(*),  INTENT(IN) :: label
      CHARACTER(*),  INTENT(IN) :: qp_file
      CHARACTER(*),  INTENT(IN) :: qp_states
  
      !
      ! local vars
      !
      CHARACTER(12) :: subname="do_sax2qexml"
      !
      CHARACTER(256):: filename 
      CHARACTER(256):: dirname_in, dirname_out
      CHARACTER(256):: attr
      CHARACTER(15) :: str
      !
      INTEGER       :: ib, ik, isp, ig
      INTEGER       :: nbnd, nkpts, nspin, npw
      INTEGER       :: ibnd_min, ibnd_max
      INTEGER       :: ngm, nr1, nr2, nr3
      INTEGER       :: npwkx
      LOGICAL       :: lhave_qp_states
      LOGICAL       :: lgamma_only
      REAL(dbl)     :: xk(3)
      !
      INTEGER,      ALLOCATABLE :: npwk(:)
      INTEGER,      ALLOCATABLE :: igv(:,:)
      INTEGER,      ALLOCATABLE :: igk(:,:), igksort(:)
      INTEGER,      ALLOCATABLE :: fft2gv(:)
      REAL(dbl),    ALLOCATABLE :: eig(:,:,:)
      REAL(dbl),    ALLOCATABLE :: occ(:,:,:)
      COMPLEX(dbl), ALLOCATABLE :: U_rot(:,:,:,:)
      COMPLEX(dbl), ALLOCATABLE :: wfc(:,:)
      COMPLEX(dbl), ALLOCATABLE :: wfc_new(:,:)
      !
      INTEGER       :: ierr

      !   
      ! end of declarations
      !   

!
!------------------------------
! main body
!------------------------------
!

      CALL timing(subname,OPR='start')
      !CALL log_push(subname)

      !
      ! preliminaries
      !
      lhave_qp_states = ( LEN_TRIM( qp_states ) /= 0 )
      !
      dirname_in   = TRIM(work_dir) // "/" // TRIM( prefix ) // ".save/" 
      dirname_out  = TRIM(work_dir) // "/" // TRIM( prefix ) // "_" // TRIM(label) // ".save/" 
      filename     = TRIM( dirname_in ) // "data-file.xml"
      !
      CALL qexml_init( UNIT_IN=dft_unit,         UNIT_OUT=qexml_unit, &
                       DIR_IN=TRIM(dirname_in),  DIR_OUT=TRIM(dirname_out) )


      !
      ! open prefix.save and get main dimensions
      ! nbnd, nkpts, npwk (check others)
      !
      !
      IF (ionode) WRITE(stdout,"(/,2x,'Reading from dftdata file...')")
      !
      CALL qexml_openfile( filename, "read", IERR=ierr )
      IF ( ierr/=0) CALL errore(subname,'opening dftdata file: '//TRIM(filename),ABS(ierr))
      !
      IF ( ionode ) THEN
          !
          CALL qexml_read_bands_info( NBND=nbnd, NUM_K_POINTS=nkpts, &
                                      NSPIN=nspin, IERR=ierr )
          !
      ENDIF
      !
      CALL mp_bcast( nbnd,    ionode_id )
      CALL mp_bcast( nkpts,   ionode_id )
      CALL mp_bcast( nspin,   ionode_id )
      CALL mp_bcast( ierr,    ionode_id )
      !
      IF ( ierr/=0 ) CALL errore(subname,"reading bands_info",ABS(ierr))
      !
      !
      IF ( nspin /= 1 ) CALL errore(subname,"nspin/=1 not implemented",10)
      !
      ALLOCATE( npwk(nkpts), STAT=ierr )
      IF ( ierr/=0 ) CALL errore(subname,"allocating npwk",ABS(ierr))
      !
      IF ( ionode ) THEN
          !
          DO ik = 1, nkpts
              CALL qexml_read_gk(ik, NPWK=npwk(ik), GAMMA_ONLY=lgamma_only, IERR=ierr)
              IF ( ierr/=0 ) CALL errore(subname,'reading gk',ik)
          ENDDO
          !
      ENDIF
      !
      CALL mp_bcast( npwk,        ionode_id)
      CALL mp_bcast( lgamma_only, ionode_id)
      !
      npwkx = MAXVAL( npwk(1:nkpts) )
      !
      IF ( lgamma_only ) CALL errore(subname,"gamma_only not implemented",10)
      !
      ! occupations
      !
      ALLOCATE( occ(nbnd,nkpts,nspin), STAT=ierr)
      IF ( ierr/=0 ) CALL errore(subname,"alloating occ",ABS(ierr))
      !
      IF ( ionode ) THEN
          !
          DO isp = 1, nspin
          DO ik  = 1, nkpts
              !
              IF ( nspin == 1 ) THEN 
                  CALL qexml_read_bands(IK=ik, OCC=occ(:,ik,isp), IERR=ierr)
              ELSE
                  CALL qexml_read_bands(IK=ik, ISPIN=isp, OCC=occ(:,ik,isp), IERR=ierr)
              ENDIF
              !
          ENDDO
          ENDDO
          !
      ENDIF
      !
      CALL mp_bcast( occ,         ionode_id)

      !
      ! G-vectors
      !
      IF ( lhave_qp_states ) THEN
          !
          CALL qexml_read_planewaves( NR1=nr1, NR2=nr2, NR3=nr3,  NGM=ngm, IERR=ierr )
          IF ( ierr/=0 ) CALL errore(subname,"reading planewaves I",ABS(ierr))
          !
          ALLOCATE( igv(3,ngm), STAT=ierr )
          IF ( ierr/=0 ) CALL errore(subname,'allocating igv',ABS(ierr))
          !
          CALL qexml_read_planewaves( IGV=igv, IERR=ierr )
          IF ( ierr/=0 ) CALL errore(subname,"reading planewaves II",ABS(ierr))
          !
      ENDIF
      !
      !
      CALL qexml_closefile( "read", IERR=ierr)
      IF ( ierr/=0 ) CALL errore(subname,"closing dft datafile", ABS(ierr))
  

      !
      ! open qp_file, read eigenvalues & eigenvector matrix
      ! dump eigenval.xml files
      !
      IF (ionode) WRITE(stdout,"(2x,'Reading from qp_file...')")
      !
      CALL iotk_open_read( sax_unit, FILE=TRIM(qp_file), IERR=ierr )
      IF ( ierr/=0 ) CALL errore(subname,"opening qp_file: "//TRIM(qp_file),ABS(ierr))
      !
      str=TRIM(label)//"_QP"
      !
      CALL iotk_scan_begin(sax_unit, TRIM(str), IERR=ierr)
      IF ( ierr/=0 ) CALL errore(subname,"scan begin "//TRIM(str),ABS(ierr))
      !
      CALL iotk_scan_dat(sax_unit, "nbmin", ibnd_min, IERR=ierr)
      IF ( ierr/=0 ) CALL errore(subname,"scan nbmin",ABS(ierr))
      CALL iotk_scan_dat(sax_unit, "nbmax", ibnd_max, IERR=ierr)
      IF ( ierr/=0 ) CALL errore(subname,"scan nbmax",ABS(ierr))
      !
      ! checks
      !
      IF ( ibnd_min /= 1 )   CALL errore(subname,"ibnd_min /= 1", 10)
      IF ( ibnd_max > nbnd ) CALL errore(subname,"ibnd_max too large", 10)
      !
      ALLOCATE( eig(nbnd,nkpts,nspin), STAT=ierr )
      IF ( ierr/=0 ) CALL errore(subname,"allocating eig",ABS(ierr))
      !
      DO isp = 1, nspin
      DO ik  = 1, nkpts
          !
          eig(1:ibnd_min-1,ik,isp)    = ZERO
          eig(ibnd_max+1:nbnd,ik,isp) = ZERO
          !
          WRITE(str,"('energies',i3.3)") ik + (isp-1)*nspin
          CALL iotk_scan_dat( sax_unit, TRIM(str), eig(ibnd_min:ibnd_max,ik,isp), IERR=ierr )
          IF ( ierr/=0 ) CALL errore(subname,"searching for "//TRIM(str), ABS(ierr))
          !
      ENDDO
      ENDDO
      ! 
      ALLOCATE( U_rot(ibnd_min:ibnd_max,ibnd_min:ibnd_max,nkpts,nspin), STAT=ierr )
      IF ( ierr/=0 ) CALL errore(subname,"allocating U_rot",ABS(ierr))
      ! 
      DO isp = 1, nspin
      DO ik  = 1, nkpts
          !
          WRITE(str,"('eigenvec',i3.3)") ik + (isp-1)*nspin
          CALL iotk_scan_dat( sax_unit, TRIM(str), U_rot(:,:,ik,isp), IERR=ierr )
          IF ( ierr/=0 ) CALL errore(subname,"searching for "//TRIM(str), ABS(ierr))
          !
      ENDDO
      ENDDO
      ! 
      str=TRIM(label)//"_QP"
      !
      CALL iotk_scan_end(sax_unit, TRIM(str), IERR=ierr)
      IF ( ierr/=0 ) CALL errore(subname,"scan end "//TRIM(str),ABS(ierr))
      !
      CALL iotk_close_read( sax_unit, IERR=ierr )
      IF ( ierr/=0 ) CALL errore(subname,"closing qp_file: "//TRIM(qp_file),ABS(ierr))

      !
      ! dump eigenval.xml files
      !
      ! SaX units = Ryd, converting to Hartrees
      !
      eig(:,:,:) = eig(:,:,:) / TWO
      !
      IF ( ionode ) WRITE(stdout,"(2x,'Write eigenval to file...',/)")
      !
      IF ( ionode ) THEN
          !
          DO isp = 1, nspin
          DO ik  = 1, nkpts
              !
              IF ( nspin == 1 ) THEN
                  CALL qexml_write_bands( ik, NBND=nbnd,  EIG=eig(:,ik,isp), &
                                          ENERGY_UNITS="Hartree", OCC=occ(:,ik,isp) )
              ELSE
                  CALL qexml_write_bands( ik, ISPIN=isp, NBND=nbnd, EIG=eig(:,ik,isp), &
                                          ENERGY_UNITS="Hartree", OCC=occ(:,ik,isp) )
              ENDIF
              !
          ENDDO
          ENDDO
          !
      ENDIF
      ! 
      DEALLOCATE( eig, occ, STAT=ierr)
      IF ( ierr/=0 ) CALL errore(subname,"deallocating eig, occ",ABS(ierr))
      

      !
      ! if have qp_states, read wfcs and dump them
      ! otherwise, read wfcs from $prefix.save
      ! and rotate them according to the eigenvector matrices
      !
      IF ( lhave_qp_states ) THEN
          !
          ALLOCATE( igk(3,npwkx), STAT=ierr )
          IF( ierr/=0 ) CALL errore(subname,"allocating igk",ABS(ierr))
          ALLOCATE( igksort(npwkx), STAT=ierr )
          IF( ierr/=0 ) CALL errore(subname,"allocating igksort",ABS(ierr))
          !
          ALLOCATE( fft2gv(0:nr1*nr2*nr3), STAT=ierr )
          IF( ierr/=0 ) CALL errore(subname,"allocating fft2gv",ABS(ierr))
          !
          CALL ggrids_gv_indexes( igv, ngm, nr1, nr2, nr3, FFT2GV=fft2gv )
          !
      ELSE
          !
          ALLOCATE( wfc(npwkx,nbnd), STAT=ierr )
          IF( ierr/=0 ) CALL errore(subname,"allocating wfc",ABS(ierr))
          !
      ENDIF
      !
      ALLOCATE( wfc_new(npwkx,nbnd), STAT=ierr )
      IF( ierr/=0 ) CALL errore(subname,"allocating wfc_new",ABS(ierr))

      !
      ! init read
      !
      IF ( lhave_qp_states ) THEN
          !
          CALL iotk_open_read( sax_unit, FILE=TRIM(qp_states), IERR=ierr )
          IF ( ierr/=0 ) CALL errore(subname,"opening qp_states: "//TRIM(qp_states),ABS(ierr))
          !
          str=TRIM(label)//"_states"
          !
          CALL iotk_scan_begin(sax_unit, TRIM(str), IERR=ierr)
          IF ( ierr/=0 ) CALL errore(subname,"scan begin "//TRIM(str),ABS(ierr))
          !
      ENDIF
      
      !
      ! read and/or convert wfc data
      !
      DO isp = 1, nspin
      DO ik  = 1, nkpts
          !
          IF ( ionode ) WRITE(stdout,"(2x,'Read/Write wfc [',i3,' ] to file...')") ik
          !
          have_qp_states_if :&
          IF ( lhave_qp_states ) THEN
              !
              CALL iotk_scan_begin(sax_unit, "basis"//TRIM(iotk_index(ik)), IERR=ierr)
              IF ( ierr/=0 ) CALL errore(subname,"scan begin basis"//TRIM(iotk_index(ik)),ABS(ierr))
              ! 
              CALL iotk_scan_empty(sax_unit, "info", ATTR=attr, IERR=ierr)
              IF ( ierr/=0 ) CALL errore(subname,"scan info",ABS(ierr))
              CALL iotk_scan_attr(attr, "npw", npw, IERR=ierr)
              IF ( ierr/=0 ) CALL errore(subname,"scan npw",ABS(ierr))
              CALL iotk_scan_attr(attr, "k", xk, IERR=ierr)
              IF ( ierr/=0 ) CALL errore(subname,"scan k",ABS(ierr))
              !
              IF ( npw /= npwk(ik) ) CALL errore(subname,"unexpected npw",10)
              !
              CALL iotk_scan_dat(sax_unit, "g", igk(1:3,1:npwk(ik)), IERR=ierr)
              IF ( ierr/=0 ) CALL errore(subname,"scan g",ABS(ierr))
              !
              CALL iotk_scan_end(sax_unit, "basis"//TRIM(iotk_index(ik)), IERR=ierr)
              IF ( ierr/=0 ) CALL errore(subname,"scan end basis"//TRIM(iotk_index(ik)),ABS(ierr))
              !
              wfc_new(:,:)    = CZERO
              !
              DO ib = ibnd_min, ibnd_max
                  !
                  str="wfc"//TRIM(iotk_index(ik))//TRIM(iotk_index(ib))
                  !
                  CALL iotk_scan_begin(sax_unit, TRIM(str), IERR=ierr)
                  IF ( ierr/=0 ) CALL errore(subname,"scan begin "//TRIM(str),ABS(ierr))
                  !
                  CALL iotk_scan_dat(sax_unit, "val", wfc_new(1:npwk(ik),ib), IERR=ierr)
                  IF ( ierr/=0 ) CALL errore(subname,"val",ABS(ierr))
                  !
                  wfc_new(npwk(ik)+1:npwkx,ib) = CZERO
                  !
                  CALL iotk_scan_end(sax_unit, TRIM(str), IERR=ierr)
                  IF ( ierr/=0 ) CALL errore(subname,"scan end "//TRIM(str),ABS(ierr))
                  !
              ENDDO

              !
              ! find the map for SaX G-vectors in the QE density G grid
              !
              igksort(1:npwk(ik)) = 0
              !
              CALL ggrids_map_igv( npwk(ik), igk, nr1, nr2, nr3, &
                                   fft2gv, igksort )
              !
          ELSE
              !
              IF ( nspin == 1 ) THEN
                   !
                   CALL qexml_read_wfc( 1, nbnd, ik, &
                                        WF=wfc(:,:), IERR=ierr )
                   !
              ELSE
                   !
                   CALL qexml_read_wfc( 1, nbnd, ik, ISPIN=isp, &
                                        WF=wfc(:,:), IERR=ierr )
                   !
              ENDIF
              !
              IF ( ierr/=0 ) CALL errore(subname,"reading old wfcs",ABS(ierr))
              !
              CALL mat_mul( wfc_new(:,ibnd_min:ibnd_max), &
                            wfc(:,ibnd_min:ibnd_max), 'N', U_rot(:,:,ik,isp), 'N', &
                            npwk(ik), ibnd_max-ibnd_min+1, ibnd_max-ibnd_min+1 )
              !
          ENDIF have_qp_states_if

          !
          ! dump evc.dat, gkvectors.dat
          !
          IF ( ionode ) THEN
              !
              ! evc.dat
              !
              IF ( nspin == 1 ) THEN
                  !
                  CALL qexml_write_wfc( nbnd, nkpts, nspin, ik, &
                              NGW=0, IGWX=npwk(ik), GAMMA_ONLY=lgamma_only, &
                              WF=wfc_new(1:npwk(ik),1:nbnd) )
                  !
              ELSE
                  !
                  CALL qexml_write_wfc( nbnd, nkpts, nspin, ik, ISPIN=isp, &
                              NGW=0, IGWX=npwk(ik), GAMMA_ONLY=lgamma_only, &
                              WF=wfc_new(1:npwk(ik),1:nbnd) )
                  !
              ENDIF
              !
              ! gkvectors.dat
              !
              IF ( lhave_qp_states ) THEN
                  !
                  CALL qexml_write_gk( ik, npwk(ik), npwkx, lgamma_only, &
                                       xk, "crystal", INDEX=igksort, IGK=igk)
                  !
              ENDIF
              !
          ENDIF
          !
      ENDDO
      ENDDO
      !
      !
      IF ( lhave_qp_states ) THEN
          !
          str=TRIM(label)//"_states"
          !
          CALL iotk_scan_end(sax_unit, TRIM(str), IERR=ierr)
          IF ( ierr/=0 ) CALL errore(subname,"scan end "//TRIM(str),ABS(ierr))
          !
          CALL iotk_close_read( sax_unit, IERR=ierr )
          IF ( ierr/=0 ) CALL errore(subname,"closing qp_states: "//TRIM(qp_states),ABS(ierr))
          !
      ENDIF


      !
      ! cleanup
      !
      DEALLOCATE( npwk, STAT=ierr )
      IF ( ierr/=0 ) CALL errore(subname, 'deallocating npwk', ABS(ierr))
      !
      IF ( ALLOCATED( wfc ) ) THEN
          DEALLOCATE( wfc, STAT=ierr )
          IF ( ierr/=0 ) CALL errore(subname, 'deallocating wfc', ABS(ierr))
      ENDIF
      IF ( ALLOCATED( igk ) ) THEN
          DEALLOCATE( igk, STAT=ierr )
          IF ( ierr/=0 ) CALL errore(subname, 'deallocating igk', ABS(ierr))
      ENDIF
      IF ( ALLOCATED( igv ) ) THEN
          DEALLOCATE( igv, STAT=ierr )
          IF ( ierr/=0 ) CALL errore(subname, 'deallocating igv', ABS(ierr))
      ENDIF
      IF ( ALLOCATED( igksort ) ) THEN
          DEALLOCATE( igksort, STAT=ierr )
          IF ( ierr/=0 ) CALL errore(subname, 'deallocating igksort', ABS(ierr))
      ENDIF
      IF ( ALLOCATED( fft2gv ) ) THEN
          DEALLOCATE( fft2gv, STAT=ierr )
          IF ( ierr/=0 ) CALL errore(subname, 'deallocating fft2gv', ABS(ierr))
      ENDIF
      !
      DEALLOCATE( wfc_new, STAT=ierr )
      IF ( ierr/=0 ) CALL errore(subname, 'deallocating wfc_new', ABS(ierr))
      !
      IF ( ALLOCATED(U_rot) ) THEN
          DEALLOCATE( U_rot, STAT=ierr )
          IF ( ierr/=0 ) CALL errore(subname,"deallocating U_rot", ABS(ierr))
      ENDIF
      !   
      !
      CALL timing(subname,OPR='stop')
      !CALL log_pop(subname)
      !
      RETURN
      !
   END SUBROUTINE do_sax2qexml

END PROGRAM sax2qexml


!********************************************************
SUBROUTINE ggrids_map_igv( npwk, igk, nr1, nr2, nr3, fft2gv, igksort )
   !********************************************************
   !
   USE kinds
   USE ggrids_module,      ONLY : ggrids_gv_indexes
   !
   IMPLICIT NONE
   !
   INTEGER,    INTENT(IN)  :: npwk
   INTEGER,    INTENT(IN)  :: igk(3,npwk)
   INTEGER,    INTENT(IN)  :: nr1, nr2, nr3
   INTEGER,    INTENT(IN)  :: fft2gv(0:nr1*nr2*nr3)
   INTEGER,    INTENT(OUT) :: igksort(npwk)
   !
   CHARACTER(14) :: subname='ggrids_map_igv'
   INTEGER       :: ig, ierr
   INTEGER, ALLOCATABLE :: gvk2fft(:)
   !
   !
   ALLOCATE( gvk2fft( npwk ), STAT=ierr  )
   IF ( ierr/=0 ) CALL errore(subname,'allocating gvk2fft',ABS(ierr))
   !
   CALL ggrids_gv_indexes( igk, npwk, nr1, nr2, nr3, GV2FFT=gvk2fft )
   !
   DO ig = 1, npwk
       !
       igksort( ig ) = fft2gv( gvk2fft(ig) )
       !
   ENDDO
   !
   DEALLOCATE( gvk2fft, STAT=ierr  )
   IF ( ierr/=0 ) CALL errore(subname,'deallocating gvk2fft',ABS(ierr))
   !
   RETURN
   !
END SUBROUTINE ggrids_map_igv


