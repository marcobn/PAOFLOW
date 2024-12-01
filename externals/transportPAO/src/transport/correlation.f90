!
! Copyright (C) 2005 WanT Group, 2017 ERMES Group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License\'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!*********************************************
   MODULE T_correlation_module
!*********************************************
   !
   USE kinds,                   ONLY : dbl
   USE constants,               ONLY : EPS_m6
   USE parameters,              ONLY : nstrx
   USE io_module,               ONLY : ionode, ionode_id, stdout
   USE mp,                      ONLY : mp_bcast
   USE parser_module,           ONLY : change_case
   USE operator_module,         ONLY : operator_read_aux, operator_read_data
   USE timing_module,           ONLY : timing
   USE log_module,              ONLY : log_push, log_pop
   USE files_module,            ONLY : file_open, file_close
   USE T_egrid_module,          ONLY : de, ne, egrid, emin, emax, ne_buffer, egrid_alloc => alloc
   USE T_kpoints_module,        ONLY : nkpts_par, nrtot_par, ivr_par
   USE T_hamiltonian_module,    ONLY : dimL, dimC, dimR, &
                                       blc_00L, blc_01L, blc_00R, blc_01R, &
                                       blc_00C, blc_LC,  blc_CR
   USE T_control_module,        ONLY : calculation_type, transport_dir, &
                                       leads_are_identical
   USE T_operator_blc_module
   !
   IMPLICIT NONE
   PRIVATE 
   SAVE

!
! Contains correlation self-energy data
! 
    
    CHARACTER(nstrx) :: datafile_L_sgm
    CHARACTER(nstrx) :: datafile_C_sgm
    CHARACTER(nstrx) :: datafile_R_sgm
    !
    REAL(dbl)   :: shift_C_corr
    !
    LOGICAL     :: lhave_corr   = .FALSE.
    LOGICAL     :: ldynam_corr  = .FALSE.
    !
    LOGICAL     :: first = .TRUE.
    LOGICAL     :: init = .FALSE.


!
! end delcarations
!

   PUBLIC :: dimL, dimC, dimR
   PUBLIC :: nkpts_par
   !
   PUBLIC :: datafile_L_sgm, datafile_C_sgm, datafile_R_sgm 
   !
   PUBLIC :: lhave_corr, ldynam_corr
   PUBLIC :: shift_C_corr
   !
   PUBLIC :: init
   !
   PUBLIC :: correlation_init
   PUBLIC :: correlation_finalize
   PUBLIC :: correlation_read


CONTAINS

!
! subroutines
!
!**********************************************************
   SUBROUTINE correlation_init( )
   !**********************************************************
   !
   ! open the sigma files and allocate the main workspace
   ! energy grid is read from file if present
   !
   IMPLICIT NONE

   !
   ! local vars
   !
   CHARACTER(16)         :: subname="correlation_init"

!
!------------------------------
! main body
!------------------------------
!
   CALL log_push( subname )
   
   IF ( LEN_TRIM( datafile_C_sgm ) /= 0 ) THEN
       !
       CALL correlation_open( blc_00C, datafile_C_sgm )
       CALL correlation_open( blc_LC,  datafile_C_sgm )
       CALL correlation_open( blc_CR,  datafile_C_sgm )
       !
   ENDIF
   ! 
   IF ( LEN_TRIM( datafile_R_sgm ) /= 0 ) THEN
       !
       CALL correlation_open( blc_00R, datafile_L_sgm )
       CALL correlation_open( blc_01R, datafile_L_sgm )
       !
   ENDIF
   ! 
   IF ( LEN_TRIM( datafile_L_sgm ) /= 0 .AND. .NOT. leads_are_identical ) THEN
       !
       CALL correlation_open( blc_00L, datafile_L_sgm )
       CALL correlation_open( blc_01L, datafile_L_sgm )
       !
   ENDIF


   init = .TRUE.

   ! 
   ! read all data if static 
   ! 
   IF ( .NOT. ldynam_corr ) THEN
       !
       CALL correlation_read( )
       !
   ENDIF

   CALL log_pop( subname )
   ! 
END SUBROUTINE correlation_init


!**********************************************************
   SUBROUTINE correlation_finalize( )
   !**********************************************************
   !
   ! close all the sigma files, if the case
   !
   USE T_hamiltonian_module,    ONLY : blc_00L, blc_01L, blc_00R, blc_01R, &
                                       blc_00C, blc_LC,  blc_CR
   IMPLICIT NONE
   !
   CHARACTER(20)  :: subname='correlation_finalize'
   INTEGER        :: ierr

!
!----------------------------------------
! main Body
!----------------------------------------
!

   CALL timing( subname, OPR='start' )
   CALL log_push( subname )
   

   IF ( blc_00L%iunit_sgm_opened ) THEN
       CALL file_close(blc_00L%iunit_sgm, PATH="/", ACTION="read", IERR=ierr)
       IF ( ierr/=0 ) CALL errore(subname,'closing sgm blc_00L', ABS(ierr) )
   ENDIF
   !
   IF ( blc_01L%iunit_sgm_opened ) THEN
       CALL file_close(blc_01L%iunit_sgm, PATH="/", ACTION="read", IERR=ierr)
       IF ( ierr/=0 ) CALL errore(subname,'closing sgm blc_01L', ABS(ierr) )
   ENDIF
   !
   IF ( blc_00R%iunit_sgm_opened ) THEN
       CALL file_close(blc_00R%iunit_sgm, PATH="/", ACTION="read", IERR=ierr)
       IF ( ierr/=0 ) CALL errore(subname,'closing sgm blc_00R', ABS(ierr) )
   ENDIF
   !
   IF ( blc_01R%iunit_sgm_opened ) THEN
       CALL file_close(blc_01R%iunit_sgm, PATH="/", ACTION="read", IERR=ierr)
       IF ( ierr/=0 ) CALL errore(subname,'closing sgm blc_01R', ABS(ierr) )
   ENDIF
   !
   IF ( blc_00C%iunit_sgm_opened ) THEN
       CALL file_close(blc_00C%iunit_sgm, PATH="/", ACTION="read", IERR=ierr)
       IF ( ierr/=0 ) CALL errore(subname,'closing sgm blc_00C', ABS(ierr) )
   ENDIF
   !
   IF ( blc_CR%iunit_sgm_opened ) THEN
       CALL file_close(blc_CR%iunit_sgm, PATH="/", ACTION="read", IERR=ierr)
       IF ( ierr/=0 ) CALL errore(subname,'closing sgm blc_CR', ABS(ierr) )
   ENDIF
   !
   IF ( blc_LC%iunit_sgm_opened ) THEN
       CALL file_close(blc_LC%iunit_sgm, PATH="/", ACTION="read", IERR=ierr)
       IF ( ierr/=0 ) CALL errore(subname,'closing sgm blc_LC', ABS(ierr) )
   ENDIF

   CALL log_pop( subname )
   CALL timing( subname, OPR='stop' )
   !
END SUBROUTINE correlation_finalize


!**********************************************************
   SUBROUTINE correlation_open( opr, datafile )
   !**********************************************************
   !
   ! open the sigma file and allocate the main workspace
   ! energy grid is read from file if the case
   !
   USE iotk_module,       ONLY : iotk_free_unit
   USE mp_global,         ONLY : nproc
   !
   IMPLICIT NONE
   !
   ! I/O vars
   !
   TYPE(operator_blc),      INTENT(INOUT) :: opr
   CHARACTER(*),            INTENT(IN)    :: datafile

   !
   ! local vars
   !
   CHARACTER(16)     :: subname="correlation_open"
   CHARACTER(nstrx)  :: analyticity
   INTEGER           :: iunit, iunit0
   LOGICAL           :: ldynam, do_open
   INTEGER           :: dimx_corr, nrtot_corr, ne_corr, ierr
   !
   INTEGER,     ALLOCATABLE :: ivr_corr(:,:)
   REAL(dbl),   ALLOCATABLE :: egrid_corr(:)

!
!------------------------------
! main body
!------------------------------
!
   CALL log_push( subname )

   !
   ! get IO data
   !
   IF ( opr%iunit_sgm_opened ) CALL errore(subname,"unit_sgm already connected",10)
   !
   CALL iotk_free_unit( iunit )


   !
   ! This file must be opened by all the processors
   !
   INQUIRE( FILE=TRIM(datafile), NUMBER=iunit0 )
   !
   do_open = .TRUE.
   !
#ifdef __GFORTRAN   
   !
   ! the file is already connected
   IF ( iunit0 > 0 )  THEN
       !
       iunit=iunit0
       do_open = .FALSE.
       !
   ENDIF
   !
#endif
   !
   IF ( do_open ) THEN
       !
       CALL file_open( iunit, TRIM(datafile), PATH="/", ACTION="read", IERR=ierr )
       IF ( ierr/=0 ) CALL errore(subname,'opening '//TRIM(datafile), ABS(ierr) )
       !
       opr%iunit_sgm_opened = .TRUE.
       !
   ELSE
       !
       !REWIND( iunit )
       opr%iunit_sgm_opened = .FALSE.
       !
   ENDIF
   !
   opr%iunit_sgm = iunit


   !
   ! get main data and check them
   !
   !IF ( ionode ) THEN
      !
      CALL operator_read_aux( iunit, DIMWANN=dimx_corr, NR=nrtot_corr, &
                              DYNAMICAL=ldynam, &
                              NOMEGA=ne_corr, ANALYTICITY=analyticity, IERR=ierr )
      !
      IF ( ierr/=0 ) CALL errore(subname,'reading DIMWANN--ANALYTICITY', ABS(ierr))
      !
   !ENDIF
   !!
   !CALL mp_bcast( dimx_corr,    ionode_id )
   !CALL mp_bcast( nrtot_corr,   ionode_id )
   !CALL mp_bcast( ldynam,       ionode_id )
   !CALL mp_bcast( ne_corr,      ionode_id )
   !CALL mp_bcast( analyticity,  ionode_id )
   !
   !
   !IF ( dimx_corr > dimC)              CALL errore(subname,'invalid dimx_corr',3)
   IF ( nrtot_corr <= 0 )               CALL errore(subname,'invalid nrtot_corr',3)
   IF ( ne_corr <= 0 .AND. ldynam_corr) CALL errore(subname,'invalid ne_corr',3)

   !
   ! reset buffering
   !
   IF ( ldynam_corr ) THEN
       !
       ne_buffer = MIN( ne_buffer, INT(  ne_corr / nproc ) + 1 )
       CALL warning( subname, 'buffering reset')
       !
   ELSE
       !
       ne_buffer = 1
       CALL warning( subname, 'buffering reset to 1 because of static sgm')
       !
   ENDIF

   !
   !
   CALL change_case( analyticity, 'lower' )
   IF ( TRIM(analyticity) /= 'retarded' .AND. ldynam_corr) &
             CALL errore(subname,'invalid analyticity = '//TRIM(analyticity),1)
   !
   IF ( first ) THEN
       ldynam_corr = ldynam
       first       = .FALSE.
   ELSE
       IF ( .NOT.  ldynam .EQV. ldynam_corr ) &
          CALL errore(subname,'wrong dynam',10)
   ENDIF

   !
   ALLOCATE ( ivr_corr(3,nrtot_corr), STAT=ierr )
   IF( ierr /=0 ) CALL errore(subname, 'allocating ivr_corr', ABS(ierr) )
   !
   !
   IF ( ldynam_corr ) THEN
       !
       ALLOCATE( egrid_corr(ne_corr), STAT=ierr )
       IF (ierr/=0) CALL errore(subname,'allocating egrid_corr',ABS(ierr))
       !
   ENDIF

   !
   ! read data
   !
!   IF ( ionode ) THEN
       !
       IF ( ldynam_corr ) THEN
           !
           CALL operator_read_aux( iunit, GRID=egrid_corr, IVR=ivr_corr, IERR=ierr )
           IF (ierr/=0) CALL errore(subname,'reading GRID, IVR',ABS(ierr))
           !
       ELSE
           !
           CALL operator_read_aux( iunit, IVR=ivr_corr, IERR=ierr )
           IF (ierr/=0) CALL errore(subname,'reading IVR',ABS(ierr))
           !
       ENDIF
       !
!   ENDIF
!   !
!   IF ( ldynam_corr ) CALL mp_bcast( egrid, ionode_id )
!   CALL mp_bcast( ivr_corr, ionode_id )

   !
   ! setting egrid if the case
   !
   IF ( ldynam_corr ) THEN
       !
       IF ( .NOT. egrid_alloc ) THEN
           !
           ne = ne_corr
           !
           ALLOCATE( egrid(ne), STAT=ierr )
           IF (ierr/=0) CALL errore(subname,'allocating egrid',ABS(ierr))
           !
           egrid = egrid_corr
           !
           CALL warning( subname, "energy egrid is forced from SGM datafile" )
           WRITE( stdout, "()")
           !
           emin = egrid(1)
           emax = egrid(ne)
           de   = ( emax - emin ) / REAL( ne -1, dbl )  
           egrid_alloc = .TRUE.
           !
       ELSE
           !
           IF ( ne /= ne_corr ) CALL errore(subname,'invalid ne_corr /= ne',10) 
           !
       ENDIF
       !
       DEALLOCATE( egrid_corr, STAT=ierr )
       IF (ierr/=0) CALL errore(subname,'deallocating egrid_corr',ABS(ierr))
       !
   ENDIF
   

   !
   ! store data
   !
   CALL operator_blc_allocate( opr%dim1, opr%dim2, nkpts_par, NRTOT_SGM=nrtot_corr, &
                               NE_SGM=ne_buffer, LHAVE_CORR=.TRUE., OBJ=opr)
   !
   opr%ivr_sgm    = ivr_corr
   opr%dimx_sgm   = dimx_corr
   !
   DEALLOCATE ( ivr_corr, STAT=ierr )
   IF( ierr /=0 ) CALL errore(subname, 'allocating ivr_corr', ABS(ierr) )


   CALL log_pop( 'correlation_open' )
   ! 
END SUBROUTINE correlation_open


!*******************************************************************
   SUBROUTINE correlation_read( ie_s, ie_e )
   !*******************************************************************
   !
   ! Read correlation data for all the blocks.
   ! If IE_S, IE_E are present, it means that we are reading 
   ! dynamic self-energies
   !
   IMPLICIT NONE

   !
   ! Input variabls
   !
   INTEGER, OPTIONAL, INTENT(IN) :: ie_s, ie_e

   !
   ! local variables
   !
   CHARACTER(16) :: subname="correlation_read"
   INTEGER       :: ie, ie_buff

   !
   ! end of declarations
   !

!
!----------------------------------------
! main Body
!----------------------------------------
!

   CALL timing( subname, OPR='start')
   CALL log_push( subname )

   !
   ! few checks
   !
   IF ( .NOT. init )       CALL errore(subname,'correlation not init',10)
   IF ( .NOT. lhave_corr ) CALL errore(subname,'correlation not required',10)
   !
   IF ( .NOT. ( PRESENT( ie_s ) .EQV.  PRESENT( ie_e ) )  ) &
       CALL errore(subname,' ie_s and ie_e should be given together or not given at all',10)
   !
   IF ( ( PRESENT( ie_s ) .OR. PRESENT( ie_e ) ) .AND. .NOT. ldynam_corr ) &
       CALL errore(subname,'correlation is not dynamic',10)

   IF ( ldynam_corr .AND. .NOT. PRESENT( ie_s ) ) &
       CALL errore(subname,'ie_s should be present',10)



   !
   ! chose whether to do 'conductor' or 'bulk'
   !
   SELECT CASE ( TRIM(calculation_type) )
   !
   CASE ( "conductor" )
       !
       !
       IF ( ldynam_corr ) THEN
           !
           ie_buff = 0
           !
           DO ie = ie_s, ie_e
               !
               ie_buff = ie_buff + 1
               ! 
               CALL correlation_sgmread( blc_00C, IE=ie, IE_BUFF=ie_buff )
               CALL correlation_sgmread( blc_CR,  IE=ie, IE_BUFF=ie_buff )
               CALL correlation_sgmread( blc_LC,  IE=ie, IE_BUFF=ie_buff )
               !
               CALL correlation_sgmread( blc_00R, IE=ie, IE_BUFF=ie_buff )
               CALL correlation_sgmread( blc_01R, IE=ie, IE_BUFF=ie_buff )
               !
               IF ( .NOT. leads_are_identical ) THEN
                   CALL correlation_sgmread( blc_00L, IE=ie, IE_BUFF=ie_buff )
                   CALL correlation_sgmread( blc_01L, IE=ie, IE_BUFF=ie_buff )
               ENDIF
               !
           ENDDO
           !
       ELSE
           !
           CALL correlation_sgmread( blc_00C )
           CALL correlation_sgmread( blc_CR  )
           CALL correlation_sgmread( blc_LC  )
           !
           CALL correlation_sgmread( blc_00R )
           CALL correlation_sgmread( blc_01R )
           !
           IF ( .NOT. leads_are_identical ) THEN
               CALL correlation_sgmread( blc_00L )
               CALL correlation_sgmread( blc_01L )
           ENDIF
           !
       ENDIF
       !
       !
   CASE ( "bulk" )
       !
       !
       IF ( ldynam_corr ) THEN
           !
           ie_buff = 0
           !
           DO ie = ie_s, ie_e
               !
               ie_buff = ie_buff + 1
               !
               CALL correlation_sgmread( blc_00C, IE=ie, IE_BUFF=ie_buff )
               CALL correlation_sgmread( blc_CR,  IE=ie, IE_BUFF=ie_buff )
               !
           ENDDO
           !
       ELSE
           !
           CALL correlation_sgmread( blc_00C )
           CALL correlation_sgmread( blc_CR  )
           !
       ENDIF
       !
       ! rearrange the data already read
       !
       blc_00R = blc_00C
       blc_01R = blc_CR
       blc_LC  = blc_CR
       !
       blc_00R%iunit_sgm = -2
       blc_01R%iunit_sgm = -2
       blc_LC%iunit_sgm = -2
       !
       blc_00R%iunit_sgm_opened = .FALSE.
       blc_01R%iunit_sgm_opened = .FALSE.
       blc_LC%iunit_sgm_opened = .FALSE.
       !
       IF ( .NOT. leads_are_identical ) THEN ! this is never the case
           blc_00L = blc_00C
           blc_01L = blc_CR
       ENDIF
       !
   CASE DEFAULT
       !
       CALL errore(subname,'Invalid calculation_type = '// TRIM(calculation_type),5)
       !
   END SELECT

   
   CALL timing( subname, OPR='STOP' )
   CALL log_pop( subname )
   !
   RETURN
   !
END SUBROUTINE correlation_read


!**********************************************************
   SUBROUTINE correlation_sgmread( opr, ie, ie_buff )
   !**********************************************************
   !
   IMPLICIT NONE
   !
   TYPE(operator_blc), INTENT(INOUT) :: opr
   INTEGER, OPTIONAL,  INTENT(IN)    :: ie, ie_buff
   !
   CHARACTER(19)              :: subname="correlation_sgmread"
   COMPLEX(dbl), ALLOCATABLE  :: caux(:,:,:), caux_small(:,:,:)
   LOGICAL                    :: lfound
   INTEGER                    :: iun
   INTEGER                    :: ind, ivr_aux(3)
   INTEGER                    :: i, j, ie_bl, ir, ir_par, ierr


   CALL timing( subname, OPR='start' )
   CALL log_push( subname )

   IF ( .NOT. init )             CALL errore(subname,'correlation not init',71)
   IF ( .NOT. opr%alloc )        CALL errore(subname,'opr not alloc',71)
   IF ( opr%nkpts /= nkpts_par ) CALL errore(subname,'invalid nkpts',3)

   IF ( PRESENT(ie) .AND. .NOT. PRESENT(ie_buff) ) &
       CALL errore(subname,'ie and ie_buff should be present together',3)

   !
   ! setting index for buffering
   !
   ie_bl = 1
   IF ( PRESENT(ie_buff) ) ie_bl = ie_buff
  

   !
   ! if we do not have any self-energy for this block, 
   ! return
   !
   IF ( .NOT. ASSOCIATED( opr%sgm) ) THEN
       !
       CALL timing( subname, OPR='stop' )
       CALL log_pop( subname )
       RETURN
       !
   ENDIF
   !
   iun = opr%iunit_sgm


   !
   ! allocate auxiliary quantities
   !
   ALLOCATE( caux( opr%dimx_sgm, opr%dimx_sgm, opr%nrtot_sgm), STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'allocating caux', ABS(ierr))
   !
   ALLOCATE( caux_small(opr%dim1, opr%dim2, nrtot_par), STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'allocating caux_small', ABS(ierr))

   !
   ! get the required data
   !
   IF ( PRESENT( ie ) ) THEN
       !
       opr%ie      = ie
       CALL operator_read_data( iun, ie, R_OPR=caux, IERR=ierr )
       !
   ELSE
       !
       CALL operator_read_data( iun, R_OPR=caux, IERR=ierr )
       !
   ENDIF
   !
   IF ( ierr/=0 ) CALL errore(subname, 'reading data from file', ABS(ierr))


   !
   ! get the required matrix elements
   !
   R_loop: &
   DO ir_par = 1, nrtot_par

       !
       ! set the indexes
       !
       j = 0
       DO i=1,3
           !
           ivr_aux(i)=0
           IF ( i == transport_dir ) THEN
               !
               ! set ivr_aux(i) = 0 , 1 depending on the
               ! required matrix (detected from opr%blc_name)
               !
               SELECT CASE( TRIM(opr%blc_name) )
               !
               CASE( "block_00C", "block_00R", "block_00L" )
                   ivr_aux(i) = 0
               CASE( "block_01R", "block_01L", "block_LC", "block_CR" )
                   ivr_aux(i) = 1
               CASE DEFAULT
                   CALL errore(subname, 'invalid label = '//TRIM(opr%blc_name), 1009 )
               END SELECT
               !
           ELSE
               !
               ! set the 2D parallel indexes
               !
               j = j + 1
               ivr_aux(i) = ivr_par( j, ir_par)
               !
           ENDIF
           !
       ENDDO
   
       !
       ! search the 3D index corresponding to ivr_aux
       !
       lfound = .FALSE.
       !
       DO ir = 1, opr%nrtot_sgm
           !
           ind=0
           !
           IF ( ALL( opr%ivr_sgm(:,ir) == ivr_aux(:) ) )  THEN
               !
               lfound = .TRUE.
               ind    = ir
               EXIT
               !
           ENDIF
           !
       ENDDO
       !
       IF ( .NOT. lfound ) CALL errore(subname, '3D R-vector not found', ir_par )


       !
       ! cut the operator (caux) 
       ! according to the required rows and cols
       !
       DO j=1,opr%dim2
       DO i=1,opr%dim1
           !
           caux_small(i, j, ir_par) = caux( opr%irows_sgm(i), opr%icols_sgm(j), ind )
           !
       ENDDO
       ENDDO
       !
       !
   ENDDO R_loop


   !
   ! Compute the 2D fourier transform
   !
   CALL fourier_par (opr%sgm(:,:,:,ie_bl), opr%dim1, opr%dim2, caux_small, opr%dim1, opr%dim2)


   !
   ! local cleaning
   !
   DEALLOCATE( caux, caux_small, STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'deallocating caux, caux_small', ABS(ierr))

   CALL timing( subname, OPR='stop' )
   CALL log_pop( subname )
   !
END SUBROUTINE correlation_sgmread
    

END MODULE T_correlation_module

