!
! Copyright (C) 2004 WanT Group, 2017 ERMES Group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
! Based on a previous version by Marco Buongiorno Nardelli
! See the README file in the main directory for full references
!
!***********************************************
   PROGRAM conductor
   !***********************************************
   !
   ! Computes the quantum transmittance across a junction.
   !
   USE version_module,       ONLY : version_number
   USE io_module,            ONLY : stdout, ionode
   USE timing_module,        ONLY : timing
   USE log_module,           ONLY : log_push, log_pop
   USE T_input_module,       ONLY : input_manager
   USE T_control_module,     ONLY : datafile_C, transport_dir
   USE T_smearing_module,    ONLY : smearing_init
   USE T_egrid_module,       ONLY : egrid_init, egrid_init_ph, egrid_alloc => alloc
   USE T_kpoints_module,     ONLY : kpoints_init
   USE datafiles_module,     ONLY : t_datafiles_init
   USE T_correlation_module, ONLY : lhave_corr, ldynam_corr, correlation_init, &
                                    correlation_read
   USE T_input_parameters_module, ONLY : carriers
   !
   IMPLICIT NONE

   !
   ! local variables
   !
   CHARACTER(9)    :: subname="conductor"


!
!------------------------------
! main body
!------------------------------
!
   CALL startup(version_number,subname)

   !
   ! read input file
   !
   CALL input_manager()
      

   !
   ! Init main data
   !
   CALL write_header( stdout, "Conductor Init" )
   !
   CALL t_datafiles_init()
   !
   CALL smearing_init()
   !
   CALL kpoints_init( datafile_C, transport_dir )
   !
   CALL hamiltonian_init( )


   !
   ! Init correlation data and energy grid
   !
   ! If correlation is used, the energy grid is read
   ! from datafile_sgm and input parameters are overwritten
   !
   ! otherwise, grid is built indipendently
   !
   IF ( lhave_corr ) THEN 
       !
       CALL correlation_init( )
       !
   ENDIF   
   !
   IF ( .NOT. egrid_alloc) THEN
        IF ( TRIM(carriers) == 'electrons') CALL egrid_init( )
        IF ( TRIM(carriers) == 'phonons') CALL egrid_init_ph( )
   ENDIF

   !
   ! print data to output
   !
   CALL summary( stdout )

   !
   ! do the main task
   !
   CALL do_conductor() 

   !
   ! clean global memory
   !
   CALL cleanup()

   !
   ! finalize
   !
   CALL shutdown ( subname ) 

END PROGRAM conductor
  

!********************************************************
   SUBROUTINE do_conductor()
   !********************************************************
   !
   ! perform the main task of the calculation
   !
   USE kinds,                ONLY : dbl
   USE constants,            ONLY : PI, ZERO, CI
   USE parameters,           ONLY : nstrx 
   USE files_module,         ONLY : file_open, file_close
   USE timing_module,        ONLY : timing, timing_upto_now
   USE log_module,           ONLY : log_push, log_pop
   USE util_module,          ONLY : mat_mul, mat_inv
   USE mp_global,            ONLY : mpime, nproc
   USE mp,                   ONLY : mp_sum
   USE io_module,            ONLY : io_name, ionode, stdout, stdin, &
                                    sgmL_unit => aux3_unit, sgmR_unit => aux4_unit, &
                                    gf_unit => aux5_unit, work_dir, prefix, postfix, aux_unit
   USE operator_module,      ONLY : operator_write_init, operator_write_close, &
                                    operator_write_aux, operator_write_data
   USE T_control_module,     ONLY : conduct_formula, nprint, transport_dir, &
                                    write_kdata, write_lead_sgm, write_gf, &
                                    do_eigenchannels, neigchn, neigchnx, &
                                    do_eigplot, ie_eigplot, ik_eigplot, leads_are_identical
   USE T_egrid_module,       ONLY : ne, egrid, &
                                    ne_buffer, egrid_buffer_doread, egrid_buffer_iend
   USE T_kpoints_module,     ONLY : nkpts_par, vkpt_par3D, wk_par, ivr_par3D, &
                                    vr_par3D, nrtot_par
   USE T_hamiltonian_module, ONLY : dimL, dimR, dimC, dimx, dimx_lead,  &
                                    blc_00L, blc_01L, blc_00R, blc_01R, &
                                    blc_00C, blc_LC,  blc_CR
   USE T_workspace_module,   ONLY : tsum, tsumt, work, &
                                    gL, gR, gC, gamma_R, gamma_L, sgm_L, sgm_R, &
                                    rsgm_L, rsgm_R, kgC, rgC, workspace_allocate
   USE T_correlation_module, ONLY : lhave_corr, ldynam_corr, &
                                    correlation_read, correlation_finalize
   USE T_write_data_module,  ONLY : wd_write_data, wd_write_eigchn
   USE T_operator_blc_module
   USE T_input_parameters_module, ONLY : carriers, surface
   !
   IMPLICIT NONE

   !
   ! local variables
   !
   CHARACTER(12)    :: subname="do_conductor"
   !
   INTEGER          :: i, ir, ik, ik_eff, ierr, niter
   INTEGER          :: ie_g
   INTEGER          :: iomg_s, iomg_e
   INTEGER          :: ie_buff, ie_buff_s, ie_buff_e
   LOGICAL          :: read_buffer
   LOGICAL          :: write_eigchn
   REAL(dbl)        :: avg_iter
   CHARACTER(4)     :: ctmp
   CHARACTER(nstrx) :: filename
   !   
   REAL(dbl),    ALLOCATABLE :: conduct_k(:,:,:), conduct(:,:)
   REAL(dbl),    ALLOCATABLE :: dos_k(:,:), dos(:), cond_aux(:)
   COMPLEX(dbl), ALLOCATABLE :: z_eigplot(:,:)
   !
   ! end of declarations
   !

    REAL(dbl) :: rydcm1 = 13.6058d0*8065.5d0
    REAL(dbl) :: amconv = 1.66042d-24/9.1095d-28*0.5d0

!
!------------------------------
! main body 
!------------------------------
!

   CALL timing(subname,OPR='start')
   CALL log_push(subname)


   !
   ! local variable allocations
   !
   CALL workspace_allocate()

   !
   ! memory usage
   !
   IF ( ionode ) WRITE( stdout, "()" )
   CALL memusage( stdout )
   !
   CALL flush_unit( stdout )


   ALLOCATE ( dos_k(ne,nkpts_par), STAT=ierr )
   IF( ierr /=0 ) CALL errore(subname,'allocating dos_k', ABS(ierr) )
   ALLOCATE ( dos(ne), STAT=ierr )
   IF( ierr /=0 ) CALL errore(subname,'allocating dos', ABS(ierr) )
   !
   !
   IF ( do_eigenchannels ) THEN
       !
       neigchn = MIN( dimC,dimR,dimL,  neigchnx )
       !
   ELSE
       !
       neigchn = 0
       !
   ENDIF
   !
   ALLOCATE ( conduct(1+neigchn,ne), STAT=ierr )
   IF( ierr /=0 ) CALL errore(subname,'allocating conduct', ABS(ierr) )
   !
   ALLOCATE ( conduct_k(1+neigchn, nkpts_par,ne), STAT=ierr )
   IF( ierr /=0 ) CALL errore(subname,'allocating conduct_k', ABS(ierr) )
   !
   !
   IF ( do_eigenchannels .AND. do_eigplot ) THEN
       !
       ALLOCATE ( z_eigplot(dimC, dimC ), STAT=ierr )
       IF( ierr /=0 ) CALL errore(subname,'allocating z_eigplot', ABS(ierr) )
       !
   ELSE
       ALLOCATE ( z_eigplot( 1, 1) )
   ENDIF
   !
   !
   ALLOCATE ( cond_aux(dimC), STAT=ierr )
   IF( ierr /=0 ) CALL errore(subname,'allocating cond_aux', ABS(ierr) )


!
!================================
! main loop over frequency
!================================
! 

   IF ( ionode ) THEN
       !
       CALL write_header( stdout, "Frequency Loop" )
       CALL flush_unit( stdout )
       !
   ENDIF

   !
   ! init parallelism over frequencies
   !
   CALL divide_et_impera( 1, ne,  iomg_s, iomg_e, mpime, nproc )

   
   !
   ! init output files for lead sgm
   !
   IF ( write_lead_sgm ) THEN
       !
       CALL io_name( "sgm", filename, BODY="sgmlead_L" )
       CALL operator_write_init(sgmL_unit, filename)
       CALL operator_write_aux( sgmL_unit, dimC, .TRUE., ne, iomg_s, iomg_e, &
                                NRTOT=nrtot_par, IVR=ivr_par3D, GRID=egrid, &
                                ANALYTICITY="retarded", EUNITS="eV" )
       !
       CALL io_name( "sgm", filename, BODY="sgmlead_R" )
       CALL operator_write_init(sgmR_unit, filename)
       CALL operator_write_aux( sgmR_unit, dimC, .TRUE., ne, iomg_s, iomg_e, &
                                NRTOT=nrtot_par, IVR=ivr_par3D, GRID=egrid, &
                                ANALYTICITY="retarded", EUNITS="eV" )
       !
   ENDIF
   !
   ! init output files for the conductor GF
   !
   IF ( write_gf ) THEN
       !
       CALL io_name( "gf", filename, BODY="greenf" )
       CALL operator_write_init(gf_unit, filename)
       CALL operator_write_aux( gf_unit, dimC, .TRUE., ne, iomg_s, iomg_e, &
                                NRTOT=nrtot_par, IVR=ivr_par3D, GRID=egrid, &
                                ANALYTICITY="retarded", EUNITS="eV" )
       !
   ENDIF


   dos(:)           = ZERO
   dos_k(:,:)       = ZERO
   conduct(:,:)     = ZERO
   conduct_k(:,:,:) = ZERO
   !
   ie_buff = 1
   !
   energy_loop: &
   DO ie_g = iomg_s, iomg_e

      
      !
      ! grids and misc
      !
      IF ( (MOD( ie_g, nprint) == 0 .OR. ie_g == iomg_s .OR. ie_g == iomg_e ) &
           .AND. ionode ) THEN
            IF (TRIM(carriers) == 'phonons') THEN
                WRITE(stdout,"(2x, 'Computing omega( ',i5,' ) = ', f12.5, ' cm-1' )") &
                ie_g, dsqrt(egrid(ie_g)*rydcm1**2/amconv)
            ELSE
            WRITE(stdout,"(2x, 'Computing E( ',i5,' ) = ', f12.5, ' eV' )") &
                         ie_g, egrid(ie_g)
            ENDIF
      ENDIF


      !
      ! get correlation self-energy if the case
      !
      IF ( lhave_corr .AND. ldynam_corr ) THEN
          !
          read_buffer = egrid_buffer_doread ( ie_g, iomg_s, iomg_e, ne_buffer )
          !
          IF ( read_buffer ) THEN
              !
              ie_buff_s = ie_g
              ie_buff_e = egrid_buffer_iend( ie_g, iomg_s, iomg_e, ne_buffer ) 
              !
              CALL correlation_read( IE_S=ie_buff_s, IE_E=ie_buff_e )
              !
              ie_buff = 1
              !
          ELSE
              !
              ie_buff = ie_buff +1
              !
          ENDIF
          !
      ENDIF


      !
      ! initialization of the average number of iteration 
      !
      avg_iter = ZERO
      !
      ! parallel kpt loop
      !
      kpt_loop: &
      DO ik = 1, nkpts_par

          !
          ! define aux quantities for each data block
          !
          CALL hamiltonian_setup( ik, ie_g, ie_buff )

          ! 
          !=================================== 
          ! construct leads self-energies 
          !=================================== 
          ! 

          ALLOCATE ( tsum(dimx_lead,dimx_lead), STAT=ierr )
          IF( ierr /=0 ) CALL errore(subname,'allocating tsum', ABS(ierr) )
          ALLOCATE ( tsumt(dimx_lead,dimx_lead), STAT=ierr )
          IF( ierr /=0 ) CALL errore(subname,'allocating tsumt', ABS(ierr) )
          !
          ALLOCATE ( gL(dimL,dimL), STAT=ierr )
          IF( ierr /=0 ) CALL errore(subname,'allocating gL', ABS(ierr) )
          ALLOCATE ( gR(dimR,dimR), STAT=ierr )
          IF( ierr /=0 ) CALL errore(subname,'allocating gR', ABS(ierr) )

          !
          ! right lead
          !
          CALL transfer_mtrx( dimR, blc_00R, blc_01R, dimx_lead, tsum, tsumt, niter )
          avg_iter = avg_iter + REAL(niter)
          !
          CALL green( dimR, blc_00R, blc_01R, dimx_lead, tsum, tsumt, gR, 1 )
          !
 
          ! 
          ! left lead (if needed)
          !
          IF ( .NOT. leads_are_identical ) THEN
              !
              CALL transfer_mtrx( dimL, blc_00L, blc_01L, dimx_lead, tsum, tsumt, niter )
              avg_iter = avg_iter + REAL(niter)
              !
              CALL green( dimL, blc_00L, blc_01L, dimx_lead, tsum, tsumt, gL, -1 )
              !
          ELSE
              !
              CALL green( dimR, blc_00R, blc_01R, dimx_lead, tsum, tsumt, gL, -1 )
              !
          ENDIF
          !
          DEALLOCATE( tsum, tsumt, STAT=ierr )
          IF ( ierr/=0 ) CALL errore(subname,"deallocating tsum, tsumt", ABS(ierr))


          !
          ! lead self-energies
          !
          IF ( .NOT. write_lead_sgm ) THEN
              !
              ALLOCATE( sgm_R(dimC,dimC,1), STAT=ierr )
              IF( ierr /=0 ) CALL errore(subname,'allocating sgm_R', ABS(ierr) )
              ALLOCATE( sgm_L(dimC,dimC,1), STAT=ierr )
              IF( ierr /=0 ) CALL errore(subname,'allocating sgm_L', ABS(ierr) )
              !
              ik_eff = 1
              !
          ELSE
              ik_eff = ik
          ENDIF
          !
          ALLOCATE( work(dimC, MAX(dimL,dimR) ), STAT=ierr ) 
          IF ( ierr/=0 ) CALL errore(subname,"allocating work I", ABS(ierr) )

          !
          ! sgm_R
          !
          CALL mat_mul( work, blc_CR%aux, 'N',          gR, 'N', dimC, dimR, dimR)
          CALL mat_mul( sgm_R(1:dimC,1:dimC,ik_eff),  work, 'N', blc_CR%aux, 'C', dimC, dimC, dimR)
          !
          ! sgm_L
          !
          CALL mat_mul( work, blc_LC%aux, 'C',         gL, 'N', dimC, dimL, dimL)
          CALL mat_mul( sgm_L(1:dimC,1:dimC,ik_eff), work, 'N', blc_LC%aux, 'N', dimC, dimC, dimL) 
 

          DEALLOCATE( work, STAT=ierr)
          IF ( ierr/=0 ) CALL errore(subname,"deallocating work I", ABS(ierr))
          DEALLOCATE( gR, gL, STAT=ierr)
          IF ( ierr/=0 ) CALL errore(subname,"deallocating gR, gL", ABS(ierr))


          !
          !=================================== 
          ! Construct the conductor green's function
          ! gC = work^-1  (retarded)
          !=================================== 
          !
          ALLOCATE( work(dimC, dimC), STAT=ierr ) 
          IF ( ierr/=0 ) CALL errore(subname,"allocating work II", ABS(ierr) )
          ALLOCATE( gC(dimC, dimC), STAT=ierr ) 
          IF ( ierr/=0 ) CALL errore(subname,"allocating gC", ABS(ierr) )
          !
          CALL gzero_maker ( dimC, blc_00C, dimC, work, 'inverse', ' ')
          !
          ! Marcio - surface bandsructure
          if (surface) then
            work(:,:) = work(:,:) -sgm_L(:,:,ik_eff)  ! modification to calculate projected surface bands
          else
            work(:,:) = work(:,:) -sgm_L(:,:,ik_eff) -sgm_R(:,:,ik_eff)
          end if
          !
          CALL mat_inv( dimC, work, gC)
          !
          DEALLOCATE( work, STAT=ierr ) 
          IF ( ierr/=0 ) CALL errore(subname,"deallocating work II", ABS(ierr) )
          !
          !
          IF ( write_gf ) THEN
              !
              kgC(:,:,ik) = gC
              !
          ENDIF

          !
          ! Compute density of states for the conductor layer
          !
          DO i = 1, dimC
             dos_k(ie_g,ik) = dos_k(ie_g,ik) - wk_par(ik) * AIMAG( gC(i,i) ) / PI
          ENDDO
          !
          dos(ie_g) = dos(ie_g) + dos_k(ie_g,ik)


          !
          !=================================== 
          ! Coupling matrices
          !=================================== 
          !
          ! gamma_L & gamma_R
          !
          ALLOCATE( gamma_L(dimC, dimC), STAT=ierr ) 
          IF ( ierr/=0 ) CALL errore(subname,"allocating gamma_L", ABS(ierr) )
          ALLOCATE( gamma_R(dimC, dimC), STAT=ierr ) 
          IF ( ierr/=0 ) CALL errore(subname,"allocating gamma_R", ABS(ierr) )

          gamma_L(:,:) = CI * (  sgm_L(:,:,ik_eff) - CONJG( TRANSPOSE( sgm_L(:,:,ik_eff) ) )  )
          gamma_R(:,:) = CI * (  sgm_R(:,:,ik_eff) - CONJG( TRANSPOSE( sgm_R(:,:,ik_eff) ) )  )


          IF ( .NOT. write_lead_sgm ) THEN
              !
              DEALLOCATE( sgm_L, sgm_R, STAT=ierr)
              IF ( ierr/=0 ) CALL errore(subname,"deallocating sgm_L, sgm_R", ABS(ierr) )
              !
          ENDIF


          !
          !=================================== 
          ! Transmittance
          !=================================== 
          !
          ! evaluate the transmittance according to the Fisher-Lee formula
          ! or (in the correlated case) to the generalized expression as 
          ! from PRL 94, 116802 (2005)
          !
          CALL transmittance( dimC, gamma_L, gamma_R, gC, blc_00C, conduct_formula, &
                              cond_aux, do_eigenchannels, do_eigplot, z_eigplot )

          !
          ! free some memory
          !
          DEALLOCATE( gamma_L, gamma_R, STAT=ierr ) 
          IF ( ierr/=0 ) CALL errore(subname,"deallocating gamma_L, gamma_R", ABS(ierr) )
          DEALLOCATE( gC, STAT=ierr ) 
          IF ( ierr/=0 ) CALL errore(subname,"deallocating gC", ABS(ierr) )


          !
          ! get the total trace
          !
          DO i=1,dimC
              !
              conduct(1,ie_g)       = conduct(1,ie_g)      + wk_par(ik) * cond_aux(i)
              conduct_k(1,ik,ie_g)  = conduct_k(1,ik,ie_g) + wk_par(ik) * cond_aux(i)
              !
          ENDDO
          !
          ! resolve over eigenchannels
          ! NOTE: the # of non-null eigenchannels is lower than MIN( dimC, dimR, dimL )
          !
          IF ( do_eigenchannels ) THEN
              !
              conduct( 2:neigchn+1, ie_g )   = conduct( 2:neigchn+1, ie_g ) &
                                                  + wk_par(ik) * cond_aux( 1:neigchn )
              !
              conduct_k(2:neigchn+1,ik,ie_g) = conduct_k(2:neigchn+1,ik,ie_g) &
                                                  + wk_par(ik) * cond_aux( 1:neigchn )
              !
          ENDIF
          !
          IF ( do_eigenchannels .AND. do_eigplot .AND. &
               ik == ik_eigplot .AND. ie_g == ie_eigplot ) THEN
              !
              write_eigchn = .TRUE.
              !
          ELSE
              !
              write_eigchn = .FALSE.
              !
          ENDIF
          
          !
          ! write auxiliary data for eigenchannel analysis
          !
          IF ( write_eigchn ) THEN
              !
              CALL wd_write_eigchn( aux_unit, ie_eigplot, ik_eigplot, vkpt_par3D(:,ik) , &
                                    transport_dir, dimC, neigchn, z_eigplot)
              !
          ENDIF
          ! 
      ENDDO kpt_loop 

      !
      ! write massive data for lead sgm
      !
      IF ( write_lead_sgm ) THEN
          !
          DO ir = 1, nrtot_par
              !
              CALL compute_rham(dimC, vr_par3D(:,ir), rsgm_L(:,:,ir), nkpts_par, vkpt_par3D, wk_par, sgm_L)
              CALL compute_rham(dimC, vr_par3D(:,ir), rsgm_R(:,:,ir), nkpts_par, vkpt_par3D, wk_par, sgm_R)
              !
          ENDDO
          !
          CALL operator_write_data( sgmL_unit, rsgm_L, .TRUE., ie_g )
          CALL operator_write_data( sgmR_unit, rsgm_R, .TRUE., ie_g )
          !
      ENDIF
      !
      ! write massive data for conductor GF
      !
      IF ( write_gf ) THEN
          !
          DO ir = 1, nrtot_par
              !
              CALL compute_rham(dimC, vr_par3D(:,ir), rgC(:,:,ir), nkpts_par, vkpt_par3D, wk_par, kgC)
              !
          ENDDO
          !
          CALL operator_write_data( gf_unit, rgC, .TRUE., ie_g )
          !
      ENDIF


      !
      ! report to stdout
      !
      avg_iter = avg_iter/REAL(2*nkpts_par)
      !
      IF ( MOD( ie_g, nprint) == 0 .OR.  ie_g == iomg_s .OR. ie_g == iomg_e ) THEN
          !
          IF ( ionode ) WRITE(stdout,"(2x,'T matrix converged after avg. # of iterations ',&
                                      & f10.3,/)") avg_iter
          !
          CALL timing_upto_now(stdout)
          !
      ENDIF
      !
      CALL flush_unit(stdout)

   ENDDO energy_loop

   !
   ! recover over frequencies
   !
   CALL mp_sum( dos )
   CALL mp_sum( dos_k )
   CALL mp_sum( conduct )
   CALL mp_sum( conduct_k )

   !
   ! close sgm file's
   !
   IF ( lhave_corr ) THEN
       !
       CALL correlation_finalize( )
       !
   ENDIF
       
   !
   ! close lead sgm output files
   !
   IF ( write_lead_sgm ) THEN
       !
       CALL operator_write_close(sgmL_unit)
       CALL operator_write_close(sgmR_unit)
       !
   ENDIF
   !
   IF ( write_gf ) THEN
       !
       CALL operator_write_close(gf_unit)
       !
   ENDIF



   !
   ! write DOS and CONDUCT data on files
   !
   CALL write_header( stdout, "Writing data" )
   CALL flush_unit( stdout )

    IF ( TRIM(carriers) == 'phonons') egrid(:)=dsqrt(egrid(:)*rydcm1**2/amconv)

   ! 
   CALL wd_write_data(aux_unit, ne, egrid, SIZE(conduct,1), conduct, 'conductance' ) 
   !
   CALL wd_write_data(aux_unit, ne, egrid, 1, dos, 'doscond' ) 
                            
   !
   !  write kpoint-resolved dos and transmittance data on files
   !
   IF ( write_kdata .AND. ionode ) THEN
      !
      DO ik = 1, nkpts_par
         !
         WRITE( ctmp , "(i4.4)" ) ik
         filename= TRIM(work_dir)//'/'//TRIM(prefix)// &
                                        '_cond-'//ctmp//TRIM(postfix)//'.dat'
         !
         OPEN( aux_unit, FILE=TRIM(filename), FORM='formatted', IOSTAT=ierr )
         !
         IF (ierr/=0) CALL errore(subname,'opening '//TRIM(filename),ABS(ierr))
         !
         WRITE( aux_unit, *) "# E (eV)   cond(E)"
         !
         DO ie_g = 1, ne
             WRITE( aux_unit, '(I2,2000(f15.9))') ik, egrid(ie_g), conduct_k(:,ik,ie_g) 
         ENDDO
         !
         CLOSE( aux_unit )         
         !
      ENDDO
      !
      !
      DO ik = 1, nkpts_par
         !
         WRITE( ctmp , "(i4.4)" ) ik
         filename= TRIM(work_dir)//'/'//TRIM(prefix)// &
                                       '_doscond-'//ctmp//TRIM(postfix)//'.dat'

         OPEN( aux_unit, FILE=TRIM(filename), FORM='formatted', IOSTAT=ierr )
         !
         IF (ierr/=0) CALL errore(subname,'opening '//TRIM(filename),ABS(ierr))
         !
         WRITE( aux_unit, *) "# E (eV)   doscond(E)"
         !
         DO ie_g = 1, ne
             WRITE( aux_unit, '(2(f15.9))') egrid(ie_g), dos_k(ie_g,ik) 
         ENDDO
         !
         CLOSE( aux_unit )         
         !
      ENDDO
      !
   ENDIF
   

!
! ... shutdown
!

   !
   ! clean local memory
   !
   DEALLOCATE ( dos, STAT=ierr )
   IF( ierr /=0 ) CALL errore(subname,'deallocating dos', ABS(ierr) )
   !
   DEALLOCATE ( conduct_k, conduct, STAT=ierr )
   IF( ierr /=0 ) CALL errore(subname,'deallocating conduct_k, conduct', ABS(ierr) )
   !
   DEALLOCATE ( cond_aux, STAT=ierr )
   IF( ierr /=0 ) CALL errore(subname,'deallocating cond_aux', ABS(ierr) )
   !
   IF ( ALLOCATED( z_eigplot) ) THEN
       DEALLOCATE ( z_eigplot, STAT=ierr )
       IF( ierr /=0 ) CALL errore(subname,'deallocating z_eigplot', ABS(ierr) )
   ENDIF


   CALL timing(subname,OPR='stop')
   CALL log_pop(subname)
      !
END SUBROUTINE do_conductor
   

