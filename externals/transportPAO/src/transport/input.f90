! 
! Copyright (C) 2005 WanT Group, 2017 ERMES Group
! 
! This file is distributed under the terms of the 
! GNU General Public License. See the file `License' 
! in the root directory of the present distribution, 
! or http://www.gnu.org/copyleft/gpl.txt . 
! 
!********************************************
   MODULE T_input_module
   !********************************************
   !
   USE kinds,      ONLY : dbl
   USE parameters, ONLY : nstrx
   USE io_module,  ONLY : stdin, log_unit, io_name
   USE log_module, ONLY : log_init, log_push
   USE constants,  ONLY : ZERO
   !
   IMPLICIT NONE
   PRIVATE
!
! This module handles the reading of input data
!
! routines in this module:
! SUBROUTINE input_manager()
! SUBROUTINE setup_control()
! SUBROUTINE setup_hamiltonian()
! SUBROUTINE setup_kpoints()
! SUBROUTINE setup_egrid()
! SUBROUTINE setup_smearing()
! 


   PUBLIC :: input_manager


CONTAINS

!
! subroutines
!


!**********************************************************
   SUBROUTINE input_manager()
   !**********************************************************
      !
      USE T_input_parameters_module,  ONLY : read_namelist_input_conductor
      !
      IMPLICIT NONE

      !
      ! attach input from file if the case
      !
      CALL input_from_file ( stdin )

      !
      ! reading and checking namelists
      !
      CALL read_namelist_input_conductor(stdin)

      !
      ! scattering data in their own modules
      !
      CALL setup_control()
      CALL setup_io()
      CALL setup_egrid()
      CALL setup_smearing()
      CALL setup_hamiltonian()
      CALL setup_correlation()
      CALL setup_kpoints()


   END SUBROUTINE input_manager


!**********************************************************
   SUBROUTINE setup_control()
   !**********************************************************
      USE T_control_module,         ONLY : calculation_type,  &
                                           conduct_formula,   &
                                           niterx,            &
                                           nprint,            &
                                           bias,              &
                                           transport_dir,     &
                                           datafile_L,        &
                                           datafile_C,        &
                                           datafile_R,        &
                                           do_orthoovp,       &
                                           write_kdata,       &
                                           write_lead_sgm,    &
                                           write_gf,          &
                                           debug_level,       &
                                           use_debug_mode,    &
                                           do_eigenchannels,  &
                                           neigchnx,          &
                                           do_eigplot,        &
                                           ie_eigplot,        &
                                           ik_eigplot,        &
                                           transfer_thr,      &
                                           nfailx
                                            
      USE T_input_parameters_module,ONLY : calculation_type_  => calculation_type, &
                                           conduct_formula_   => conduct_formula, &
                                           niterx_            => niterx, &
                                           nprint_            => nprint, &
                                           bias_              => bias, &
                                           transport_dir_     => transport_dir, &
                                           datafile_L_        => datafile_L, &
                                           datafile_C_        => datafile_C, &
                                           datafile_R_        => datafile_R, &
                                           do_orthoovp_       => do_orthoovp, &
                                           write_kdata_       => write_kdata, &
                                           write_lead_sgm_    => write_lead_sgm, &
                                           write_gf_          => write_gf, &
                                           debug_level_       => debug_level, &
                                           do_eigenchannels_  => do_eigenchannels, &
                                           neigchnx_          => neigchnx, &
                                           do_eigplot_        => do_eigplot, &
                                           ie_eigplot_        => ie_eigplot, &
                                           ik_eigplot_        => ik_eigplot, &
                                           transfer_thr_      => transfer_thr, &
                                           nfailx_            => nfailx

      IMPLICIT NONE

      calculation_type    = calculation_type_
      conduct_formula     = conduct_formula_
      niterx              = niterx_
      nprint              = nprint_
      bias                = bias_
      datafile_L          = datafile_L_
      datafile_C          = datafile_C_
      datafile_R          = datafile_R_
      do_orthoovp         = do_orthoovp_
      transport_dir       = transport_dir_
      write_kdata         = write_kdata_
      write_lead_sgm      = write_lead_sgm_
      write_gf            = write_gf_
      debug_level         = debug_level_
      do_eigenchannels    = do_eigenchannels_
      neigchnx            = neigchnx_
      do_eigplot          = do_eigplot_
      ie_eigplot          = ie_eigplot_
      ik_eigplot          = ik_eigplot_
      nfailx              = nfailx_
      transfer_thr        = transfer_thr_

      use_debug_mode = .FALSE.
      IF ( debug_level_ > 0 )  use_debug_mode  = .TRUE.

   END SUBROUTINE setup_control
      

!**********************************************************
   SUBROUTINE setup_io()
   !**********************************************************
      USE io_module,                ONLY : work_dir, &
                                           prefix,   &
                                           postfix
      USE atmproj_tools_module,     ONLY : atmproj_sh, &
                                           atmproj_thr, &
                                           atmproj_nbnd
      USE T_control_module,         ONLY : debug_level, &
                                           use_debug_mode
      USE T_input_parameters_module,ONLY : work_dir_       => work_dir,    &
                                           prefix_         => prefix,      &
                                           postfix_        => postfix,     &
                                           atmproj_sh_     => atmproj_sh,  &
                                           atmproj_thr_    => atmproj_thr, &
                                           atmproj_nbnd_   => atmproj_nbnd,&
                                           debug_level_    => debug_level
      IMPLICIT NONE
      !
      CHARACTER( nstrx ) :: logfile

      work_dir            = work_dir_
      prefix              = prefix_
      postfix             = postfix_
      !
      atmproj_sh          = atmproj_sh_
      atmproj_thr         = atmproj_thr_
      atmproj_nbnd        = atmproj_nbnd_

      !
      ! this part is replicated for the sake of robustness
      !
      debug_level         = debug_level_
      use_debug_mode      = .FALSE.
      !
      IF ( debug_level_ > 0 )  use_debug_mode  = .TRUE.

      CALL io_name( "log", logfile, LBODY=.TRUE. )
      !
      CALL log_init( log_unit, use_debug_mode, logfile, debug_level)
      CALL log_push("main")

   END SUBROUTINE setup_io


!**********************************************************
   SUBROUTINE setup_egrid()
   !**********************************************************
      USE T_egrid_module,           ONLY : ne,           &
                                           ne_buffer,    &
                                           emin, emax
      USE T_input_parameters_module,ONLY : ne_          => ne,   &
                                           ne_buffer_   => ne_buffer, &
                                           emin_        => emin, &
                                           emax_        => emax 
      IMPLICIT NONE

      ne          = ne_
      ne_buffer   = ne_buffer_
      emin        = emin_
      emax        = emax_

   END SUBROUTINE setup_egrid


!**********************************************************
   SUBROUTINE setup_smearing()
   !**********************************************************
      USE T_smearing_module,        ONLY : delta,         &
                                           smearing_type, &
                                           delta_ratio, xmax

      USE T_input_parameters_module,ONLY : delta_         => delta,         &
                                           smearing_type_ => smearing_type, &
                                           delta_ratio_   => delta_ratio,   &
                                           xmax_          => xmax

      IMPLICIT NONE

      delta         = delta_
      smearing_type = smearing_type_
      delta_ratio   = delta_ratio_
      xmax          = xmax_

   END SUBROUTINE setup_smearing


!**********************************************************
   SUBROUTINE setup_hamiltonian()
   !**********************************************************
      USE T_hamiltonian_module, ONLY :     dimL,    &
                                           dimR,    &
                                           dimC,    & 
                                           dimx,    & 
                                           ispin,   &
                                           shift_L, &
                                           shift_C, &
                                           shift_R
      USE T_input_parameters_module,ONLY : dimL_     => dimL,    &
                                           dimR_     => dimR,    &
                                           dimC_     => dimC,    &
                                           ispin_    => ispin,   &
                                           shift_L_  => shift_L, &
                                           shift_C_  => shift_C, &
                                           shift_R_  => shift_R
      IMPLICIT NONE
      ! 
      dimL            = dimL_
      dimR            = dimR_
      dimC            = dimC_
      dimx            = MAX( dimL, dimR, dimC)
      !
      ispin           = ispin_
      shift_L         = shift_L_
      shift_C         = shift_C_
      shift_R         = shift_R_
      !
   END SUBROUTINE setup_hamiltonian


!**********************************************************
   SUBROUTINE setup_kpoints()
   !**********************************************************
      USE T_kpoints_module,         ONLY :     nk_par, s_par, use_symm, use_safe_kmesh
      USE T_input_parameters_module,ONLY :     nk, s, &
                                               write_lead_sgm, write_gf, &
                                               use_symm_ => use_symm
      IMPLICIT NONE
      
      nk_par(1:2)      = nk(1:2)
      s_par(1:2)       = s(1:2)
      use_symm         = use_symm_
      use_safe_kmesh   = write_lead_sgm .OR. write_gf

   END SUBROUTINE setup_kpoints


!**********************************************************
   SUBROUTINE setup_correlation()
   !**********************************************************
      USE T_correlation_module,     ONLY :     lhave_corr, &
                                               shift_C_corr, &
                                               datafile_L_sgm, &
                                               datafile_C_sgm, &
                                               datafile_R_sgm
      USE T_input_parameters_module,ONLY :     datafile_sgm_   => datafile_sgm, &
                                               datafile_L_sgm_ => datafile_L_sgm, &
                                               datafile_C_sgm_ => datafile_C_sgm, &
                                               datafile_R_sgm_ => datafile_R_sgm, &
                                               shift_corr_     => shift_corr
      IMPLICIT NONE
    
      datafile_C_sgm   = datafile_C_sgm_
      datafile_L_sgm   = datafile_L_sgm_
      datafile_R_sgm   = datafile_R_sgm_
      !
      IF ( LEN_TRIM(datafile_sgm_) /=0 .AND. LEN_TRIM(datafile_C_sgm_) == 0 ) THEN
           datafile_C_sgm = datafile_sgm_     
      ENDIF
      !
      lhave_corr       = ( LEN_TRIM( datafile_L_sgm  ) /= 0  .OR. &
                           LEN_TRIM( datafile_C_sgm  ) /= 0  .OR. &
                           LEN_TRIM( datafile_R_sgm  ) /= 0   )
      !
      shift_C_corr     = shift_corr_


   END SUBROUTINE setup_correlation

END MODULE T_input_module

