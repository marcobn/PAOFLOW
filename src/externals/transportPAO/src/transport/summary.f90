!
! Copyright (C) 2005 WanT Group, 2017 ERMES Group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License\'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!

!**********************************************************
   SUBROUTINE summary(iunit)
   !**********************************************************
   ! 
   ! Print out all the informnatins obtained from the 
   ! input and initialization routines.
   !
   USE kinds,                ONLY : dbl
   USE constants,            ONLY : ZERO
   USE parser_module,        ONLY : log2char
   USE mp_global,            ONLY : nproc
   USE log_module,           ONLY : log_push, log_pop
   USE io_module,            ONLY : work_dir, prefix, postfix, ionode
   USE T_hamiltonian_module, ONLY : dimL, dimC, dimR, &
                                    shift_L, shift_C, shift_R, shift_corr
   USE T_correlation_module, ONLY : lhave_corr, &
                                    datafile_C_sgm, datafile_L_sgm, datafile_R_sgm
   USE T_control_module,     ONLY : calculation_type, conduct_formula,  &
                                    datafile_C, datafile_L, datafile_R, &
                                    transport_dir, niterx, nprint, & 
                                    write_kdata, write_lead_sgm, write_gf, &
                                    do_orthoovp, leads_are_identical
   USE T_egrid_module,       ONLY : ne, emin, emax, de, ne_buffer
   USE T_smearing_module,    ONLY : delta, smearing_type, nx_smear => nx, xmax
   USE T_kpoints_module,     ONLY : nkpts_par, nk_par, s_par, vkpt_par3D, wk_par, use_symm, &
                                    kpoints_alloc => alloc, kpoints_imask
   USE T_kpoints_module,     ONLY : nrtot_par, nr_par, ivr_par3D, wr_par
   USE T_input_parameters_module, ONLY : carriers
   !
   IMPLICIT NONE

   !
   ! input variables
   !
   INTEGER,   INTENT(in)  :: iunit

   !
   ! local variables
   !
   INTEGER      :: ik, ir
   INTEGER      :: nk_par3D(3)       ! 3D kpt mesh generator
   INTEGER      :: s_par3D(3)        ! 3D shifts
   INTEGER      :: nr_par3D(3)       ! 3D R-vect mesh generator
   !
   ! end of declaration
   !

    REAL(dbl) :: rydcm1 = 13.6058d0*8065.5d0
    REAL(dbl) :: amconv = 1.66042d-24/9.1095d-28*0.5d0

!
!------------------------------
! main body
!------------------------------
!
   CALL log_push( 'summary' )
   !
   CALL write_header( iunit, "INPUT Summary" )

   !
   ! <INPUT> section
   !
   IF ( ionode ) THEN
       !
       WRITE(iunit,"( 2x,'<INPUT>')" )
       WRITE(iunit,"( 7x,'   Calculation Type :',5x,a)") TRIM(calculation_type)
       WRITE(iunit,"( 7x,'             prefix :',5x,   a)") TRIM(prefix)
       WRITE(iunit,"( 7x,'            postfix :',5x,   a)") TRIM(postfix)
       IF ( LEN_TRIM(work_dir) <= 65 ) THEN
          WRITE(iunit,"(7x,'           work_dir :',5x,   a)") TRIM(work_dir)
       ELSE
          WRITE(iunit,"(7x,'           work_dir :',5x,/,10x,a)") TRIM(work_dir)
       ENDIF
       WRITE(iunit,"( 7x,'        L-lead dim. :',5x,i5)") dimL
       WRITE(iunit,"( 7x,'     conductor dim. :',5x,i5)") dimC
       WRITE(iunit,"( 7x,'        R-lead dim. :',5x,i5)") dimR
       WRITE(iunit,"( 7x,'Conductance Formula :',5x,a)") TRIM(conduct_formula)
       WRITE(iunit,"( 7x,'           Carriers :',5x,a)") TRIM(carriers)
       WRITE(iunit,"( 7x,'Transport Direction :',8x,i2)") transport_dir
       WRITE(iunit,"( 7x,'   Have Correlation :',5x,a)") log2char(lhave_corr)
       WRITE(iunit,"( 7x,'       Write k-data :',5x,a)") log2char(write_kdata)
       WRITE(iunit,"( 7x,'     Write sgm lead :',5x,a)") log2char(write_lead_sgm)
       WRITE(iunit,"( 7x,'         Write gf C :',5x,a)") log2char(write_gf)
       WRITE(iunit,"( 7x,'    Max iter number :',5x,i5)") niterx
       WRITE(iunit,"( 7x,'             nprint :',5x,i5)") nprint
       WRITE(iunit,"( )")
       WRITE(iunit,"( 7x,' Conductor datafile :',5x,a)") TRIM(datafile_C)
       IF (calculation_type == 'conductor') THEN
          WRITE(iunit,"( 7x,'    L-lead datafile :',5x,a)") TRIM(datafile_L)
          WRITE(iunit,"( 7x,'    R-lead datafile :',5x,a)") TRIM(datafile_R)
       ENDIF
       IF (lhave_corr) THEN
          WRITE(iunit,"( 7x,'     L-Sgm datafile :',5x,a)") TRIM(datafile_L_sgm)
          WRITE(iunit,"( 7x,'     C-Sgm datafile :',5x,a)") TRIM(datafile_C_sgm)
          WRITE(iunit,"( 7x,'     R-Sgm datafile :',5x,a)") TRIM(datafile_R_sgm)
       ENDIF
       WRITE(iunit,"( 7x,'leads are identical :',5x,a)") log2char(leads_are_identical)
       WRITE(iunit,"( 7x,'  ovp orthogonaliz. :',5x,a)") log2char(do_orthoovp)
       WRITE( iunit,"( 2x,'</INPUT>',2/)" )
    
       WRITE(iunit,"( 2x,'<ENERGY_GRID>')" )
       WRITE(iunit,"( 7x,'          Dimension :',5x,i6)")    ne
       WRITE(iunit,"( 7x,'          Buffering :',5x,i6)")    ne_buffer
       IF (TRIM(carriers) == 'phonons') THEN
       WRITE(iunit,"( 7x,'      Min Frequency :',5x,f10.5)") emin*(rydcm1/dsqrt(amconv))**2
       WRITE(iunit,"( 7x,'      Max Frequency :',5x,f10.5)") emax*(rydcm1/dsqrt(amconv))**2
       WRITE(iunit,"( 7x,'        Energy Step :',5x,f10.5)") de*(rydcm1/dsqrt(amconv))**2
       ELSE
       WRITE(iunit,"( 7x,'         Min Energy :',5x,f10.5)") emin
       WRITE(iunit,"( 7x,'         Max Energy :',5x,f10.5)") emax
       WRITE(iunit,"( 7x,'        Energy Step :',5x,f10.5)") de
       ENDIF
       WRITE(iunit,"( 7x,'              Delta :',5x,f10.5)") delta
       WRITE(iunit,"( 7x,'      Smearing Type :',5x,a)")     TRIM(smearing_type)
       WRITE(iunit,"( 7x,'      Smearing grid :',5x,i6)")    nx_smear
       WRITE(iunit,"( 7x,'      Smearing gmax :',5x,f10.5)") xmax
       WRITE(iunit,"( 7x,'            Shift_L :',5x,f10.5)") shift_L
       WRITE(iunit,"( 7x,'            Shift_C :',5x,f10.5)") shift_C
       WRITE(iunit,"( 7x,'            Shift_R :',5x,f10.5)") shift_R
       WRITE(iunit,"( 7x,'         Shift_corr :',5x,f10.5)") shift_corr
       WRITE(iunit,"( 2x,'</ENERGY_GRID>',/)" )
       !
   ENDIF

   IF ( kpoints_alloc .AND. ionode ) THEN
       !
       WRITE( iunit, "( /,2x,'<K-POINTS>')" )
       WRITE( iunit, "( 7x, 'nkpts_par = ',i4 ) " ) nkpts_par
       WRITE( iunit, "( 7x, 'nrtot_par = ',i4 ) " ) nrtot_par
       WRITE( iunit, "( 7x, ' use_symm = ',a  ) " ) TRIM(log2char(use_symm))
       !
       !
       nk_par3D(:) = kpoints_imask( nk_par, 1, transport_dir )
       s_par3D(:)  = kpoints_imask(  s_par, 0, transport_dir )
       !
       WRITE( iunit, "(/,7x, 'Parallel kpoints grid:',8x, &
                           &'nk = (',3i3,' )',3x,'s = (',3i3,' )') " ) nk_par3D(:), s_par3D(:) 
       !
       DO ik=1,nkpts_par
           !
           WRITE( iunit, "(7x, 'k (', i4, ') =    ( ',3f9.5,' ),   weight = ', f8.4 )") &
                  ik, vkpt_par3D(:,ik), wk_par(ik)
           !
       ENDDO
       !
       nr_par3D(:) = kpoints_imask( nr_par, 1, transport_dir )
       !
       WRITE( iunit, "(/,7x, 'Parallel R vector grid:       nr = (',3i3,' )') " ) nr_par3D(:) 
       !
       DO ir=1,nrtot_par
           !
           WRITE( iunit, "(7x, 'R (', i4, ') =    ( ',3f9.5,' ),   weight = ', f8.4 )") &
                  ir, REAL( ivr_par3D(:,ir),dbl), wr_par(ir)
           !
       ENDDO
       !    
       WRITE( iunit, " ( 2x,'</K-POINTS>',/)" )
       !
   ENDIF
   !
   !
   IF ( ionode ) THEN
       !
       WRITE( iunit, "( /,2x,'<PARALLELISM>')" )
       WRITE( iunit, "(   7x, 'Paralellization over frequencies' ) " )
       WRITE( iunit, "(   7x, '# of processes: ', i5 ) " ) nproc
       WRITE( iunit, "(   2x,'</PARALLELISM>',/)" )
       !
   ENDIF
   !
   CALL flush_unit( iunit )
   CALL log_pop( 'summary' )
   !
   RETURN
   !
END SUBROUTINE summary

