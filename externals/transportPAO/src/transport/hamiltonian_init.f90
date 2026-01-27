!
!      Copyright (C) 2005 WanT Group, 2017 ERMES Group
!
!      This file is distributed under the terms of the
!      GNU General Public License. See the file `License'
!      in the root directory of the present distribution,
!      or http://www.gnu.org/copyleft/gpl.txt .
!
!*******************************************************************
   SUBROUTINE hamiltonian_init( )
   !*******************************************************************
   !
   ! Initialize hamiltonian data:
   !
   !...  Matrix definition
   !
   !     Given a conductor (C) bonded to a right lead (A) and a left lead (B)
   !
   !       H01_L    H00_L   H_LC    H00_C     H_CR   H00_R   H01_R
   !       S01_L    S00_L   S_LC    S00_C     S_CR   S00_R   S01_R
   !   ...--------------------------------------------------------------...
   !         |                |                   |                | 
   !         |     LEAD L     |    CONDUCTOR C    |     LEAD R     |
   !         |                |                   |                | 
   !   ...--------------------------------------------------------------...
   !
   !     H00_L, H00_R    = on site hamiltonian of the leads (from bulk calculation)
   !     H01_L, H01_R    = hopping hamiltonian of the leads (from bulk calculation)
   !     H00_C           = on site hamiltonian of the conductor (from supercell calculation)
   !     H_LC, H_CR  = coupling matrix between leads and conductor 
   !                       (from supercell calculation)
   !
   !     S00_L, S00_R, S00_C  = on site overlap matrices
   !     S01_L, S01_R         = hopping overlap matrices
   !     S_LC, S_CR           = coupling overlap matrices
   !
   !...  Units
   !     energies are supposed to be in eV
   !
   USE kinds
   USE io_module,            ONLY : stdin, ionode, ionode_id
   USE log_module,           ONLY : log_push, log_pop
   USE timing_module,        ONLY : timing
   USE mp,                   ONLY : mp_bcast
   USE util_module,          ONLY : mat_herm
   USE T_control_module,     ONLY : calculation_type, idir => transport_dir, &
                                    datafile_L, datafile_C, datafile_R, &
                                    leads_are_identical
   USE T_hamiltonian_module, ONLY : hamiltonian_allocate, ispin,        &
                                    blc_00L, blc_01L, blc_00R, blc_01R, &
                                    blc_00C, blc_LC,  blc_CR
   USE T_operator_blc_module
   USE T_correlation_module, ONLY : datafile_R_sgm, datafile_L_sgm
   USE iotk_module
   !
   IMPLICIT NONE

   !
   ! local variables
   !
   CHARACTER(16) :: subname="hamiltonian_init"
   INTEGER       :: ik, ierr

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
   ! allocations
   !
   CALL hamiltonian_allocate()

   !
   ! Read the HAMILTONIAN_DATA card from input file
   !
   IF ( ionode ) THEN
       !
       CALL iotk_scan_begin( stdin, 'HAMILTONIAN_DATA', IERR=ierr )
       IF (ierr/=0) CALL errore(subname,'searching HAMILTONIAN_DATA',ABS(ierr))
       !
       ! these data must always be present
       !
       CALL iotk_scan_empty( stdin, "H00_C", ATTR=blc_00C%tag, IERR=ierr)
       IF (ierr/=0) CALL errore(subname, 'searching for tag H00_C', ABS(ierr) )       
       !
       CALL iotk_scan_empty( stdin, "H_CR", ATTR=blc_CR%tag, IERR=ierr)
       IF (ierr/=0) CALL errore(subname, 'searching for tag H_CR', ABS(ierr) )       
       !
       ! read the remaing data if the case
       !
       IF ( TRIM(calculation_type) == "conductor" ) THEN
           !
           CALL iotk_scan_empty( stdin, "H_LC", ATTR=blc_LC%tag, IERR=ierr)
           IF (ierr/=0) CALL errore(subname, 'searching for tag H_LC', ABS(ierr) )       
           !
           CALL iotk_scan_empty( stdin, "H00_L", ATTR=blc_00L%tag, IERR=ierr)
           IF (ierr/=0) CALL errore(subname, 'searching for tag H00_L', ABS(ierr) )       
           CALL iotk_scan_empty( stdin, "H01_L", ATTR=blc_01L%tag, IERR=ierr)
           IF (ierr/=0) CALL errore(subname, 'searching for tag H01_L', ABS(ierr) )       
           !
           CALL iotk_scan_empty( stdin, "H00_R", ATTR=blc_00R%tag, IERR=ierr)
           IF (ierr/=0) CALL errore(subname, 'searching for tag H00_R', ABS(ierr) )       
           CALL iotk_scan_empty( stdin, "H01_R", ATTR=blc_01R%tag, IERR=ierr)
           IF (ierr/=0) CALL errore(subname, 'searching for tag H01_R', ABS(ierr) )       
           !
       ENDIF
       !
       CALL iotk_scan_end( stdin, 'HAMILTONIAN_DATA', IERR=ierr )
       IF (ierr/=0) CALL errore(subname,'searching end for HAMILTONIAN_DATA',ABS(ierr))
       !
   ENDIF
   !
   CALL mp_bcast(  blc_00C%tag,      ionode_id )
   CALL mp_bcast(  blc_CR%tag,       ionode_id )
   CALL mp_bcast(  blc_LC%tag,       ionode_id )
   CALL mp_bcast(  blc_00L%tag,      ionode_id )
   CALL mp_bcast(  blc_01L%tag,      ionode_id )
   CALL mp_bcast(  blc_00R%tag,      ionode_id )
   CALL mp_bcast(  blc_01R%tag,      ionode_id )



   !
   ! Read basic quantities from datafile
   !
   CALL read_matrix( datafile_C, ispin, idir, blc_00C )
   CALL read_matrix( datafile_C, ispin, idir, blc_CR )

   !
   ! chose whether to do 'conductor' or 'bulk'
   !
   SELECT CASE ( TRIM(calculation_type) )

   CASE ( "conductor" )
       !
       ! read the missing data
       !
       CALL read_matrix( datafile_C, ispin, idir, blc_LC )
       CALL read_matrix( datafile_L, ispin, idir, blc_00L )
       CALL read_matrix( datafile_L, ispin, idir, blc_01L )
       CALL read_matrix( datafile_R, ispin, idir, blc_00R )
       CALL read_matrix( datafile_R, ispin, idir, blc_01R )
       !
   CASE ( "bulk" )
       !
       ! rearrange the data already read
       !
       blc_00L = blc_00C
       blc_00R = blc_00C
       blc_01L = blc_CR
       blc_01R = blc_CR
       blc_LC  = blc_CR
       !
   CASE DEFAULT
       !
       CALL errore(subname,'Invalid calculation_type = '// TRIM(calculation_type),5)
       !
   END SELECT

   !
   ! Force hemiticity on blc-on-site hamiltonians and overlaps
   ! Non-hermiticity can raise due to non-full convergence wrt R-grids and kpt
   ! in the original WF calculation.
   !
   IF ( blc_00C%nkpts /= blc_00L%nkpts ) CALL errore(subname,'unexpected error, nkpts diff',10)
   IF ( blc_00C%nkpts /= blc_00R%nkpts ) CALL errore(subname,'unexpected error, nkpts diff',11)
   !
   DO ik = 1, blc_00C%nkpts
       !
       CALL mat_herm( blc_00C%H(:,:,ik), blc_00C%dim1 )
       CALL mat_herm( blc_00C%S(:,:,ik), blc_00C%dim1 )
       !
       CALL mat_herm( blc_00L%H(:,:,ik), blc_00L%dim1 )
       CALL mat_herm( blc_00L%S(:,:,ik), blc_00L%dim1 )
       ! 
       CALL mat_herm( blc_00R%H(:,:,ik), blc_00R%dim1 )
       CALL mat_herm( blc_00R%S(:,:,ik), blc_00R%dim1 )
       ! 
   ENDDO
   

   !
   ! finally check whether the two leads are the same
   !
   leads_are_identical = .FALSE.
   !
   IF ( TRIM(datafile_L)      == TRIM(datafile_R)     .AND. & 
        TRIM(datafile_L_sgm)  == TRIM(datafile_R_sgm) .AND. &
        !
        ALL( blc_00L%icols(:) == blc_00R%icols(:) )   .AND. &
        ALL( blc_00L%irows(:) == blc_00R%irows(:) )   .AND. &  
        ALL( blc_01L%icols(:) == blc_01R%icols(:) )   .AND. &  
        ALL( blc_01L%irows(:) == blc_01R%irows(:) )   .AND. &
        !
        ALL( blc_00L%icols_sgm(:) == blc_00R%icols_sgm(:) )   .AND. &
        ALL( blc_00L%irows_sgm(:) == blc_00R%irows_sgm(:) )   .AND. &  
        ALL( blc_01L%icols_sgm(:) == blc_01R%icols_sgm(:) )   .AND. &  
        ALL( blc_01L%irows_sgm(:) == blc_01R%irows_sgm(:) ) ) THEN 
       !
       leads_are_identical = .TRUE.
       !
       CALL operator_blc_deallocate( OBJ=blc_00L )
       CALL operator_blc_deallocate( OBJ=blc_01L )
       !
   ENDIF


   CALL timing( subname, OPR='STOP' )
   CALL log_pop( subname )
   !
   RETURN
   !
END SUBROUTINE hamiltonian_init

