! 
! Copyright (C) 2006 WanT Group, 2017 ERMES Group
! 
! This file is distributed under the terms of the 
! GNU General Public License. See the file `License' 
! in the root directory of the present distribution, 
! or http://www.gnu.org/copyleft/gpl.txt . 
!
!*****************************************************************
   SUBROUTINE define_smear_ham( hdim, c00, c01, s00, s01, c00_eff, c01_eff, smearing_type )
   !*****************************************************************
   !
   ! Given c00 and c01, the routine computes the 
   ! effective hamiltonian blocks according to the used smearing
   !
   ! c00  = \omega S00 - H00
   ! c01  = \omega S01 - H01
   !
   USE kinds
   USE constants,          ONLY : ONE, CZERO, CI, TPI
   USE T_smearing_module,  ONLY : smear_alloc => alloc, delta, nkpts_smear
   USE timing_module,      ONLY : timing
   USE log_module,         ONLY : log_push, log_pop
   !
   IMPLICIT NONE 

   !  
   ! input variables 
   !  
   INTEGER,      INTENT(in)  :: hdim
   COMPLEX(dbl), INTENT(in)  :: c00( hdim, hdim ),     c01( hdim, hdim )
   COMPLEX(dbl), INTENT(in)  :: s00( hdim, hdim ),     s01( hdim, hdim )
   COMPLEX(dbl), INTENT(out) :: c00_eff( hdim, hdim ), c01_eff( hdim, hdim )
   CHARACTER(*), INTENT(in)  :: smearing_type
   !
   ! local variables
   !
   CHARACTER(16)      :: subname = 'define_smear_ham'
   INTEGER            :: i, j, ik, ierr
   REAL(dbl)          :: arg
   COMPLEX(dbl)       :: phase
   REAL(dbl),    ALLOCATABLE :: vkpt_int(:)
   COMPLEX(dbl), ALLOCATABLE :: haux(:,:), haux_eff(:,:)
   !
   ! end of declarations 
   !

!
!----------------------------------------
! main Body
!----------------------------------------
!
   CALL timing('define_smear_ham',OPR='start')
   CALL log_push('define_smear_ham')
      
   !
   ! check
   !
   IF ( .NOT. smear_alloc ) CALL errore(subname,'smearing module not allocated',1)


   !
   ! If smearing_type is lorentzian, the usual technique is used, 
   ! otherwise a numerical interpolation is adopted.
   !
   SELECT CASE ( TRIM(smearing_type) )
   CASE ( 'lorentzian' )
        
        c00_eff(:,:)  = c00(:,:) + CI * delta * s00(:,:)
        c01_eff(:,:)  = c01(:,:) + CI * delta * s01(:,:)

   CASE DEFAULT
        !
        ! Numeric smearing:
        ! define the effective c00 and c01 according to the chosen smearing
        !
        ALLOCATE( haux( hdim, hdim ), STAT=ierr )
        IF (ierr/=0) CALL errore(subname,'allocating haux',ABS(ierr))
        ALLOCATE( haux_eff( hdim, hdim ), STAT=ierr )
        IF (ierr/=0) CALL errore(subname,'allocating haux_eff',ABS(ierr))
        ALLOCATE( vkpt_int( nkpts_smear ), STAT=ierr )
        IF (ierr/=0) CALL errore(subname,'allocating vkpt_int',ABS(ierr))
        !
        ! define the auxiliary 1D kpt mesh used to define c00_eff and c01_eff
        !
        DO ik = 1, nkpts_smear
             !
             vkpt_int( ik ) = REAL( ik-1, dbl) * ONE / REAL( nkpts_smear, dbl)
             !
        ENDDO
        !
        ! For each kpt, define the corresponding hamiltonian (haux) taking into
        ! account we only have 2 neighbors
        ! then compute the smeared hamiltonian corresponding to these haux(k)
        !
        c00_eff (:,:) = CZERO
        c01_eff (:,:) = CZERO
        !
        DO ik = 1, nkpts_smear
            !
            arg   = TPI * vkpt_int( ik )
            phase = CMPLX( COS(arg), SIN(arg), dbl ) 
            !
            ! since we are using nearest neighbors only, the Bloch sum is:
            ! Haux(k) = c00 + e^{ik*R1} c01 + e^{-ik*R1} * c01 ^dag
            !
            DO j = 1, hdim
            DO i = 1, hdim
                !
                haux( i, j) = c00(i,j) + phase * c01(i,j) +  &
                                         CONJG(phase) * CONJG( c01(j,i) )
                !
            ENDDO
            ENDDO
            !
            CALL gzero_maker( hdim, haux, s00, hdim, haux_eff, 'inverse', TRIM(smearing_type) )
            !
            !
            ! now FT back khaux(k) to c00_eff c01_eff
            !
            DO j = 1, hdim
            DO i = 1, hdim
                !
                c00_eff (i,j) = c00_eff (i,j) + haux_eff(i,j)
                c01_eff (i,j) = c01_eff (i,j) + CONJG(phase) * haux_eff(i,j)
                !    
            ENDDO
            ENDDO
            !
        ENDDO
        !
        ! impose normalization
        !
        c00_eff(:,:) = c00_eff(:,:) / REAL( nkpts_smear, dbl)
        c01_eff(:,:) = c01_eff(:,:) / REAL( nkpts_smear, dbl)

        !
        DEALLOCATE( haux, haux_eff, STAT=ierr )
        IF (ierr/=0) CALL errore(subname,'deallocating haux, haux_eff',ABS(ierr))
        DEALLOCATE( vkpt_int, STAT=ierr )
        IF (ierr/=0) CALL errore(subname,'deallocating vkpt_int',ABS(ierr))
        !
   END SELECT

   CALL timing('define_smear_ham',OPR='stop')
   CALL log_pop('define_smear_ham')

END SUBROUTINE define_smear_ham

