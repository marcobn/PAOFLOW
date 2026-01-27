!
!      Copyright (C) 2005 WanT Group, 2017 ERMES Group
!
!      This file is distributed under the terms of the
!      GNU General Public License. See the file `License'
!      in the root directory of the present distribution,
!      or http://www.gnu.org/copyleft/gpl.txt .
!
!***********************************************
   SUBROUTINE transmittance( dimC, gamma_L, gamma_R, G_ret, opr00, &
                             which_formula, conduct, do_eigenchannels, do_eigplot, z_eigplot)
   !***********************************************
   !
   ! Calculates the matrix involved in the quantum transmittance, 
   ! returned in CONDUCT variable, using the LANDAUER formula in the MEAN FIELD case
   !
   ! otherwise the generalized expression derived in:
   !     A. Ferretti et al, PRL 94, 116802 (2005).
   !
   ! is used instead.
   !
   USE kinds,                   ONLY : dbl
   USE constants,               ONLY : CZERO, CONE, CI, ZERO , EPS_m5, EPS_m6
   USE util_module,             ONLY : mat_mul, mat_sv, mat_hdiag
   USE timing_module,           ONLY : timing
   USE log_module,              ONLY : log_push, log_pop
   USE T_operator_blc_module
   !
   IMPLICIT NONE

   !
   ! input/output variables
   !
   INTEGER,             INTENT(IN)  :: dimC
   COMPLEX(dbl),        INTENT(IN)  :: gamma_L(dimC,dimC), gamma_R(dimC,dimC)
   COMPLEX(dbl),        INTENT(IN)  :: G_ret(dimC,dimC)
   !COMPLEX(dbl),        INTENT(INOUT) :: G_ret(dimC,dimC)
   TYPE(operator_blc),  INTENT(IN)  :: opr00
   CHARACTER(*),        INTENT(IN)  :: which_formula
   LOGICAL,             INTENT(IN)  :: do_eigenchannels, do_eigplot
   REAL(dbl),           INTENT(OUT) :: conduct(dimC)
   COMPLEX(dbl),        INTENT(OUT) :: z_eigplot(dimC,dimC)

   !
   ! local variables
   !
   CHARACTER(13)             :: subname='transmittance'
   COMPLEX(dbl), ALLOCATABLE :: work(:,:), work1(:,:), work2(:,:)
   COMPLEX(dbl), ALLOCATABLE :: z(:,:), lambda(:,:)
   REAL(dbl),    ALLOCATABLE :: w(:)
   INTEGER :: i, j, ik, ierr, ie_buff
   !
   ! end of declarations
   !

!
!------------------------------
! main body
!------------------------------
!
   CALL timing(subname, OPR='start')
   CALL log_push(subname)

   ALLOCATE( work(dimC,dimC), STAT=ierr )
   IF (ierr/=0) CALL errore(subname,'allocating work, work2',ABS(ierr))
   !
   IF ( TRIM(which_formula) == "generalized" ) THEN
       !
       ALLOCATE( lambda(dimC,dimC), STAT=ierr )
       IF (ierr/=0) CALL errore(subname,'allocating lambda',ABS(ierr))
       !
       ALLOCATE( work1(dimC,dimC), STAT=ierr )
       IF (ierr/=0) CALL errore(subname,'allocating work1',ABS(ierr))
       !
   ENDIF
   !
   IF ( do_eigenchannels ) THEN
       !
       ALLOCATE( z(dimC,dimC), w(dimC), STAT=ierr )
       IF (ierr/=0) CALL errore(subname,'allocating z, w',ABS(ierr))
       !
       ALLOCATE( work2(dimC, dimC), STAT=ierr )
       IF (ierr/=0) CALL errore(subname,'allocating work2',ABS(ierr))
       !
       IF ( .NOT. ALLOCATED( work1 ) ) THEN
           ALLOCATE( work1(dimC, dimC), STAT=ierr )
           IF (ierr/=0) CALL errore(subname,'allocating work1',ABS(ierr))
       ENDIF
       !
   ENDIF


   !
   ! if which_formula = "generalized"
   ! calculates the correction term (lambda)
   !
   ! lambda = (gamma_R + gamma_L +2*eta)^{-1} * ( g_corr + gamma_R + gamma_L + 2*eta )
   !         = I + (gamma_R + gamma_L +2*eta)^{-1} * ( g_corr )
   !
   ! where g_corr = i (sgm_corr - sgm_corr^\dag)
   ! 
   !
   ik       = opr00%ik
   ie_buff  = opr00%ie_buff
   !
   IF ( TRIM(which_formula) == "generalized" )  THEN
       !
       !
       DO j=1,dimC
           !
           DO i=1,dimC
               lambda(i,j) =  CI * ( opr00%sgm(i,j,ik, ie_buff) - CONJG(opr00%sgm(j,i,ik, ie_buff))  )
               work(i,j) =  gamma_L(i,j) + gamma_R(i,j) 
           ENDDO
           !
           work(j,j) = work(j,j) + 2*EPS_m5
           !
       ENDDO
       !
       CALL mat_sv(dimC, dimC, work, lambda)

       !
       ! adding the identity matrix
       !
       DO i=1,dimC
           !
           lambda(i,i) = lambda(i,i) + CONE
           !
       ENDDO
       !
   ENDIF


!
! calculates the matrix product 
! whose trace (CONDUCT) is the main term of the transmittance 
! units of 2e^2/h for spin degenerate, or e^2/h for spin-polarized cases.
! 

   !
   ! WORK  = gamma_L * G_ret
   !
   CALL mat_mul(work, gamma_L, 'N', G_ret, 'N', dimC, dimC, dimC)

   !
   ! WORK2 = G_adv * gamma_L * G_ret
   ! this array will be stored to be used to compute eigenchannels
   !
   IF ( do_eigenchannels ) THEN
       !
       CALL mat_mul(work2, G_ret, 'C', work, 'N', dimC, dimC, dimC)
       !
   ELSE
       !
       CALL mat_mul(work, G_ret, 'C', work, 'N', dimC, dimC, dimC)
       !
   ENDIF

   !
   ! NOTE: G_ret is no longer used from here on, 
   !       in principles it would be possible to save memory by 
   !       using G_ret as workspace
   !


   !
   ! If we are not interested in the eigenchannels analysis
   ! we just take the diagonal matrix elements, otherwise
   ! we diagonalize and take the eigenvalues.
   !
   IF ( .NOT. do_eigenchannels ) THEN
       
       !
       ! WORK  = G_adv * gamma_L * G_ret * gamma_R
       !
       !CALL mat_mul(work, work2, 'N', gamma_R, 'N', dimC, dimC, dimC)
       CALL mat_mul(work, work, 'N', gamma_R, 'N', dimC, dimC, dimC)

       !
       ! WORK1 = G_adv * gamma_L * G_ret * gamma_R * Lambda
       ! This is calculated only if needed
       !
       IF ( TRIM(which_formula) ==  'generalized' ) THEN
           !
           CALL mat_mul(work1, work, 'N', lambda, 'N', dimC, dimC, dimC)
           !
           work = work1
           !
       ENDIF

       ! 
       ! select the data to compute the trace
       !
       DO i=1,dimC
           conduct(i) = REAL( work(i,i), dbl )
       ENDDO
       

   ELSE IF ( do_eigenchannels .AND. .NOT. do_eigplot ) THEN
       !
       ! Here we compute only the eigenchannel decomposition of
       ! the transmittance and diagonalize the product
       !
       ! work1 = gamma_R^1/2 * G_adv * gamma_L * G_ret * gamma_R^1/2
       !       = gamma_R^1/2 * work2 * gamma_R^1/2
       !
       ! which is a hermitean matrix.
       !
       ! To do this, we need to compute gamma_R^1/2  ( stored in work )
       !
       IF ( opr00%lhave_ovp ) THEN
           CALL mat_hdiag( z, w, gamma_R, opr00%S(:,:,ik), dimC ) 
       ELSE
           CALL mat_hdiag( z, w, gamma_R, dimC ) 
       ENDIF
       !
       !
       DO i = 1, dimC
           !
           IF ( w(i) < -EPS_m6 ) &
                 CALL errore(subname,'gamma_R not positive defined', 10)
           !
           ! clean any numerical noise leading to eigenvalues  -eps_m6 < w < 0.0
           IF ( w(i) < ZERO ) w(i) = ZERO
           !
           w(i) = SQRT( w(i) )
           !
       ENDDO
       !
       !
       DO j = 1, dimC
       DO i = 1, dimC
           !
           work1(i,j) = z(i,j) * w(j)
           !
       ENDDO
       ENDDO
       !
       CALL mat_mul( work, work1, 'N', z, 'C', dimC, dimC, dimC)
       
       !
       ! get the hermiteanized T matrix
       !
       CALL mat_mul( work1, work,  'N', work2, 'N', dimC, dimC, dimC)
       CALL mat_mul( work2, work1, 'N', work,  'N', dimC, dimC, dimC)
       
       !
       ! get the eigenvalues
       ! we invert the sign of work2 in such a way to have an increasing
       ! ordering of eigenvalues (according to lapack diag routine)
       !
       work2 = -work2
       !
       IF ( opr00%lhave_ovp ) THEN
           CALL mat_hdiag( z, conduct, work2, opr00%S(:,:,ik), dimC ) 
       ELSE
           CALL mat_hdiag( z, conduct, work2, dimC ) 
       ENDIF
       !
       conduct = -conduct
       

   ELSE IF ( do_eigenchannels .AND. do_eigplot ) THEN
       !
       ! here we follow the method given by: 
       !
       ! M Paulsson and M.Brandbyge, PRB 76, 115117 (2007)
       !
       ! This allows to compute data to plot the orbitals corresponding to
       ! the eigenchannels.
       !
       ! Once we have calculated A_L = G^adv Gamma_L G^ret  (already stored in work2)
       ! this matrix is diagonalized, and the sqrt is taken.
       !
       ! basically, we diagonalize   ( G^adv Gamma_L G^ret )^1/2  Gamma_R  ( G^adv Gamma_L G^ret )^1/2
       !
       !
       IF ( opr00%lhave_ovp ) THEN
           CALL mat_hdiag( z, w, work2, opr00%S(:,:,ik), dimC ) 
       ELSE
           CALL mat_hdiag( z, w, work2, dimC ) 
       ENDIF
       !
       ! define \bar{z} = z * diag(sqrt(w))
       ! Eq.(28) of the above paper
       !
       DO i = 1, dimC
           !
           IF ( w(i) < -EPS_m6 ) &
                 CALL errore(subname,'G^adv*Gamma_L*G^ret not positive defined', 10)
           !
           ! clean any numerical noise leading to eigenvalues  -eps_m6 < w < 0.0
           IF ( w(i) < ZERO ) w(i) = ZERO
           !
           w(i)   = SQRT( w(i) )
           !
       ENDDO
       !
       DO j = 1, dimC
       DO i = 1, dimC
           !
           work1(i,j) = z(i,j) * w(j)
           !
       ENDDO
       ENDDO
       !
       ! build  ( G^adv*Gamma_L*G^ret )^1/2
       CALL mat_mul( work, work1, 'N', z, 'C', dimC, dimC, dimC)
       
       !
       ! define the matrix to diagonalize
       !
       CALL mat_mul( work1, gamma_R, 'N',  work, 'N', dimC, dimC, dimC)
       CALL mat_mul( work2, work,    'C',  work1, 'N', dimC, dimC, dimC)
       
       !
       ! get the eigenvalues
       ! we invert the sign of work2 in such a way to have an increasing
       ! ordering of eigenvalues (according to lapack diag routine)
       !
       work2 = -work2
       !
       IF ( opr00%lhave_ovp ) THEN
           CALL mat_hdiag( z, conduct, work2, opr00%S(:,:,ik), dimC ) 
       ELSE
           CALL mat_hdiag( z, conduct, work2, dimC ) 
       ENDIF
       !
       z_eigplot(:,:) = z(:,:)
       !
       conduct = -conduct
       
   ELSE
       CALL errore(subname,'Unepxected error do_eigenchannels--do_eigplot',10)
   ENDIF

   !
   ! local memopry clean
   !
   IF ( ALLOCATED( work  ) ) THEN
       DEALLOCATE( work , STAT=ierr )
       IF (ierr/=0) CALL errore(subname,'deallocating work ',ABS(ierr))
   ENDIF
   IF ( ALLOCATED( work1 ) ) THEN
       DEALLOCATE( work1, STAT=ierr )
       IF (ierr/=0) CALL errore(subname,'deallocating work1',ABS(ierr))
   ENDIF
   IF ( ALLOCATED( work2 ) ) THEN
       DEALLOCATE( work2, STAT=ierr )
       IF (ierr/=0) CALL errore(subname,'deallocating work2',ABS(ierr))
   ENDIF
   IF ( ALLOCATED( lambda ) ) THEN
       DEALLOCATE( lambda, STAT=ierr )
       IF (ierr/=0) CALL errore(subname,'deallocating lambda',ABS(ierr))
   ENDIF
   !      
   IF ( do_eigenchannels ) THEN
       !
       DEALLOCATE( z, w, STAT=ierr )
       IF (ierr/=0) CALL errore(subname,'deallocating z, w',ABS(ierr))
       !
   ENDIF

   CALL timing(subname, OPR='stop')
   CALL log_pop(subname)
   !
   RETURN
   !
END SUBROUTINE transmittance

