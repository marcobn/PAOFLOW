!
!      Copyright (C) 2004 WanT Group, 2017 ERMES Group
!      Copyright (C) 1999 Marco Buongiorno Nardelli
!
!      This file is distributed under the terms of the
!      GNU General Public License. See the file `License'
!      in the root directory of the present distribution,
!      or http://www.gnu.org/copyleft/gpl.txt .
!
!***********************************************************************
   SUBROUTINE transfer_mtrx( ndim, opr00, opr01, ldt, tot, tott, niter )
   !***********************************************************************
   !
   !...  Iterative construction of the transfer matrix
   !     as Lopez-Sancho and Rubio, J.Phys.F:Met.Phys., v.14, 1205 (1984)
   !     and ibid. v.15, 851 (1985)
   !
   USE kinds
   USE io_global_module,        ONLY : stdout
   USE constants,               ONLY : CZERO, CONE, ZERO, EPS_m7, CI
   USE timing_module,           ONLY : timing
   USE log_module,              ONLY : log_push, log_pop
   USE util_module,             ONLY : mat_mul, mat_inv
   USE T_smearing_module,       ONLY : delta
   USE T_control_module,        ONLY : nfail, nfailx, niterx, transfer_thr
   USE T_operator_blc_module
   !
   IMPLICIT NONE


      !
      ! I/O variables
      !
      INTEGER,                 INTENT(IN)    :: ndim, ldt
      TYPE(operator_blc),      INTENT(IN)    :: opr00, opr01
      INTEGER,                 INTENT(OUT)   :: niter
      COMPLEX(dbl),            INTENT(OUT)   :: tot(ldt,ndim)
      COMPLEX(dbl),            INTENT(OUT)   :: tott(ldt,ndim)



      !
      ! local variables
      !
      CHARACTER(13) :: subname='transfer_mtrx'
      INTEGER       :: ik, i, j, m, ierr
      REAL(dbl)     :: conver, conver2
      LOGICAL       :: lconverged
      COMPLEX(dbl), ALLOCATABLE :: tau(:,:,:)
      COMPLEX(dbl), ALLOCATABLE :: taut(:,:,:)
      COMPLEX(dbl), ALLOCATABLE :: tsum(:,:)
      COMPLEX(dbl), ALLOCATABLE :: tsumt(:,:)
      COMPLEX(dbl), ALLOCATABLE :: t11(:,:), t12(:,:)
      COMPLEX(dbl), ALLOCATABLE :: s1(:,:), s2(:,:)

!
!----------------------------------------
! main Body
!----------------------------------------
!
      CALL timing(subname,OPR='start')
      CALL log_push(subname)

      IF ( .NOT. opr00%alloc )   CALL errore(subname,'opr00 not alloc',1)
      IF ( .NOT. opr01%alloc )   CALL errore(subname,'opr01 not alloc',1)

      IF ( ldt < ndim )          CALL errore(subname,'invalid ldt',1)

      ALLOCATE( tau(ndim, ndim, 2), taut(ndim, ndim, 2), STAT=ierr)
      IF (ierr/=0) CALL errore(subname,'allocating tau, taut',ABS(ierr))
      ALLOCATE( tsum(ndim, ndim), tsumt(ndim, ndim), STAT=ierr)
      IF (ierr/=0) CALL errore(subname,'allocating tsum, tsumt',ABS(ierr))
      ALLOCATE( t11(ndim, ndim), t12(ndim, ndim), STAT=ierr)
      IF (ierr/=0) CALL errore(subname,'allocating t11, t12',ABS(ierr))
      ALLOCATE( s1(ndim, ndim), s2(ndim, ndim), STAT=ierr)
      IF (ierr/=0) CALL errore(subname,'allocating s1, s2',ABS(ierr))

      !
      ! Construction of the transfer matrix
      !

      !
      ! Compute (ene * s00 - h00)^-1 and store it in t11 
      ! here opr00%aux = ene * s00 -h00
      !
      ik = opr00%ik
      !
      t12(:,:) = opr00%aux(:,:) +CI*delta*opr00%S(:,:,ik) 
      !
      CALL mat_inv( ndim, t12, t11 )

      !
      ! Compute intermediate t-matrices (defined as tau(ndim,ndim,niter)
      ! and taut(...))

! 
! as we do above, 
! we should take into account also possible contributions of 
! CI*delta due to S01
!

      CALL mat_mul( tau(:,:,1),  t11, 'N', opr01%aux, 'C', ndim, ndim, ndim)
      CALL mat_mul( taut(:,:,1), t11, 'N', opr01%aux, 'N', ndim, ndim, ndim)

      !
      ! Initialize T
      !
      tot ( 1:ndim, 1:ndim) = tau ( :, :, 1)
      tsum( :, :) = taut( :, :, 1)

      !
      ! Initialize T^bar
      !
      tott( 1:ndim, 1:ndim) = taut(:,:,1)
      tsumt( :, :) = tau(:,:,1)


      !
      ! Main loop
      !
      lconverged = .FALSE.

      convergence_loop: &
      DO m = 1, niterx

          CALL mat_mul(t11, tau(:,:,1), 'N', taut(:,:,1), 'N', ndim, ndim, ndim)
          CALL mat_mul(t12, taut(:,:,1), 'N', tau(:,:,1), 'N', ndim, ndim, ndim)  

          s1(:,:) = -( t11(:,:) + t12(:,:) )
          !
          DO i=1,ndim
              s1(i,i) = CONE + s1(i,i)
          ENDDO
          !
          CALL mat_inv( ndim, s1, s2, IERR=ierr)

          !
          ! exit the main loop, 
          ! set all the matrices to be computed to zero
          ! and print a warning
          !
          IF ( ierr/=0 ) THEN
              !
              tot  = CZERO
              tott = CZERO
              !
              WRITE(stdout, "(2x, 'WARNING: singular matrix at iteration', i4)" ) m
              WRITE(stdout, "(2x, '         energy descarted')" )
              !
              lconverged = .FALSE.
              EXIT convergence_loop
              !
          ENDIF


          CALL mat_mul( t11, tau(:,:,1),  'N', tau(:,:,1),  'N', ndim, ndim, ndim)
          CALL mat_mul( t12, taut(:,:,1), 'N', taut(:,:,1), 'N', ndim, ndim, ndim) 
          CALL mat_mul( tau(:,:,2),  s2,  'N', t11, 'N', ndim, ndim, ndim)
          CALL mat_mul( taut(:,:,2), s2,  'N', t12, 'N', ndim, ndim, ndim)

          !
          ! Put the transfer matrices together
          !
          CALL mat_mul( t11, tsum, 'N', tau(:,:,2),  'N', ndim, ndim, ndim)
          CALL mat_mul( s1,  tsum, 'N', taut(:,:,2), 'N', ndim, ndim, ndim)
  
          tot( 1:ndim, 1:ndim ) = tot( 1:ndim, 1:ndim) + t11
          tsum = s1


          CALL mat_mul(t11, tsumt, 'N', taut(:,:,2), 'N', ndim, ndim, ndim)
          CALL mat_mul(s1,  tsumt, 'N', tau(:,:,2),  'N', ndim, ndim, ndim)

          tott(1:ndim, 1:ndim)  = tott(1:ndim,1:ndim) + t11(:,:)
          tsumt = s1
          !
          tau(:,:,1)  = tau(:,:,2)
          taut(:,:,1) = taut(:,:,2)


          !
          ! Convergence chech on the t-matrices
          !
          conver  = ZERO
          conver2 = ZERO

          DO j = 1, ndim
          DO i = 1, ndim
              !
              conver  =  conver + REAL( tau(i,j,2)  * CONJG( tau(i,j,2) )) 
              conver2 = conver2 + REAL( taut(i,j,2) * CONJG( taut(i,j,2) )) 
              !
          ENDDO
          ENDDO
          !
          IF ( conver < transfer_thr .AND. conver2 < transfer_thr ) THEN 
              lconverged = .TRUE.
              EXIT
          ENDIF
          !
          niter = m
          !
      ENDDO convergence_loop

      ! 
      ! if not converged, print a WARNING but do not crash
      ! 
      IF ( .NOT. lconverged ) THEN 
          !
          ! de-comment here if you want to allow for a hard crash
          !
          !CALL errore(subname, 'bad t-matrix convergence', 10 )
          
          !
          ! this is instead a soft way to stop the code if
          ! the failing message is printed more that 
          !
          nfail = nfail + 1
          !
          tot  = CZERO
          tott = CZERO
          !
          WRITE(stdout, "(2x, 'WARNING: t-matrix not converged', i4)" ) niterx
          WRITE(stdout, "(2x, '         energy descarted')" )
          !
      ENDIF
      !
      IF ( nfail > nfailx ) CALL errore(subname,'too many failures in t-matrix conv',71)

!
! ... local cleaning
!
      DEALLOCATE( tau, taut, STAT=ierr)
      IF (ierr/=0) CALL errore(subname,'deallocating tau, taut',ABS(ierr))
      DEALLOCATE( tsum, tsumt, STAT=ierr)
      IF (ierr/=0) CALL errore(subname,'deallocating tsum, tsumt',ABS(ierr))
      DEALLOCATE( t11, t12, STAT=ierr)
      IF (ierr/=0) CALL errore(subname,'deallocating t11, t12',ABS(ierr))
      DEALLOCATE( s1, s2, STAT=ierr)
      IF (ierr/=0) CALL errore(subname,'deallocating s1, s2',ABS(ierr))

      CALL timing(subname,OPR='stop')
      CALL log_pop(subname)

   END SUBROUTINE transfer_mtrx

