!
!      Copyright (C) 2008 WanT Group, 2017 ERMES Group
!
!      This file is distributed under the terms of the
!      GNU General Public License. See the file `License'
!      in the root directory of the present distribution,
!      or http://www.gnu.org/copyleft/gpl.txt .
!
!*******************************************************************
   SUBROUTINE hamiltonian_setup( ik, ie_g, ie_buff )
   !*******************************************************************
   !
   ! For each block (00C, 00L, 00R, 01L, 01R, CR, LC) define
   ! the aux quantity:
   !
   !  aux = E*ovp -ham -sgm_corr
   !
   ! for a given kpt
   !
   !
   USE kinds,                ONLY : dbl
   USE T_hamiltonian_module, ONLY : shift_L, shift_C, shift_R,             &
                                    blc_00L, blc_01L, blc_00R, blc_01R,    &
                                    blc_00C, blc_LC,  blc_CR
   USE T_correlation_module, ONLY : shift_C_corr
   USE T_egrid_module,       ONLY : egrid
   USE T_operator_blc_module
   !
   USE timing_module,        ONLY : timing
   USE log_module,           ONLY : log_push, log_pop
   !
   IMPLICIT NONE

   !
   ! local variables
   !
   INTEGER,            INTENT(IN) :: ie_g, ik
   INTEGER, OPTIONAL,  INTENT(IN) :: ie_buff

   !
   ! local variables
   !
   CHARACTER(17) :: subname="hamiltonian_setup"
   REAL(dbl)     :: omg
   INTEGER       :: i,j,dim1,dim2
   INTEGER       :: ie_bl

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

   omg = egrid(ie_g)

   ie_bl = 1
   IF ( PRESENT (ie_buff) ) ie_bl = ie_buff

   !
   ! hamiltonian and overlap
   !
   IF ( blc_00L%alloc ) THEN
       !blc_00L%aux(:,:)  =  (omg -shift_L) * blc_00L%S(:,:,ik) -blc_00L%H(:,:,ik)
       DO j = 1,blc_00L%dim2
       DO i = 1,blc_00L%dim1
           blc_00L%aux(i,j)  =  (omg -shift_L) * blc_00L%S(i,j,ik)-blc_00L%H(i,j,ik)
       ENDDO
       ENDDO
       !
   ENDIF
   !
   IF ( blc_01L%alloc ) THEN
      !blc_01L%aux(:,:)  =  (omg -shift_L) * blc_01L%S(:,:,ik) -blc_01L%H(:,:,ik)
       DO j = 1,blc_01L%dim2
       DO i = 1,blc_01L%dim1
           blc_01L%aux(i,j)  =  (omg -shift_L) * blc_01L%S(i,j,ik)-blc_01L%H(i,j,ik)
       ENDDO
       ENDDO
       !
   ENDIF
   !
   IF ( blc_00R%alloc ) THEN
       !blc_00R%aux(:,:)  =  (omg -shift_R) * blc_00R%S(:,:,ik) -blc_00R%H(:,:,ik)
       DO j = 1,blc_00R%dim2
       DO i = 1,blc_00R%dim1
           blc_00R%aux(i,j)  =  (omg -shift_R) *blc_00R%S(i,j,ik)-blc_00R%H(i,j,ik)
       ENDDO
       ENDDO
       !
   ENDIF
   !
   IF ( blc_01R%alloc ) THEN
       !blc_01R%aux(:,:)  =  (omg -shift_R) * blc_01R%S(:,:,ik) -blc_01R%H(:,:,ik)
       DO j = 1,blc_01R%dim2
       DO i = 1,blc_01R%dim1
           blc_01R%aux(i,j)  =  (omg -shift_R) *blc_01R%S(i,j,ik)-blc_01R%H(i,j,ik)
       ENDDO
       ENDDO
       !
   ENDIF
   ! 
   IF ( blc_00C%alloc ) THEN
       !blc_00C%aux(:,:)  =  (omg -shift_C) * blc_00C%S(:,:,ik) -blc_00C%H(:,:,ik)
       DO j = 1,blc_00C%dim2 
       DO i = 1,blc_00C%dim1
           blc_00C%aux(i,j)  =  (omg -shift_C) * blc_00C%S(i,j,ik) -blc_00C%H(i,j,ik)
       ENDDO
       ENDDO
       !
   ENDIF
   !
   IF ( blc_LC%alloc ) THEN
       !blc_LC%aux(:,:)   =  (omg -shift_C) * blc_LC%S(:,:,ik) -blc_LC%H(:,:,ik)
       DO j = 1,blc_LC%dim2
       DO i = 1,blc_LC%dim1
           blc_LC%aux(i,j)  =  (omg -shift_C) *blc_LC%S(i,j,ik)-blc_LC%H(i,j,ik)
       ENDDO
       ENDDO
       !
   ENDIF
   !
   IF ( blc_CR%alloc ) THEN
       !blc_CR%aux(:,:)   =  (omg -shift_C) * blc_CR%S(:,:,ik) -blc_CR%H(:,:,ik)
       DO j = 1,blc_CR%dim2
       DO i = 1,blc_CR%dim1
           blc_CR%aux(i,j)  =  (omg -shift_C) *blc_CR%S(i,j,ik)-blc_CR%H(i,j,ik)
       ENDDO
       ENDDO
       !
   ENDIF

   !
   ! correlation
   !
   IF ( blc_00L%alloc .AND. ASSOCIATED( blc_00L%sgm ) ) THEN
       blc_00L%aux(:,:) = blc_00L%aux(:,:) -blc_00L%sgm(:,:,ik, ie_bl)
   ENDIF
   IF ( blc_01L%alloc .AND. ASSOCIATED( blc_01L%sgm ) ) THEN
       blc_01L%aux(:,:) = blc_01L%aux(:,:) -blc_01L%sgm(:,:,ik, ie_bl)
   ENDIF
   !
   IF ( blc_00R%alloc .AND. ASSOCIATED( blc_00R%sgm ) ) THEN
       blc_00R%aux(:,:) = blc_00R%aux(:,:) -blc_00R%sgm(:,:,ik, ie_bl)
   ENDIF
   IF ( blc_01R%alloc .AND. ASSOCIATED( blc_01R%sgm ) ) THEN
       blc_01R%aux(:,:) = blc_01R%aux(:,:) -blc_01R%sgm(:,:,ik, ie_bl)
   ENDIF
   !
   IF ( blc_00C%alloc .AND. ASSOCIATED( blc_00C%sgm ) ) THEN
       blc_00C%aux(:,:) = blc_00C%aux(:,:) -blc_00C%sgm(:,:,ik, ie_bl) &
                                           -shift_C_corr * blc_00C%S(:,:,ik)
   ENDIF
   IF ( blc_LC%alloc .AND. ASSOCIATED( blc_LC%sgm ) ) THEN
       blc_LC%aux(:,:)  = blc_LC%aux(:,:)  -blc_LC%sgm(:,:,ik, ie_bl) &
                                           -shift_C_corr * blc_LC%S(:,:,ik)
   ENDIF
   IF ( blc_CR%alloc .AND. ASSOCIATED( blc_CR%sgm ) ) THEN
       blc_CR%aux(:,:)  = blc_CR%aux(:,:)  -blc_CR%sgm(:,:,ik, ie_bl) &
                                           -shift_C_corr * blc_CR%S(:,:,ik)
   ENDIF


   !
   ! finalize setup
   !
   IF ( blc_00C%alloc ) THEN
       CALL operator_blc_update( IE=ie_g, IK=ik, IE_BUFF=ie_bl, OBJ=blc_00C )
   ENDIF
   IF ( blc_LC%alloc ) THEN
       CALL operator_blc_update( IE=ie_g, IK=ik, IE_BUFF=ie_bl, OBJ=blc_LC  )
   ENDIF
   IF ( blc_CR%alloc ) THEN
       CALL operator_blc_update( IE=ie_g, IK=ik, IE_BUFF=ie_bl, OBJ=blc_CR  )
   ENDIF
   !
   IF ( blc_00L%alloc ) THEN
       CALL operator_blc_update( IE=ie_g, IK=ik, IE_BUFF=ie_bl, OBJ=blc_00L )
   ENDIF
   IF ( blc_01L%alloc ) THEN
       CALL operator_blc_update( IE=ie_g, IK=ik, IE_BUFF=ie_bl, OBJ=blc_01L )
   ENDIF
   !
   IF ( blc_00R%alloc ) THEN
       CALL operator_blc_update( IE=ie_g, IK=ik, IE_BUFF=ie_bl, OBJ=blc_00R )
   ENDIF
   IF ( blc_01R%alloc ) THEN
       CALL operator_blc_update( IE=ie_g, IK=ik, IE_BUFF=ie_bl, OBJ=blc_01R )
   ENDIF
   
   CALL timing( subname, OPR='STOP' )
   CALL log_pop( subname )
   !
   RETURN
   !
END SUBROUTINE hamiltonian_setup

