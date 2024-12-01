! 
! Copyright (C) 2005 WanT Group, 2017 ERMES Group
! 
! This file is distributed under the terms of the 
! GNU General Public License. See the file `License' 
! in the root directory of the present distribution, 
! or http://www.gnu.org/copyleft/gpl.txt . 
! 
MODULE grids_module
!
CONTAINS
!
!*********************************************************
SUBROUTINE grids_get_rgrid(nr, nrtot, wr, ivr)
   !*********************************************************
   !
   ! Given the three generators nr(:), this subroutine defines
   ! a R-grid in real space. Output is in crystal coords
   !
   USE kinds
   USE constants,         ONLY : ZERO, ONE
   !
      IMPLICIT NONE

! <INFO>
!
! This subroutine generates the regular R-vector grid according to
! the input generators nr(1:3). The number nrtot of R vectors is 
! also given to output 
!

   INTEGER,              INTENT(IN)  :: nr(3)
   INTEGER,   OPTIONAL,  INTENT(OUT) :: nrtot
   REAL(dbl), OPTIONAL,  INTENT(OUT) :: wr(*)
   INTEGER,   OPTIONAL,  INTENT(OUT) :: ivr(3,*)

! </INFO>
! ... local variables

   CHARACTER(15)   :: subname="grids_get_rgrid"
   INTEGER         :: ir, ir2
   INTEGER         :: i,j,k, ierr
   INTEGER         :: nrx, nrtot_, counter
   LOGICAL         :: found
   !
   REAL(dbl), ALLOCATABLE :: wr_(:)
   INTEGER,   ALLOCATABLE :: ivr_(:,:)
  
!
! ... end of declarations
!-------------------------------------------------------------
!

   IF ( ANY(nr(:) <= 0 )  ) CALL errore(subname,'invalid nr',1)
   !
   ! temp workspace
   !
   nrx = PRODUCT( nr(:) ) * 2
   !
   ALLOCATE(  ivr_(3, nrx ), STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'allocating ivr_', ABS(ierr))
   ALLOCATE(  wr_( nrx ), STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'allocating wr_', ABS(ierr))
   !
   ivr_ = 0
   wr_  = ZERO

   !
   ! setup ivr in crystal coordinates
   !
   ir = 0

   DO k=1,nr(3)
   DO j=1,nr(2)
   DO i=1,nr(1)
       !
       ir = ir + 1
       ! 
       ivr_(1,ir) =  i -( nr(1) +1 ) / 2
       ivr_(2,ir) =  j -( nr(2) +1 ) / 2
       ivr_(3,ir) =  k -( nr(3) +1 ) / 2
       !
       ! the sum rule on the weights depends on the definition 
       ! of the kpt-weight sum
       !
       wr_( ir ) = ONE
       !
   ENDDO
   ENDDO
   ENDDO
   !
   nrtot_  = ir
   counter = ir
   !
   ! check if -R is always included
   !
   DO ir = 1, nrtot_
       !
       found=.FALSE.
       !
       inner_loop:&
       DO ir2 = 1, nrtot_
           !
           IF ( ALL( ivr_(:,ir2) == -ivr_(:,ir)) ) THEN
               found=.TRUE.
               EXIT inner_loop
           ENDIF
           !
       ENDDO inner_loop
       !
       IF ( .NOT. found ) THEN
           !
           counter=counter+1
           ivr_(:,counter) = -ivr_(:,ir)
           wr_(counter) = 0.5_dbl * wr_(ir)
           wr_(ir)      = 0.5_dbl * wr_(ir)
           !
       ENDIF
       !
   ENDDO
   !
   nrtot_ = counter
   !
   IF ( SUM(wr_) /= REAL( PRODUCT(nr) , dbl ) ) &
        CALL errore(subname,'invalid r-weight sum-rule',10)
   !
   IF ( PRESENT(nrtot) )     nrtot = nrtot_
   IF ( PRESENT(ivr) )       ivr(1:3, 1:nrtot_) = ivr_(1:3, 1:nrtot_)
   IF ( PRESENT(wr) )        wr(1:nrtot_) = wr_(1:nrtot_)
   ! 
   DEALLOCATE(  ivr_, wr_, STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'deallocating ivr_, wr_', ABS(ierr))
   !
   RETURN
   !
END SUBROUTINE grids_get_rgrid

END MODULE grids_module

