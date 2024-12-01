! 
! Copyright (C) 2004 WanT Group, 2017 ERMES Group
! 
! This file is distributed under the terms of the 
! GNU General Public License. See the file `License' 
! in the root directory of the present distribution, 
! or http://www.gnu.org/copyleft/gpl.txt . 
! 
!*********************************************************
   SUBROUTINE get_monkpack(nk,s,nkpts,vkpt,coordinate,bvec,ierr)
   !*********************************************************
   !
   USE kinds
   USE constants,         ONLY : EPS_m6, ZERO, ONE, TWO
   USE converters_module, ONLY : cart2cry
   !
   IMPLICIT NONE

! <INFO>
! This subroutine generates the nk(3) and s(3) parameters 
! of the Monkhorst-Pack grid if the input vkpt(3,nkpts) are
! generated using the MP algorithm (and IERR=0). 
! If not, IERR gives /= 0 and nk(3) = 0 and s(3) = 0.
!
! VKPT are supposed to be in the coordinates given in input by
! 'CARTESIAN' or 'CRYSTAL' strings.
! the units should be the same as in BVEC.
!
! NOTE: this soubroutine should be generalized to treat
!       cases where the s shift is non-integer and completely
!       general.
!

   INTEGER,  INTENT(out)     :: nk(3), s(3)
   INTEGER,  INTENT(in)      :: nkpts
   REAL(dbl),INTENT(in)      :: vkpt(3,nkpts)
   CHARACTER(*)              :: coordinate   
   REAL(dbl),INTENT(in)      :: bvec(3,3)
   INTEGER,  INTENT(out)     :: ierr

! </INFO>
! ... local variables

   CHARACTER(12)             :: subname="get_monkpack"
   REAL(dbl), ALLOCATABLE    :: kpt_loc(:,:), kpt_gen(:,:)
   LOGICAL                   :: found
   INTEGER                   :: ik,ierrl
   INTEGER                   :: i

!
! ... end of declarations
!-------------------------------------------------------------
!

   ! 
   ! set IERR to zero
   ierr = 0

   !
   ! first take the kpts to crystal coordinates (and computes moduli)
   ! assuming they are in cartesian ones
   !
   ALLOCATE(kpt_loc(3,nkpts), STAT=ierrl)
   IF (ierrl/=0) CALL errore(subname,'allocating kpt_loc',ABS(ierrl))
   !
   ALLOCATE(kpt_gen(3,nkpts), STAT=ierrl)
   IF (ierrl/=0) CALL errore(subname,'allocating kpt_gen',ABS(ierrl))
   !
   kpt_loc(:,:) = vkpt(:,:)

   SELECT CASE (TRIM(coordinate))
   CASE ( 'CARTESIAN', 'cartesian' )
       CALL cart2cry(kpt_loc, bvec)
   CASE ( 'CRYSTAL', 'crystal' )

   CASE DEFAULT
        CALL errore(subname,'invalid coordinate '//TRIM(coordinate),1)
   END SELECT


   ! 
   ! locate each component between (0,1] 
   ! verify if the zero component is present, eventually set s = 0.
   !
   s(:) = 1
   !
   DO ik=1,nkpts
       !
       DO i=1,3
           !
           kpt_loc(i,ik) = MOD( kpt_loc(i,ik), ONE )
           !
           IF ( kpt_loc (i,ik) < -EPS_m6 ) kpt_loc(i,ik) = kpt_loc(i,ik) + ONE
           !
           IF ( ABS( kpt_loc(i,ik) ) < EPS_m6 ) THEN 
                s(i) = 0
                kpt_loc(i,ik) = ONE
           ENDIF
           !
       ENDDO
       !
   ENDDO

   ! 
   ! shift the grid to the origin if needed,
   ! invert the components in order to search for the 
   ! maximum in each direction, and determine the nk value.
   !
   DO i=1,3
       !
       ! save kpt_loc for later user
       kpt_gen(i,:) = kpt_loc(i,:)
       !
       IF ( s(i) == 1 )  kpt_gen(i,:) = kpt_gen(i,:) - MINVAL(kpt_gen(i,:))
       !
       !
       DO ik=1,nkpts
            !
            IF ( ABS( kpt_gen(i,ik) ) > EPS_m6 ) THEN
               !
               kpt_gen(i,ik) = ONE / kpt_gen(i,ik)
               !
            ELSE
               !
               kpt_gen(i,ik) = ZERO
               !
            ENDIF
       ENDDO
       !
       nk(i) = MAXVAL( NINT( kpt_gen(i,:) ) )
       !
   ENDDO

   !
   ! generates the kpoints according to the parameters found
   ! anche check the consistency.
   !
   IF ( PRODUCT(nk) /= nkpts ) ierr = ABS(PRODUCT(nk))+1
        

   !
   ! the check is performed
   ! only if the mesh may be consistent with monkhorst-pack
   !
   IF ( ierr == 0 ) THEN

       kpt_gen(:,:) = ZERO
       CALL monkpack( nk, s, kpt_gen )

       !
       ! now take the indexes as before in the (0,1] interval
       ! and compare kpt by kpt
       !
       DO ik=1,nkpts
           DO i=1,3
               kpt_gen(i,ik) = MOD( kpt_gen(i,ik), ONE )
               IF ( kpt_gen(i,ik) < -EPS_m6 ) kpt_gen(i,ik) = kpt_gen(i,ik)+ONE
               IF ( ABS( kpt_gen(i,ik) ) < EPS_m6 ) kpt_gen(i,ik) = ONE
           ENDDO
       ENDDO

       !
       ! check the points
       generated: DO ik=1,nkpts
           found = .FALSE.
           input: DO i=1,nkpts
                IF ( (kpt_gen(1,ik)-kpt_loc(1,i))**2 + &
                     (kpt_gen(2,ik)-kpt_loc(2,i))**2 + &
                     (kpt_gen(3,ik)-kpt_loc(3,i))**2 < EPS_m6 ) THEN
                   found = .TRUE.
                   EXIT input
                ENDIF
            ENDDO input
            IF ( .NOT. found ) THEN  
                 ierr = ik
                 EXIT generated
            ENDIF
       ENDDO generated
          
   ENDIF

   IF (ierr /= 0) THEN
      nk(:) = 0
      s(:)  = 0
   ENDIF

   DEALLOCATE( kpt_loc, kpt_gen, STAT=ierrl)
     IF (ierrl/=0) CALL errore(subname,'deallocating kpt_*',ABS(ierrl))


END SUBROUTINE get_monkpack


