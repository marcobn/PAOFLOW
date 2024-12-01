!
! Copyright (C) 2004 WanT Group, 2017 ERMES Group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
! Based on a previous version by D. Vanderbilt, N. Marzari and I. Souza
! See the file README in the root directory for a full list of credits
!
!*********************************************
   MODULE sph_har_module
!*********************************************
   USE constants, ONLY : EPS_m9, PI, TPI, ZERO
   USE kinds, ONLY: dbl
   IMPLICIT NONE
   PRIVATE
!
! This module contains the routine which computes the
! real spherical harmonics needed for the trial center projections
!
! routines in this module:
! SUBROUTINE sph_har_setup(ndim, g, gg, ndir, l, m, ylm)
! SUBROUTINE sph_har_index(lmax, index)
!

   !
   ! ... constants for the normalization of spherical harmonics
   !
   REAL(dbl), SAVE :: sph00  
   !
   REAL(dbl), SAVE :: sph1m1
   REAL(dbl), SAVE :: sph10
   REAL(dbl), SAVE :: sph11
   !
   REAL(dbl), SAVE :: sph2m2
   REAL(dbl), SAVE :: sph2m1
   REAL(dbl), SAVE :: sph20
   REAL(dbl), SAVE :: sph21
   REAL(dbl), SAVE :: sph22

!
! end of declarations
!
   PUBLIC :: sph_har_setup
   PUBLIC :: sph_har_index

CONTAINS

!****************************************************
   SUBROUTINE sph_har_index( lmax, indx)
   !****************************************************
   IMPLICIT NONE
     !
     ! this routine computes the map between the 
     ! usual (l,m) notation used for spherical harmonics
     ! and the internal ordering used in the ylmr2 routine
     ! (from espresso package, based on the Numerical Recipies
     ! recursive algorithm)
     !
     ! i = index(m,l)    ! note the reversal of the indexes
     !
     ! where i is the index in the array of Y_lm produced by
     ! ylmr2 where the order is [ Y_lm is described as (l,m) ]
     !
     ! (0,0)  /
     ! (1,0)  (1,1)  (1,-1)  /
     ! (2,0)  (2,1)  (2,-1)  (2,2)  (2,-2) /
     ! ...
     !
     INTEGER, INTENT(in)   :: lmax  ! the number of angular mom
                                    ! channel (0 -> l = lmax)
     INTEGER, INTENT(out)  :: indx(-lmax:lmax, 0:lmax)
     !
     ! local variables
     !
     INTEGER :: il, im, icount
     !
     indx = 0

     icount=1
     indx(0,0)= icount
     !
     DO il=1,lmax
        icount = icount + 1
        indx(0,il) = icount

        DO im = 1, il
            !
            icount = icount +1
            indx(im,il) = icount
            !
            icount = icount +1
            indx(-im,il) = icount
            !
        ENDDO
     ENDDO

     RETURN
  END SUBROUTINE sph_har_index


!****************************************************
   SUBROUTINE sph_har_setup( ndim, g, gg, ndir, l, m, ylm)
   !****************************************************
   IMPLICIT NONE
     !
     ! compute the real spherical harmonics for the g(3,dim)
     ! vectors and store them in ylm. l, m self-explaining, ndir
     ! is the direction of the polar axis.
     !
     ! the m parameter is defined as the opposite of the one
     ! used in ylmr2
     !
     ! l == -1 gives the sp^3 hybrid sph_arm.
     !

     INTEGER,        INTENT(in) :: ndim
     INTEGER,        INTENT(in) :: ndir, l, m
     REAL(dbl),      INTENT(in) :: g(3,ndim), gg(ndim)
     REAL(dbl),     INTENT(out) :: ylm(ndim)

     ! ... local variables
     CHARACTER(13)  :: subname='sph_har_setup'
     INTEGER   :: ig
     REAL(dbl) :: dist_pl, dist_cos
     REAL(dbl) :: th_cos, th_sin
     REAL(dbl) :: ph_cos, ph_sin

     sph00  = 1.0_dbl/SQRT( 2.0_dbl * TPI )
     !
     sph1m1 = SQRT(  1.5_dbl / TPI )
     sph10  = SQRT(  1.5_dbl / TPI )
     sph11  = SQRT(  1.5_dbl / TPI )
     ! 
     sph2m2 = SQRT( 15.0_dbl / 8.0_dbl / TPI )
     sph2m1 = SQRT( 15.0_dbl / 2.0_dbl / TPI )
     sph20  = SQRT(  5.0_dbl / 8.0_dbl / TPI )
     sph21  = SQRT( 15.0_dbl / 2.0_dbl / TPI )
     sph22  = SQRT( 15.0_dbl / 8.0_dbl / TPI )
     !
     dist_pl  = ZERO
     dist_cos = ZERO
   
     DO ig =1, ndim
         !
         ! set the polar (z) direction 
         !
         SELECT CASE ( ndir )
         CASE ( 3 )
             dist_pl  = SQRT( g(1,ig)**2 + g(2,ig)**2 )
             dist_cos = g(3,ig)
         CASE ( 2 )
             dist_pl  = SQRT( g(1,ig)**2 + g(3,ig)**2 )
             dist_cos = g(2,ig)
         CASE ( 1 )
             dist_pl  = SQRT( g(2,ig)**2 + g(3,ig)**2 )
             dist_cos = g(1,ig)
         CASE DEFAULT
             CALL errore(subname, 'wrong z-direction ', ABS(ndir)+1 )
         END SELECT

         !
         ! ... IF  rpos is on the origin, or on the z axis, I give arbitrary
         !     values to cos/sin of theta, or of phi, respectively
         !
         IF ( ABS( gg(ig) ) <= EPS_m9 ) THEN
            th_cos = ZERO
            th_sin = ZERO        ! this should be ONE for coherence with ylmr2
         ELSE
            th_cos = dist_cos / gg(ig)
            th_sin = dist_pl / gg(ig)
         ENDIF
    
         IF (ABS( dist_pl ) <= EPS_m9 ) THEN
            ph_cos = ZERO
            ph_sin = ZERO
         ELSE
            IF ( ndir == 3 ) THEN
               ph_cos = g(1,ig) / dist_pl
               ph_sin = g(2,ig) / dist_pl
            ELSE IF ( ndir == 2 ) THEN
               ph_cos = g(3,ig) / dist_pl
               ph_sin = g(1,ig) / dist_pl
            ELSE
               ph_cos = g(2,ig) / dist_pl
               ph_sin = g(3,ig) / dist_pl
            ENDIF
         ENDIF
    
         ! 
         ! select the L main quantum number
         ! 
         IF ( l == 2 ) THEN
    
            IF ( m == -2 ) THEN
              ylm(ig) = sph2m2 * ( th_sin**2 ) * ( ph_cos**2 - ph_sin**2 )
            ELSE IF ( m == -1 ) THEN
              ylm(ig) = -sph2m1 * th_sin * th_cos * ph_cos
            ELSE IF ( m == 0 ) THEN
              ylm(ig) = sph20 * ( 3.0_dbl * th_cos**2 - 1.0_dbl )
            ELSE IF ( m == 1 ) THEN
              ylm(ig) = -sph21 * th_sin * th_cos * ph_sin
            ELSE IF ( m == 2 ) THEN
              ylm(ig) = sph22 * ( th_sin**2 ) * 2.0_dbl * ph_sin * ph_cos
            ELSE
              CALL errore(subname, ' invalid m for L=2 ', ABS(m) )
            END IF
    
         ELSE IF ( l == 1 ) THEN
    
            IF ( m == -1 ) THEN
              ylm(ig) = -sph1m1 * th_sin * ph_cos
            ELSE IF ( m == 0 ) THEN
              ylm(ig) = sph10 * th_cos
            ELSE IF ( m == 1 ) THEN
              ylm(ig) = -sph11 * th_sin * ph_sin
            ELSE
              CALL errore(subname, ' invalid m for L=1 ', ABS(m) )
            END IF
    
         ELSE IF ( l == 0 ) THEN
    
            ylm(ig) = sph00 
      
         ELSE IF ( l == -1 ) THEN
    
            !
            ! ...  sp^3 orbitals
            ! 
            IF ( m == 1 ) THEN
                !
                ! ... sp^3 along 111 direction IF  ndir=3
                !
                ylm(ig) = ( sph00 + sph1m1 * th_sin * ph_cos +        &
                       sph11 * th_sin * ph_sin + sph10 * th_cos ) / 2.0_dbl
            ELSE IF ( m == 2 ) THEN
                !
                ! ... sp^3 along 1,-1,-1 direction IF  ndir=3
                !
                ylm(ig) = ( sph00 + sph1m1 * th_sin * ph_cos -        &
                       sph11 * th_sin * ph_sin - sph10 * th_cos ) / 2.0_dbl
            ELSE IF ( m == 3 ) THEN
                !
                ! ...  sp^3 along -1,1,-1 direction IF  ndir=3
                !
                ylm(ig) = ( sph00 - sph1m1 * th_sin * ph_cos +        &
                       sph11 * th_sin * ph_sin - sph10 * th_cos ) / 2.0_dbl
            ELSE IF ( m == 4 ) THEN
                !
                ! ... sp^3 along -1,-1,1 direction IF  ndir=3
                !
                ylm(ig) = ( sph00 - sph1m1 * th_sin * ph_cos -        &
                       sph11 * th_sin * ph_sin + sph10 * th_cos ) / 2.0_dbl
            ELSE IF ( m == -1 ) THEN
                !
                ! ...  sp^3 along -1,-1,-1 direction IF  ndir=3
                !
                ylm(ig) = ( sph00 - sph1m1 * th_sin * ph_cos -        &
                       sph11 * th_sin * ph_sin - sph10 * th_cos ) / 2.0_dbl
            ELSE IF ( m == -2 ) THEN
                !
                ! ...  sp^3 along -1,1,1 direction IF  ndir=3
                !
                ylm(ig) = ( sph00 - sph1m1 * th_sin * ph_cos +        &
                       sph11 * th_sin * ph_sin + sph10 * th_cos ) / 2.0_dbl
            ELSE IF ( m == -3 ) THEN
                !
                ! ...  sp^3 along 1,-1,1 direction IF  ndir=3
                !
                ylm(ig) = ( sph00 + sph1m1 * th_sin * ph_cos -        &
                       sph11 * th_sin * ph_sin + sph10 * th_cos ) / 2.0_dbl
            ELSE IF ( m == -4 ) THEN
                !
                ! ...  sp^3 along 1,1,-1 direction IF  ndir=3
                !
                ylm(ig) = ( sph00 + sph1m1 * th_sin * ph_cos +        &
                       sph11 * th_sin * ph_sin - sph10 * th_cos ) / 2.0_dbl
            ELSE
                CALL errore(subname,'invalid m for sp^3 hybrid sph_harm ', ABS(m)+100 )
            ENDIF
        ELSE
            CALL errore(subname, ' Invalid L quantum number ', m )
        ENDIF

    ENDDO

    RETURN
  END SUBROUTINE sph_har_setup

  !
END MODULE sph_har_module

