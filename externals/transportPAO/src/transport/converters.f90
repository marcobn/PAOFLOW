! 
! Copyright (C) 2004 WanT Group, 2017 ERMES Group
! 
! This file is distributed under the terms of the 
! GNU General Public License. See the file `License' 
! in the root directory of the present distribution, 
! or http://www.gnu.org/copyleft/gpl.txt . 
! 
! <INFO>
!*********************************************
   MODULE converters_module
!*********************************************
   !
   USE kinds,       ONLY : dbl
   USE constants,   ONLY : ZERO, EPS_m8
   USE util_module, ONLY : mat_inv
   !
   IMPLICIT NONE
   PRIVATE

! This module contains some utilities to convert 
! coordinates in different units.
! In the case of direct lattice vectors, BASIS contains
! the direct lattice basis columnwise, while for
! reciprocal lattice vectors BASIS is the reciprocal
! lattice basis matrix.
! 
! routines in this module:
! SUBROUTINE  cart2cry(coord(3,:),basis(3,3)[,unit_str])
! SUBROUTINE  cry2cart(coord(3,:),basis(3,3)[,unit_str])
! </INFO>
!

   INTERFACE cart2cry
      MODULE PROCEDURE cart2cry_rnk1
      MODULE PROCEDURE cart2cry_rnk2
   END INTERFACE
   INTERFACE cry2cart
      MODULE PROCEDURE cry2cart_rnk1
      MODULE PROCEDURE cry2cart_rnk2
   END INTERFACE
 
   PUBLIC :: cart2cry
   PUBLIC :: cry2cart

CONTAINS

!**********************************************************
   SUBROUTINE cart2cry_rnk1(coord,basis,unit_str)
   !**********************************************************
      IMPLICIT NONE
      REAL(dbl),   INTENT(inout)   :: coord(3)
      REAL(dbl),   INTENT(in)      :: basis(3,3)
      CHARACTER(*), OPTIONAL, INTENT(out)     :: unit_str
      REAL(dbl)    :: coord_tmp(3,1)

      coord_tmp(:,1) = coord(:) 
      CALL cart2cry_rnk2(coord_tmp,basis,unit_str)
      coord(:) = coord_tmp(:,1)
      RETURN
   END SUBROUTINE cart2cry_rnk1


!**********************************************************
   SUBROUTINE cart2cry_rnk2(coord,basis,unit_str)
   !**********************************************************
      IMPLICIT NONE
      REAL(dbl),   INTENT(inout)   :: coord(:,:)
      REAL(dbl),   INTENT(in)      :: basis(3,3)
      CHARACTER(*), OPTIONAL, INTENT(out)     :: unit_str

      REAL(dbl):: dtmp(3)
      REAL(dbl):: transf(3,3), det
      INTEGER  :: nvect 
      INTEGER  :: i,j,l

      nvect = SIZE(coord(:,:),2)
      IF ( SIZE( coord(:,:),1 ) /= 3 ) CALL errore('cart2cry','Invalid COORD lead DIM',1)

      !
      ! TRANSF is the inverse of the basis matrix because
      ! vcart(i) = \Sum_{j} vcry(j) * basis(j,i)
      ! Instead of the INV3 routine here TRANSF is directly theinverse of BASIS
      !
      CALL mat_inv( 3, basis, transf, det )
      !
      IF ( ABS(det) < EPS_m8 ) CALL errore('cart2cry','basis vectors are linearly dependent',1)

      DO j=1,nvect 
          DO i=1,3
             dtmp(i) = ZERO
             DO l=1,3
                 dtmp(i) = dtmp(i) + transf(i,l) * coord(l,j)
             ENDDO
          ENDDO
          coord(:,j) = dtmp(:)
      ENDDO

      IF ( PRESENT(unit_str) ) unit_str='crystal'
      RETURN
   END SUBROUTINE cart2cry_rnk2


!**********************************************************
   SUBROUTINE cry2cart_rnk1(coord,basis,unit_str)
   !**********************************************************
      IMPLICIT NONE
      REAL(dbl),   INTENT(inout)   :: coord(3)
      REAL(dbl),   INTENT(in)      :: basis(3,3)
      CHARACTER(*), OPTIONAL, INTENT(out)     :: unit_str
      REAL(dbl)    :: coord_tmp(3,1)

      coord_tmp(:,1) = coord(:) 
      CALL cry2cart_rnk2(coord_tmp,basis,unit_str)
      coord(:) = coord_tmp(:,1)
      RETURN
   END SUBROUTINE cry2cart_rnk1


!**********************************************************
   SUBROUTINE cry2cart_rnk2(coord,basis,unit_str)
   !**********************************************************
      IMPLICIT NONE
      REAL(dbl),   INTENT(inout)   :: coord(:,:)
      REAL(dbl),   INTENT(in)      :: basis(3,3)
      CHARACTER(*),OPTIONAL,INTENT(out) :: unit_str

      REAL(dbl):: dtmp(3)
      INTEGER  :: nvect 
      INTEGER  :: i,j,l

      nvect = SIZE(coord(:,:),2)
      IF ( SIZE( coord(:,:),1 ) /= 3 ) CALL errore('cry2cart','Invalid COORD lead DIM',1)

      !
      ! use the direct transformation
      !
      DO j=1,nvect 
          DO i=1,3
             dtmp(i) = ZERO
             DO l=1,3
                 dtmp(i) = dtmp(i) + basis(i,l) * coord(l,j)
             ENDDO
          ENDDO
          coord(:,j) = dtmp(:)
      ENDDO

      IF ( PRESENT(unit_str) ) unit_str='cartesian'
      RETURN
   END SUBROUTINE cry2cart_rnk2

END MODULE converters_module
    
    






