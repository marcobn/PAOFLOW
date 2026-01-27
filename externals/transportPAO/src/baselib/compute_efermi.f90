!
! Copyright (C) 2012   A. Ferretti
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License\'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!**********************************************************
   SUBROUTINE compute_efermi( nbnd, nspin, nelec, eig, delta, smearing_type, efermi )
   !**********************************************************
   !
   ! Compute the fermi energy once the eigenvalues and related data are given
   ! Different kind of broadening can be considered (such as "gaussian" or "lorentzian")
   !
   USE kinds
   !
   IMPLICIT NONE

   !
   ! input variables
   !
   INTEGER,                   INTENT(IN)  :: nbnd, nspin
   REAL(dbl),                 INTENT(IN)  :: nelec
   REAL(dbl),                 INTENT(IN)  :: eig(nbnd,nspin)
   REAL(dbl),                 INTENT(IN)  :: delta
   CHARACTER(*),              INTENT(IN)  :: smearing_type
   REAL(dbl),                 INTENT(OUT) :: efermi

   !
   ! local variables
   !
   CHARACTER(14) :: subname='compute_efermi'
   !
   REAL(dbl),    ALLOCATABLE :: wks(:)
   INTEGER,      ALLOCATABLE :: isk(:)
   !
   REAL(dbl), EXTERNAL :: efermig

!
!------------------------------
! main body
!------------------------------
!

   ALLOCATE( wks(nspin), isk(nspin) )
   wks(:) = 2.0d0/DBLE(nspin)
   isk(:) = 0
   !
   SELECT CASE ( TRIM(smearing_type) )
   CASE ( "gaussian", "selfenergy", "selfenergy1", "selfenergy2" )
       !
       ! selfenergy temporarily is here
       !
       efermi = efermig( eig, nbnd, nspin, nelec, wks, delta, 0, 0, isk )
       !
   CASE ( "fermi-dirac", "f-d", "fd")
       !
       efermi = efermig( eig, nbnd, nspin, nelec, wks, delta, -99, 0, isk )
       !
   CASE ( "methfessel-paxton", "m-p", "mp")
       !
       efermi = efermig( eig, nbnd, nspin, nelec, wks, delta,  1, 0, isk )
       !
   CASE ( "marzari-vanderbilt", "m-v", "mv")
       !
       efermi = efermig( eig, nbnd, nspin, nelec, wks, delta, -1, 0, isk )
       !
   CASE ( "lorentzian", "selfenergy0" )
       !
       efermi = efermig( eig, nbnd, nspin, nelec, wks, delta, -90, 0, isk )
       !
   CASE DEFAULT
       !
       efermi = 0.0d0
       CALL errore(subname,"invalid smearing_type: "//TRIM(smearing_type),10)
       !
   END SELECT
   !  
   DEALLOCATE( wks, isk )
 
   ! 
END SUBROUTINE compute_efermi

