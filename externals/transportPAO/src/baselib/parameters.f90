!
! Copyright (C) 2001-2003 PWSCF group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!
module parameters
  !
  !
     IMPLICIT NONE
     SAVE

  !
  !       First all the parameter declaration
  !
  INTEGER , PARAMETER ::   &
       nstrx  = 600,       &! max lenght for strings
       ntypx  = 10,        &! max number of different types of atom
       npsx   = ntypx,     &! obsolete, for PWscf compatibility
       npkx   = 100000,    &! max number of k-points               
       npwx   = 100000000, &! max number of density G vectors
       nshx   = 200,       &! max number of nearest neighb. k-point shells
       lmaxx  = 3,         &! max non local angular momentum       
       nchix  = 6,         &! max number of atomic wavefunctions per atom
       ndmx   = 2000        ! max number of points in the atomic radial mesh

  INTEGER , PARAMETER  ::  &
    cp_lmax = lmaxx + 1,   &! maximum number of channels
       nbrx = 14,          &! max number of beta functions
       lqmax= 2*lmaxx+1,   &! max number of angular momenta of Q
       nqfx = 8             ! max number of coefficients in Q smoothing

  INTEGER , PARAMETER  ::  &
       nkpts_inx = 100,    &! max number of interpolated kpoints 
       nnx = 12,           &! max number of kpt nearest-neighbours
       nnhx = 6             ! halp the previous value

  INTEGER, PARAMETER :: natx  = 600     ! maximum number of atoms
  INTEGER, PARAMETER :: nbndxx = 10000  ! maximum number of electronic states
  INTEGER, PARAMETER :: nspinx = 2      ! maximum number of spinors
  !
end module parameters

