!
! Copyright (C) 2002-2011 Quantum ESPRESSO groups
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!
!==-----------------------------------------------------------------------==!
MODULE cuda_env
  !==-----------------------------------------------------------------------==!
#ifdef __CUDA
  USE ISO_C_BINDING
  !
  INTEGER(C_INT), BIND(C) :: ngpus_detected, ngpus_used, ngpus_per_process
  !
#endif
  !
  INTEGER :: cuda_env_i__
  !
  !==-----------------------------------------------------------------------==!
END MODULE cuda_env
!==-----------------------------------------------------------------------==!

