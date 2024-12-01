!
! Copyright (C) 2005 WanT Group, 2017 ERMES Group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License\'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!*********************************************
   MODULE T_control_module
   !*********************************************
   !
   USE kinds,      ONLY : dbl
   USE parameters, ONLY : nstrx
   !
   IMPLICIT NONE
   PRIVATE 
   SAVE
!
! Contains GLOBAL CONTROL variables for transport calculations
! 
   
   CHARACTER(nstrx)          :: calculation_type
   CHARACTER(nstrx)          :: conduct_formula
   !
   CHARACTER(nstrx)          :: datafile_L, datafile_C, datafile_R
   LOGICAL                   :: do_orthoovp = .FALSE.
   !
   INTEGER                   :: transport_dir
   INTEGER                   :: debug_level
   !
   LOGICAL                   :: do_eigenchannels = .FALSE.
   LOGICAL                   :: do_eigplot = .FALSE.
   INTEGER                   :: ie_eigplot = 0
   INTEGER                   :: ik_eigplot = 0
   !
   LOGICAL                   :: use_debug_mode
   LOGICAL                   :: write_kdata
   LOGICAL                   :: write_lead_sgm
   LOGICAL                   :: write_gf
   !
   INTEGER                   :: niterx
   INTEGER                   :: nfailx
   REAL(dbl)                 :: transfer_thr
   LOGICAL                   :: leads_are_identical = .FALSE.
   INTEGER                   :: neigchnx, neigchn
   INTEGER                   :: nfail = 0
   !
   INTEGER                   :: nprint
   !
   REAL(dbl)                 :: bias

!
! end delcarations
!

   PUBLIC :: calculation_type
   PUBLIC :: conduct_formula
   PUBLIC :: datafile_L, datafile_C, datafile_R
   PUBLIC :: do_orthoovp
   !
   PUBLIC :: transport_dir
   PUBLIC :: debug_level
   !
   PUBLIC :: do_eigenchannels
   PUBLIC :: neigchnx, neigchn
   PUBLIC :: do_eigplot
   PUBLIC :: ie_eigplot, ik_eigplot
   !
   PUBLIC :: use_debug_mode
   !
   PUBLIC :: write_kdata
   PUBLIC :: write_lead_sgm
   PUBLIC :: write_gf
   !
   PUBLIC :: niterx, nfailx
   PUBLIC :: nfail
   PUBLIC :: transfer_thr
   PUBLIC :: leads_are_identical
   !
   PUBLIC :: nprint
   !
   PUBLIC :: bias

END MODULE T_control_module

