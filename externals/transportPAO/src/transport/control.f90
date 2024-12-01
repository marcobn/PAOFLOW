!
! Copyright (C) 2005 WanT Group, 2017 ERMES Group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License\'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!*********************************************
   MODULE control_module
!*********************************************
   USE kinds,      ONLY : dbl
   USE parameters, ONLY : nstrx
   IMPLICIT NONE
   PRIVATE 
   SAVE
!
! Contains GLOBAL CONTROL variables
! Many other variables governing the flow of the WanT code
! will be probably added.
! 
   
   CHARACTER(nstrx)          :: verbosity
   CHARACTER(nstrx)          :: restart_mode
   CHARACTER(nstrx)          :: subspace_init
   CHARACTER(nstrx)          :: localization_init
   CHARACTER(nstrx)          :: ordering_mode

   INTEGER                   :: nprint_dis
   INTEGER                   :: nsave_dis
   INTEGER                   :: nprint_wan
   INTEGER                   :: nsave_wan
   INTEGER                   :: nwfc_buffer
   INTEGER                   :: nkb_buffer
   INTEGER                   :: debug_level
   REAL(dbl)                 :: unitary_thr

   LOGICAL                   :: use_pseudo 
   LOGICAL                   :: use_uspp
   LOGICAL                   :: use_atomwfc 
   LOGICAL                   :: use_blimit 
   LOGICAL                   :: use_symmetry
   LOGICAL                   :: use_timerev
   LOGICAL                   :: use_debug_mode
   LOGICAL                   :: use_condmin

   LOGICAL                   :: do_overlaps
   LOGICAL                   :: do_projections
   LOGICAL                   :: do_polarization
   LOGICAL                   :: do_ordering
   LOGICAL                   :: do_collect_wf
   LOGICAL                   :: do_efermi

   LOGICAL                   :: read_pseudo 
   LOGICAL                   :: read_overlaps
   LOGICAL                   :: read_projections
   LOGICAL                   :: read_subspace
   LOGICAL                   :: read_unitary
   LOGICAL                   :: read_symmetry
   LOGICAL                   :: read_efermi
 
!
! end delcarations
!

   PUBLIC :: verbosity
   PUBLIC :: restart_mode
   PUBLIC :: ordering_mode
   PUBLIC :: subspace_init
   PUBLIC :: localization_init
   PUBLIC :: nprint_dis
   PUBLIC :: nsave_dis
   PUBLIC :: nprint_wan
   PUBLIC :: nsave_wan
   PUBLIC :: nwfc_buffer
   PUBLIC :: nkb_buffer
   PUBLIC :: unitary_thr
   PUBLIC :: debug_level

   PUBLIC :: use_pseudo
   PUBLIC :: use_atomwfc
   PUBLIC :: use_uspp
   PUBLIC :: use_blimit
   PUBLIC :: use_symmetry
   PUBLIC :: use_timerev
   PUBLIC :: use_debug_mode
   PUBLIC :: use_condmin

   PUBLIC :: do_overlaps
   PUBLIC :: do_projections
   PUBLIC :: do_polarization
   PUBLIC :: do_ordering
   PUBLIC :: do_collect_wf
   PUBLIC :: do_efermi

   PUBLIC :: read_pseudo
   PUBLIC :: read_overlaps
   PUBLIC :: read_projections
   PUBLIC :: read_subspace
   PUBLIC :: read_unitary
   PUBLIC :: read_symmetry
   PUBLIC :: read_efermi


END MODULE control_module

