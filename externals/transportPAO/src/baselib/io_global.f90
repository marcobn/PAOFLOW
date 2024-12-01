!
! Copyright (C) 2005 WanT Group, 2017 ERMES Group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
! <INFO>
!*********************************************
   MODULE io_global_module
!*********************************************
   USE iotk_module
   IMPLICIT NONE
   PRIVATE
   SAVE

!
! Basic IO variables 
! 
   INTEGER, PARAMETER         ::   &
         stderr = 0,               &! std error unit
          stdin = 5,               &! std input unit
         stdout = 6                 ! std output unit
         
   INTEGER :: ionode_id = 0
   LOGICAL :: ionode = .TRUE.
   LOGICAL :: first = .TRUE.

!
! end delcarations
!

   PUBLIC ::  stdin, stdout, stderr 
   PUBLIC ::  ionode, ionode_id
   PUBLIC ::  io_global_start, io_global_getionode

CONTAINS


!
! Copyright (C) 2002 FPMD & PWSCF group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
     !
     !-----------------------------------------------------------------------
     SUBROUTINE io_global_start( mpime, ionode_set )
       !-----------------------------------------------------------------------
       !
       IMPLICIT NONE
       !
       INTEGER, INTENT(IN) :: mpime, ionode_set
       !
       !
       IF ( mpime == ionode_set ) THEN
          ionode = .TRUE.
       ELSE
          ionode = .FALSE.
       END IF
       !
       ionode_id = ionode_set
       first = .FALSE.
       !
       RETURN
       !
     END SUBROUTINE io_global_start
     !
     !
     !-----------------------------------------------------------------------
     SUBROUTINE io_global_getionode( ionode_out, ionode_id_out )
       !-----------------------------------------------------------------------
       !
       IMPLICIT NONE
       !
       LOGICAL, INTENT(OUT) :: ionode_out
       INTEGER, INTENT(OUT) :: ionode_id_out
       !
       !
       IF ( first ) &
          CALL errore( ' get_ionode ', ' ionode not yet defined ', 1 )
       ionode_out = ionode
       ionode_id_out = ionode_id
       !
       RETURN
       !
     END SUBROUTINE io_global_getionode

END MODULE io_global_module

