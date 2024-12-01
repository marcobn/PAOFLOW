!
! Copyright (C) 2007 WanT Group, 2017 ERMES Group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License\'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!**********************************************************
   SUBROUTINE write_header(iun, msg)
   !**********************************************************
   !
   ! Print out the given header message msg
   !
   USE io_global_module, ONLY : ionode
   IMPLICIT NONE

   !
   ! input variables
   !
   INTEGER,      INTENT(IN) :: iun
   CHARACTER(*), INTENT(IN) :: msg

   !
   ! local variables
   !
   INTEGER :: msglen
   CHARACTER(256) :: str

!
!------------------------------
! main body
!------------------------------
!

   IF ( ionode ) THEN
      !
      msglen = LEN_TRIM( msg )
      WRITE( str, *) "(2x,'=  ',a,", 70-4-msglen, "x, '=')"
      !
      WRITE( iun, "(/,2x,70('='))" )
      WRITE( iun, FMT=TRIM(str) ) TRIM(msg)
      WRITE( iun, "(2x,70('='),/)" )
      !
   ENDIF
 
END SUBROUTINE write_header

