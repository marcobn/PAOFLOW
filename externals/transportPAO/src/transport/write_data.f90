!
! Copyright (C) 2009 WanT Group, 2017 ERMES Group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License\'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!*********************************************
   MODULE T_write_data_module
   !*********************************************
   !
   ! This module contains basic routines to write
   ! the main data produed by the transport calculation
   !
   USE kinds,                   ONLY : dbl
   USE parameters,              ONLY : nstrx
   USE timing_module,           ONLY : timing
   USE log_module,              ONLY : log_push, log_pop
   USE io_module,               ONLY : ionode, stdout, io_name
   USE parser_module,           ONLY : int2char
   USE files_module,            ONLY : file_open, file_close
   USE iotk_module
   !
   IMPLICIT NONE
   PRIVATE 

   CHARACTER(nstrx) :: filename
   CHARACTER(nstrx) :: attr

   PUBLIC :: wd_write_data
   PUBLIC :: wd_write_eigchn


CONTAINS

!
! subroutines
!

!**********************************************************
   SUBROUTINE wd_write_data(iunit, ne, egrid, dim, mydata, data_type)
   !**********************************************************
   !
   IMPLICIT NONE
      !
      INTEGER,      INTENT(IN) :: iunit
      INTEGER,      INTENT(IN) :: ne, dim
      REAL(dbl),    INTENT(IN) :: mydata(dim,ne)
      REAL(dbl),    INTENT(IN) :: egrid(ne)
      CHARACTER(*), INTENT(IN) :: data_type
      !
      INTEGER       :: ie
      CHARACTER(20) :: str
      !
      CALL log_push( 'wd_write_data' )
      !
      IF ( ionode ) THEN 
          !
          CALL io_name( TRIM(data_type), filename, LPATH=.FALSE. )
          !
          OPEN ( iunit, FILE=TRIM(filename), FORM='formatted' )
          !
          str = TRIM( int2char(dim+1) )
          DO ie = 1, ne
              WRITE ( iunit, '('//TRIM(str)//'(f15.9))' ) egrid(ie), mydata(:,ie)
          ENDDO
          !
          CLOSE( iunit )
          !
          CALL io_name( TRIM(data_type), filename, LPATH=.FALSE. )
          WRITE(stdout,"(/,2x,a,' written on file: ',3x,a)") TRIM(data_type), TRIM(filename)
          !
      ENDIF
      !
      CALL log_pop( 'wd_write_data' )
      RETURN
      !
END SUBROUTINE wd_write_data


!**********************************************************
   SUBROUTINE wd_write_eigchn(iun, ie, ik, vk, tdir, dim1, dim2, mydata)
   !**********************************************************
   !
   IMPLICIT NONE
      !
      INTEGER,      INTENT(IN) :: iun
      INTEGER,      INTENT(IN) :: ie, ik
      REAL(dbl),    INTENT(IN) :: vk(3)   
      INTEGER,      INTENT(IN) :: tdir
      INTEGER,      INTENT(IN) :: dim1, dim2
      COMPLEX(dbl), INTENT(IN) :: mydata(dim1,dim2)
      !
      CHARACTER(15) :: subname='wd_write_eigchn'
      INTEGER       :: i, ierr
      !
      CALL log_push( subname )
      !
      CALL io_name( "eigchn", filename )
      CALL file_open(iun,TRIM(filename),PATH="/",ACTION="write", &
                     FORM="UNFORMATTED", IERR=ierr )
      IF ( ierr/=0 ) CALL errore(subname, 'opening '//TRIM(filename), ABS(ierr) )
          !
          CALL iotk_write_begin(iun,"EIGENCHANNELS")
          !
          CALL iotk_write_attr(attr, "ik", ik, FIRST=.TRUE.)
          CALL iotk_write_attr(attr, "ie", ie )
          CALL iotk_write_attr(attr, "transport_dir", tdir )
          CALL iotk_write_attr(attr, "dim1", dim1)
          CALL iotk_write_attr(attr, "dim2", dim2)
          CALL iotk_write_empty(iun, "DATA", ATTR=attr)
          !
          CALL iotk_write_attr(attr, "units", "crystal", FIRST=.TRUE.)
          CALL iotk_write_dat(iun, "vkpt", vk(1:3), ATTR=attr )
          !
          DO i = 1, dim2
              !
              CALL iotk_write_dat(iun,"eigchn"//TRIM(iotk_index(i)), mydata(1:dim1,i))
              !
          ENDDO
          !
          CALL iotk_write_end(iun,"EIGENCHANNELS")
          !
      CALL file_close(iun,PATH="/",ACTION="write", IERR=ierr)
      IF ( ierr/=0 ) CALL errore(subname, 'closing '//TRIM(filename), ABS(ierr) )
      !
      CALL io_name( "eigchn", filename, LPATH=.FALSE. )
      !IF (ionode) 
      WRITE(stdout,"(/,2x,a,'Eigenchannels written on file: ',3x,a)") TRIM(filename)
      !
      CALL log_pop( subname )
      RETURN
      !
END SUBROUTINE wd_write_eigchn

END MODULE T_write_data_module


