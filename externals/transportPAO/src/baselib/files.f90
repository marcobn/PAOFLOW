!
! Copyright (C) 2004 WanT Group, 2017 ERMES Group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License\'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!

! <INFO>
!*********************************************
   MODULE files_module
   !*********************************************
   !
   USE parser_module,    ONLY : parser_path, change_case
   USE iotk_module
   !
   IMPLICIT NONE
   PRIVATE

! This module contains some utilities to manage
! IO from files 
!
! routines in this module:
! SUBROUTINE  file_open(unit,filename[,path][,root][,status][,access][,recl] 
!                       [,form][,position][,action], ierr)
! SUBROUTINE  file_close(unit[,path][,action], ierr)
! SUBROUTINE  file_delete(name)
! FUNCTION    file_exist(name)
! SUBROUTINE  file_rename(oldname,newname)
!
! PATH is the IOTK_PATH (root NOT included) for the
! eventual XML tree. The path is a string as
! "/tag1/tag2/...../tagN " (path string)
! A VOID string indicates the ROOT path ("/")
! while PATH = "none" avoids the use of IOTK.
!
! In OPENING READ the file will be positioned inside the
! desidered folder, while in CLOSING READ the file is
! supposed to be in the specified folder. 
! OPENING and CLOSING WRITE only works from the ROOT fld.
!
! </INFO>
!


!
! end of declarations
!

   INTEGER, PARAMETER :: nstrx = 256
   INTEGER, PARAMETER :: unitx = 100

   PUBLIC ::  file_open
   PUBLIC ::  file_close
   PUBLIC ::  file_delete
   PUBLIC ::  file_exist

CONTAINS

!
! Subroutines
!   

!**********************************************************
   SUBROUTINE file_delete(filename)
   !**********************************************************
   IMPLICIT NONE
      CHARACTER(*),  INTENT(in)  :: filename
      LOGICAL :: exist
      INTEGER :: unit,ierr
   
      INQUIRE(FILE=TRIM(filename),EXIST=exist)
      !
      IF (.NOT. exist) RETURN
      !
      CALL iotk_free_unit(unit)
      !
      OPEN(unit,FILE=TRIM(filename),IOSTAT=ierr)
      IF (ierr/=0) CALL errore('file_delete','Unable to open file '//TRIM(filename),1)
      !
      CLOSE(unit,STATUS='delete',IOSTAT=ierr)
      !
      IF (ierr/=0) CALL errore('file_delete','Unable to close file '//TRIM(filename),1)
      !
   END SUBROUTINE file_delete
      

!**********************************************************
   LOGICAL FUNCTION file_exist(filename)
   !**********************************************************
   IMPLICIT NONE
      CHARACTER(*)  :: filename
   
      INQUIRE(FILE=TRIM(filename),EXIST=file_exist)
      !
      RETURN
      !
   END FUNCTION file_exist


!**********************************************************
   SUBROUTINE file_open(unit,filename,path,root,       &
                       status,access,recl,form,position,action, ierr)
   !**********************************************************
   IMPLICIT NONE
      INTEGER,                INTENT(in)  :: unit
      CHARACTER(*),           INTENT(in)  :: filename
      CHARACTER(*), OPTIONAL, INTENT(in)  :: path
      CHARACTER(*), OPTIONAL, INTENT(in)  :: root
      CHARACTER(*), OPTIONAL, INTENT(in)  :: status
      CHARACTER(*), OPTIONAL, INTENT(in)  :: access
      INTEGER,      OPTIONAL, INTENT(in)  :: recl
      CHARACTER(*), OPTIONAL, INTENT(in)  :: form
      CHARACTER(*), OPTIONAL, INTENT(in)  :: position
      CHARACTER(*), OPTIONAL, INTENT(in)  :: action
      INTEGER,                INTENT(out) :: ierr

      !CHARACTER(9)                        :: subname='file_open'
      CHARACTER(7)                        :: status_
      CHARACTER(10)                       :: access_
      CHARACTER(11)                       :: form_
      CHARACTER(6)                        :: position_
      CHARACTER(9)                        :: action_
      CHARACTER(10*nstrx)                 :: path_
       
      LOGICAL                             :: fmt_iotk, tmp, binary
      INTEGER                             :: ndir, i
      CHARACTER(nstrx), POINTER           :: tags(:)

      !
      ! Allowed value for OPENING attributes
      !
      ! STATUS     = "old", "replace", "unknown"
      ! ACCESS     = "direct", "sequential"
      ! FORM       = "formatted", "unformatted"
      ! POSITION   = "asis", "rewind", "append"
      ! ACTION     = "read", "write", "readwrite"
      !
      ! If PATH is not present is assumed to be "/"
      ! that correspond to the IOTK ROOT file folder
      ! If PATH = "none" the iotk interface format is not used.
      ! PATH="/" just open the IOTK file in the main folder
      ! otherwise the requested path is searched.
      ! ROOT can be used only for WRITE opening
      !
     
      !
      ! Set defaults
      ! 
      NULLIFY ( tags )
      ierr = 0
      !
      status_   = "UNKNOWN"
      access_   = "SEQUENTIAL"
      form_     = "FORMATTED"
      position_ = "ASIS"
      action_   = "READWRITE"
      path_     = "/"

      !
      ! Passing Values
      ! 
      IF (PRESENT(status))      status_ = TRIM(status)
      IF (PRESENT(access))      access_ = TRIM(access)
      IF (PRESENT(form))          form_ = TRIM(form)
      IF (PRESENT(position))  position_ = TRIM(position)
      IF (PRESENT(action))      action_ = TRIM(action)
      IF (PRESENT(path))          path_ = TRIM(path)

      CALL change_case(status_,'UPPER')
      CALL change_case(access_,'UPPER')
      CALL change_case(form_,'UPPER')
      CALL change_case(position_,'UPPER')
      CALL change_case(action_,'UPPER')
      
      !
      ! Whether using IOTK
      ! 
      fmt_iotk = .TRUE.
      IF ( TRIM(path_) == "none" .OR. TRIM(path_) == "NONE" ) THEN 
         fmt_iotk = .FALSE.
      ELSE
         IF ( TRIM(path_) == "/") THEN
            ndir = 0
         ELSE
            CALL parser_path(TRIM(path_),ndir,tags)
         ENDIF
      ENDIF


      !
      ! Checking allowed values
      ! 
      IF ( TRIM(status_) /= "OLD" .AND. TRIM(status_) /= "NEW" .AND.         &
           TRIM(status_) /= "REPLACE" .AND.  TRIM(status_) /= "UNKNOWN")  ierr = 1
      IF ( TRIM(access_) /= "DIRECT" .AND. TRIM(access_) /= "SEQUENTIAL") ierr = 2
      IF ( TRIM(form_) /= "FORMATTED" .AND. TRIM(form_) /= "UNFORMATTED") ierr = 3
      IF ( TRIM(position_) /= "ASIS" .AND. TRIM(position_) /= "REWIND" .AND. &
           TRIM(position_) /= "APPEND")                                   ierr = 4
      IF ( TRIM(action_) /= "READ" .AND. TRIM(action_) /= "WRITE" .AND. &
           TRIM(action_) /= "READWRITE")                                  ierr = 5

      IF ( ierr/=0 ) RETURN

      !
      ! Compatibility
      ! 
      INQUIRE(unit, OPENED=tmp)
      !
      IF ( tmp ) ierr = 10
      IF ( fmt_iotk .AND. .NOT. PRESENT(action) ) ierr = 11
      IF ( fmt_iotk .AND. TRIM(action_) == "READWRITE" ) ierr = 12
      IF ( fmt_iotk .AND. (PRESENT(access) .OR. PRESENT(position)) ) ierr = 13
      IF ( fmt_iotk .AND. TRIM(action_) == "READ" .AND. PRESENT(status) ) ierr = 14

      IF ( TRIM(access_) == "DIRECT" .AND. .NOT. PRESENT(recl) ) ierr = 15
      IF ( PRESENT(recl) ) THEN
           IF ( recl <= 0 ) ierr = 16
           IF ( TRIM(access_) /= "DIRECT" ) ierr = 17
      ENDIF
      IF ( TRIM(access_) == "DIRECT" .AND. PRESENT(position) ) ierr = 18
      IF ( .NOT. fmt_iotk .AND. PRESENT(root) ) ierr = 19
      IF ( PRESENT(root) .AND. TRIM(action_) /= "WRITE") ierr = 20
      IF ( TRIM(action_) == "WRITE" .AND. ndir /= 0 ) ierr = 21

      IF ( ierr /= 0 ) RETURN

      
      !
      ! IOTK opening
      !
      IF ( fmt_iotk )  THEN
         
         binary = .FALSE.  
         IF ( TRIM(form_) == "UNFORMATTED" ) binary = .TRUE.
         
         SELECT CASE (TRIM(action_))
         CASE("READ")
            CALL iotk_open_read(unit,FILE=TRIM(filename),IERR=ierr)
         CASE("WRITE")
            IF ( PRESENT(root) ) THEN
               CALL iotk_open_write(unit,FILE=TRIM(filename),BINARY=binary, &
                                    ROOT=root,IERR=ierr)
            ELSE 
               CALL iotk_open_write(unit,FILE=TRIM(filename),BINARY=binary,IERR=ierr)
            ENDIF

         CASE DEFAULT
             ierr = 30
         END SELECT
         !
         IF ( ierr/= 0 ) RETURN

         !
         ! goes into path
         !
         DO i=1,ndir
             !
             CALL iotk_scan_begin(unit,TRIM(tags(i)),IERR=ierr )
             !
             IF ( ierr /= 0 ) RETURN
             !
         ENDDO
           

      !
      ! ORDINARY opening
      !
      ELSE
          !
          IF ( TRIM(access_ ) == "DIRECT" ) THEN
              OPEN(unit,FILE=filename,STATUS=status_,ACCESS=access_,RECL=recl,   &
                        ACTION=action_,FORM=form_,IOSTAT=ierr)
          ELSE
              OPEN(unit,FILE=filename,STATUS=status_,POSITION=position_,   &
                        ACTION=action_,FORM=form_,IOSTAT=ierr)
          ENDIF
          !
          IF ( ierr/= 0) RETURN
          !
      ENDIF    

      !
      ! cleaning
      !
      IF ( ASSOCIATED(tags) ) DEALLOCATE(tags)

   END SUBROUTINE file_open


!**********************************************************
   SUBROUTINE file_close(unit,path,action,ierr)
   !**********************************************************
   IMPLICIT NONE
      INTEGER,                INTENT(in)  :: unit
      CHARACTER(*), OPTIONAL, INTENT(in)  :: path
      CHARACTER(*), OPTIONAL, INTENT(in)  :: action
      INTEGER,                INTENT(out) :: ierr

      !CHARACTER(10)                       :: subname='file_close'
      CHARACTER(9)                        :: action_
      CHARACTER(10*nstrx)                 :: path_
       
      LOGICAL                             :: fmt_iotk, tmp
      INTEGER                             :: ndir, i
      CHARACTER(nstrx), POINTER           :: tags(:)

      !
      ! Set defaults
      ! 
      ierr      = 0
      NULLIFY(tags)
      !
      action_   = "READWRITE"
      path_     = "/"

      !
      ! Passing Values
      ! 
      IF (PRESENT(action))      action_ = TRIM(action)
      IF (PRESENT(path))          path_ = TRIM(path)

      CALL change_case(action_,'UPPER')

      !
      ! Whether using IOTK
      ! 
      fmt_iotk = .TRUE.
      IF ( TRIM(path_) == "none" .OR. TRIM(path_) == "NONE") THEN 
         fmt_iotk = .FALSE.
      ELSE
         IF ( TRIM(path_) == "/") THEN
            ndir = 0
         ELSE
            CALL parser_path(TRIM(path_),ndir,tags)
         ENDIF
      ENDIF


      !
      ! Checking allowed values
      ! 
      IF ( TRIM(action_) /= "READ" .AND. TRIM(action_) /= "WRITE" .AND. &
           TRIM(action_) /= "READWRITE")               ierr = 11
      IF ( TRIM(action_) == "WRITE" .AND. ndir /= 0 )  ierr = 12
      !
      IF ( ierr/=0 ) RETURN

      !
      ! Compatibility
      ! 
      INQUIRE(unit,OPENED=tmp)
      !
      IF ( .NOT. tmp ) ierr = 20
      ! 
      IF ( fmt_iotk .AND. .NOT. PRESENT(action) ) ierr = 21
      IF ( fmt_iotk .AND. TRIM(action_) == "READWRITE" ) ierr = 22
      !
      IF ( ierr/=0 ) RETURN


      
      !
      ! IOTK closing
      !
      IF ( fmt_iotk )  THEN

         !
         ! moving to the ROOT folder
         !
         DO i=1,ndir
             !
             CALL iotk_scan_end(unit,TRIM(tags(i)),IERR=ierr )
             !
             IF (ierr/= 0 ) RETURN
             !
         ENDDO
         !
         SELECT CASE (TRIM(action_))
         CASE("READ")
            CALL iotk_close_read(unit,IERR=ierr)

         CASE("WRITE")
            CALL iotk_close_write(unit,IERR=ierr)

         CASE DEFAULT
            ierr = 30

         END SELECT
         !
         IF ( ierr/= 0) RETURN


      !
      ! ORDINARY closing
      !
      ELSE
          !
          CLOSE(unit,IOSTAT=ierr)
          !
          IF ( ierr/= 0) RETURN
          !
      ENDIF

      !
      ! cleaning
      !
      IF ( ASSOCIATED(tags) ) DEALLOCATE(tags)

   END SUBROUTINE file_close

END MODULE files_module

