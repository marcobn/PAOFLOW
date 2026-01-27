!
! Copyright (C) 2004 WanT Group, 2017 ERMES Group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!*********************************************
   MODULE io_module
!*********************************************
   !
   USE kinds,                   ONLY : dbl
   USE parameters,              ONLY : nstrx
   USE io_global_module,        ONLY : stdout, stdin, stderr, ionode, ionode_id, &
                                       io_global_start, io_global_getionode
   USE files_module,            ONLY : file_open, file_close
   USE mp_global,               ONLY : nproc, mpime
   USE mp,                      ONLY : mp_bcast
   USE control_module,          ONLY : read_symmetry, use_pseudo, read_pseudo, use_debug_mode, debug_level
   USE log_module,              ONLY : log_init, log_push, log_pop
   USE crystal_io_module,       ONLY : crio_open_file, crio_read_header, crio_close_file
   USE crystal_tools_module,    ONLY : file_is_crystal
   USE wannier90_tools_module,  ONLY : file_is_wannier90
   USE internal_tools_module,   ONLY : file_is_internal
   USE qexml_module
   USE qexpt_module
   USE iotk_module
   !
#ifdef __ETSF_IO
   USE etsf_io
   USE etsf_io_tools
   USE etsf_io_data_module,  ONLY : ncid, lstat, dims, error_data, etsf_io_bcast
#endif
   !
   IMPLICIT NONE
   PRIVATE
   SAVE
   !
   ! Contains basic data concerning IO
   ! 
   ! contains:
   ! SUBROUTINE  io_init()
   ! SUBROUTINE  io_name(data,filename[,lpostfix][,lpath])
   ! SUBROUTINE  io_read_data(unit,name,prefix,postfix,work_dir,title,found)
   ! SUBROUTINE  io_write_data(unit,name,prefix,postfix,work_dir,title)
   !
   ! DATA in io_name routine should be:
   ! 
   ! * 'dft_data'
   ! * 'overlap_projection'
   ! * 'space'
   ! * 'wannier'
   ! * 'hamiltonian'
   ! * 'save'
   ! * 'log'
   !
   INTEGER, PARAMETER         ::   &
       dft_unit = 10,              &! input file (DFT data) unit
    pseudo_unit = 11,              &! input pseudopotential data unit
       log_unit = 12,              &! input pseudopotential data unit
       ovp_unit = 20,              &! overlap and projections unit
     space_unit = 21,              &! space unit
       wan_unit = 22,              &! wannier stuff unit
       ham_unit = 23,              &! hamiltonian unit
       sgm_unit = 24,              &! self-energy unit
       aux_unit = 30,              &! auxiliary units
      aux1_unit = 31,              &! 
      aux2_unit = 32,              &! 
      aux3_unit = 33,              &! 
      aux4_unit = 34,              &! 
      aux5_unit = 35,              &! 
      save_unit = 60                ! restart file unit


   CHARACTER( 6), PARAMETER   ::  suffix_space=".space"
   CHARACTER( 4), PARAMETER   ::  suffix_ovp=".ovp"
   CHARACTER( 7), PARAMETER   ::  suffix_transl=".transl"
   CHARACTER( 4), PARAMETER   ::  suffix_wannier=".wan"
   CHARACTER( 4), PARAMETER   ::  suffix_hamiltonian=".ham"
   CHARACTER( 5), PARAMETER   ::  suffix_save=".save"
   CHARACTER( 4), PARAMETER   ::  suffix_log=".log"
   CHARACTER( 4), PARAMETER   ::  suffix_sgm=".sgm"
   CHARACTER( 3), PARAMETER   ::  suffix_gf=".gf"
   CHARACTER(nstrx)           ::  suffix_qe_data=" "
   CHARACTER(12), PARAMETER   ::  suffix_etsf_io_data="_WFK-etsf.nc"
   REAL,          PARAMETER   ::  etsf_io_version_min=2.1
   
   CHARACTER(nstrx)           :: prefix
   CHARACTER(nstrx)           :: postfix
   CHARACTER(nstrx)           :: work_dir
   CHARACTER(nstrx)           :: title
   CHARACTER(nstrx)           :: pseudo_dir
         
   CHARACTER(nstrx)           :: dftdata_fmt = ' '
   CHARACTER(nstrx)           :: dftdata_fmt_version
   CHARACTER(nstrx)           :: dftdata_file = ' '
   CHARACTER(nstrx)           :: wantdata_fmt
   CHARACTER(nstrx)           :: wantdata_form
   LOGICAL                    :: wantdata_binary
   !
   CHARACTER(nstrx)           :: datafile_sgm = ' '

   LOGICAL                    :: alloc = .FALSE.
!
! end delcarations
!


   PUBLIC ::  io_global_start, io_global_getionode
   PUBLIC ::  mpime, ionode, ionode_id

   PUBLIC ::  stderr, stdin, stdout 
   PUBLIC ::  dftdata_fmt, dftdata_fmt_version, dftdata_file
   PUBLIC ::  datafile_sgm
   PUBLIC ::  wantdata_fmt, wantdata_form, wantdata_binary
   PUBLIC ::  dft_unit, pseudo_unit 
   PUBLIC ::  ovp_unit, space_unit, wan_unit, ham_unit, sgm_unit 
   PUBLIC ::  aux_unit, aux1_unit, aux2_unit, aux3_unit, aux4_unit, aux5_unit
   PUBLIC ::  save_unit, log_unit
   PUBLIC ::  etsf_io_version_min

   PUBLIC ::  prefix, postfix, work_dir, title, pseudo_dir
   PUBLIC ::  alloc
   PUBLIC ::  io_init
   PUBLIC ::  io_name
   PUBLIC ::  io_set_nd_nmbr
   PUBLIC ::  io_open_dftdata
   PUBLIC ::  io_close_dftdata
   PUBLIC ::  io_read_data
   PUBLIC ::  io_write_data


   CONTAINS

!
! subroutines
!
!**********************************************************
   SUBROUTINE io_get_dftdata_fmt(prefix_, work_dir_, dftdata_file_, dftdata_fmt_, need_wfc)
   !**********************************************************
      !
      ! get the fmt of the dftdata file (use the names)
      !
      IMPLICIT NONE
      CHARACTER(*),      INTENT(IN)  :: prefix_, work_dir_, dftdata_file_
      CHARACTER(*),      INTENT(OUT) :: dftdata_fmt_
      LOGICAL, OPTIONAL, INTENT(IN)  :: need_wfc
      !
      CHARACTER(18)    :: subname='io_get_dftdata_fmt'  
      !
      CHARACTER(nstrx) :: filename, version
      CHARACTER(nstrx) :: fmt_searched(6)
      CHARACTER(nstrx) :: fmt_filename(6)
      LOGICAL          :: lfound, lfound1, lneed_wfc
      INTEGER          :: i, ierr

      !
      ! Setting fmts to be searched
      !
      fmt_searched(1) = 'internal'
      fmt_searched(2) = 'wannier90'
      fmt_searched(3) = 'crystal'
      fmt_searched(4) = 'qexml'
      fmt_searched(5) = 'pw_export'
      fmt_searched(6) = 'etsf_io'
      !
      fmt_filename(1) = TRIM(dftdata_file_)
      fmt_filename(2) = TRIM(dftdata_file_)
      fmt_filename(3) = TRIM(dftdata_file_)
      fmt_filename(4) = '.save/data-file.xml'
      fmt_filename(5) = '.export/index.xml'
      fmt_filename(6) = '_WFK-etsf.nc'

      !
      ! init
      lfound    = .FALSE.
      lneed_wfc = .TRUE.
      IF ( PRESENT( need_wfc )) lneed_wfc = need_wfc
      !
      !
      DO i = 1, SIZE( fmt_searched )
           !
           ! set the filename
           !
           IF ( ionode ) WRITE(stdout, "(2x, 'checking for fmt ',a,'... ')", advance='no' ) &
                         TRIM( fmt_searched(i) )
           !
           IF ( TRIM( fmt_searched(i) ) == 'crystal'    .OR.  &
                TRIM( fmt_searched(i) ) == 'wannier90'  .OR.  &
                TRIM( fmt_searched(i) ) == 'internal'   ) THEN
               !
               ! in these cases, the presence of 
               ! a non-null dftdata_file is required
               !
               IF ( LEN_TRIM(fmt_filename( i )) == 0 ) THEN
                   IF ( ionode ) WRITE(stdout, "('no')" ) 
                   CYCLE
               ENDIF
               !
               filename = TRIM( fmt_filename( i ) )
               !
           ELSE
               filename = TRIM( work_dir_ ) //'/'// TRIM(prefix_) // TRIM( fmt_filename( i ) )
           ENDIF
           !
           ! check the existence of the file
           IF (ionode) INQUIRE ( FILE=TRIM(filename), EXIST=lfound )
           CALL mp_bcast( lfound,   ionode_id )
           !
           IF ( lfound .AND. lneed_wfc .AND. TRIM( fmt_searched(i) ) == 'qexml'  )  THEN
               !
               ! check also the existence of evc.dat or evc1.dat
               ! this means that file produced by espresso are fine for WanT
               !
               filename = TRIM( work_dir_ ) //'/'// TRIM(prefix_) // ".save/K00001/evc.dat"
               IF (ionode) INQUIRE ( FILE=TRIM(filename), EXIST=lfound )
               CALL mp_bcast( lfound,   ionode_id )
               !
               filename = TRIM( work_dir_ ) //'/'// TRIM(prefix_) // ".save/K00001/evc1.dat"
               IF (ionode) INQUIRE ( FILE=TRIM(filename), EXIST=lfound1 )
               CALL mp_bcast( lfound1,   ionode_id )
               !            
               lfound = lfound .OR. lfound1
               !
               !
               ! check  the version of the format.
               ! At the moment, if the header section exist, 
               ! the fmt is supported whatever version
               !
               filename = TRIM(work_dir_) //'/'// TRIM(prefix_) // '.save/data-file.xml'
               !
               IF (ionode) CALL qexml_openfile( filename, "read", IERR=ierr )
               CALL mp_bcast( ierr,   ionode_id )
               IF ( ierr /= 0 ) CALL errore(subname,'opening dftdata file',ABS(ierr))
               !
               IF (ionode) CALL qexml_read_header( FORMAT_VERSION=version, IERR=ierr )
               CALL mp_bcast( ierr,   ionode_id )
               CALL mp_bcast( version,   ionode_id )
               !
               IF ( ierr /= 0 )  lfound = .FALSE.
               !
               ! any check on the version should be placed here
               !
               IF (ionode) CALL qexml_closefile ( "read", IERR=ierr )
               CALL mp_bcast( ierr,   ionode_id )
               IF ( ierr /= 0 ) CALL errore(subname,'closing dftdata file',ABS(ierr))
               !
           ENDIF
           !
           IF ( lfound .AND. TRIM( fmt_searched(i) ) == 'internal'  )  THEN
               !
               lfound = file_is_internal( filename )
               !
           ENDIF
           !
           IF ( lfound .AND. TRIM( fmt_searched(i) ) == 'crystal'  )  THEN
               !
               lfound = file_is_crystal( filename )
               !
           ENDIF
           !
           IF ( lfound .AND. TRIM( fmt_searched(i) ) == 'wannier90'  )  THEN
               !
               lfound = file_is_wannier90( filename )
               !
           ENDIF
           !
#ifdef __ETSF_IO
           IF ( lfound .AND. TRIM( fmt_searched(i) ) == 'etsf_io')  THEN
               !
               ! Try to open the file and to get dimensions to check that
               ! the file is actually ETSF_IO formatted
               !
               IF ( ionode ) THEN
                   CALL etsf_io_low_open_read(ncid, filename, lstat, &
                                              ERROR_DATA=error_data, &
                                              VERSION_MIN=etsf_io_version_min )
               ENDIF
               CALL mp_bcast( lstat,   ionode_id )
               IF ( .NOT. lstat ) EXIT
               !
               IF ( ionode ) THEN
                   CALL etsf_io_dims_get(ncid, dims, lstat, error_data)
               ENDIF
               CALL mp_bcast( lstat,   ionode_id )
               IF ( .NOT. lstat ) EXIT
               !
               IF ( ionode ) THEN
                   CALL etsf_io_low_close(ncid, lstat, error_data)
               ENDIF
               CALL mp_bcast( lstat,   ionode_id )
               !
               lfound = lstat
               !
           ENDIF
#endif
           !
           IF ( lfound ) THEN
               IF ( ionode ) WRITE(stdout, "('ok')" ) 
               EXIT
           ELSE
               IF ( ionode ) WRITE(stdout, "('no')" ) 
           ENDIF
           !
      ENDDO
      !
      IF ( .NOT. lfound ) THEN
           dftdata_fmt_ = ""
      ELSE
           dftdata_fmt_ = TRIM( fmt_searched( i ) )
           !
           IF (ionode) WRITE( stdout , "(/,2x, 'DFT-data fmt automaticaly detected: ',a )" ) &
                  TRIM( dftdata_fmt_)
           !
      ENDIF
      !
   END SUBROUTINE io_get_dftdata_fmt
      

!**********************************************************
   SUBROUTINE io_init( need_wfc )
   !**********************************************************
   !
   ! init some data related to IO and taken from input
   !
   IMPLICIT NONE
      !
      LOGICAL, OPTIONAL, INTENT(IN) :: need_wfc
      !
      LOGICAL           :: lneed_wfc
      INTEGER           :: ierr
      CHARACTER(7)      :: subname="io_init"
      CHARACTER(nstrx)  :: dirname, logfile, filename

      !
      lneed_wfc = .TRUE.
      IF ( PRESENT(need_wfc) ) lneed_wfc = need_wfc
      !
      !
      SELECT CASE ( TRIM(wantdata_fmt) ) 
      !
      CASE ( 'textual' )
           !
           wantdata_form   = 'formatted'
           wantdata_binary = .FALSE.
           !
      CASE ( 'binary' )
           !
           wantdata_form   = 'unformatted'
           wantdata_binary = .TRUE.
           !
      END SELECT
      
      !
      ! init libs for all fmts
      !
      dirname = TRIM(work_dir) // '/' // TRIM(prefix) // '.save/'
      CALL qexml_init( dft_unit, DIR=dirname )
      !
      dirname  = TRIM(work_dir) // '/' // TRIM(prefix) // '.export/'
      CALL qexpt_init( dft_unit, dirname )
      !
      !
      IF ( LEN_TRIM( dftdata_fmt ) == 0 ) THEN 
          !
          CALL io_get_dftdata_fmt ( prefix, work_dir, dftdata_file, dftdata_fmt, lneed_wfc )
          !
      ENDIF

      !
      ! if dftdata_fmt is still empty, it means no complete dtf dataset
      ! has been found
      !
      IF ( LEN_TRIM( dftdata_fmt ) == 0 ) &
           CALL errore(subname,'No DFT dataset found',1)


      SELECT CASE ( TRIM(dftdata_fmt) )
      !
      CASE ( 'qexml' )
           !
           suffix_qe_data = ".save/data-file.xml"
           dirname = TRIM(work_dir) // '/' // TRIM(prefix) // '.save/'
           filename = TRIM( dirname) // "data-file.xml"
           !
           CALL qexml_openfile( filename, "read", IERR=ierr )
           IF ( ierr/=0) CALL errore(subname,'opening dftdata file',ABS(ierr))
           !
           CALL qexml_read_header( FORMAT_VERSION=dftdata_fmt_version, IERR=ierr)
           IF ( ierr/=0) CALL errore(subname,'no Header in dftdata file',1)
           !
           CALL qexml_closefile ( "read", IERR=ierr )
           IF ( ierr/=0) CALL errore(subname,'closing dftdata file',ABS(ierr))
           !
      CASE ( 'pw_export' )
           !
           suffix_qe_data = ".export/index.xml"
           dirname  = TRIM(work_dir) // '/' // TRIM(prefix) // '.export/'
           filename = TRIM( dirname) // "index.xml"
           !
           CALL qexpt_openfile( filename, "read", IERR=ierr )
           IF ( ierr/=0) CALL errore(subname,'opening dftdata file',ABS(ierr))
           !
           CALL qexpt_read_header( FORMAT_VERSION=dftdata_fmt_version, IERR=ierr)
           !
           IF ( ierr /= 0 ) THEN 
                !
                dftdata_fmt_version = "0.0.0" 
                read_symmetry = .FALSE.
                !
           ENDIF
           !
           CALL qexpt_closefile ( "read", IERR=ierr )
           IF ( ierr/=0) CALL errore(subname,'closing dftdata file',ABS(ierr))
           !
      CASE ( 'etsf_io' )
           !
#ifdef __ETSF_IO
           !
           dirname  = TRIM(work_dir) // '/' // TRIM(prefix) 
           filename = TRIM( dirname) // TRIM(suffix_etsf_io_data)

           CALL etsf_io_low_open_read(ncid, filename, lstat, &
                                      ERROR_DATA=error_data, &
                                      VERSION_MIN=etsf_io_version_min)
           IF (.NOT. lstat) &
               CALL etsf_error(error_data,subname,'opening '//TRIM(filename),1)
           !
           CALL etsf_io_dims_get(ncid, dims, lstat, error_data)
           IF (.NOT. lstat) CALL etsf_error(error_data,subname,'reading dims',1)
           !
           ! To be fixed
           dftdata_fmt_version="1.0.0"
           !
           CALL etsf_io_low_close(ncid, lstat, error_data)
           IF (.NOT. lstat) CALL etsf_error(error_data,subname,'closing '//TRIM(filename),1)

           !
           ! switch off pseudo readin and bcast dims
           !
           use_pseudo  = .FALSE.
           read_pseudo = .FALSE.
           !
#else
           CALL errore(subname,'ETSF_IO fmt not configured', 10 )
#endif
           !
      CASE ( 'crystal' )
           !
           filename = TRIM( dftdata_file )
           !
           CALL crio_open_file( UNIT=dft_unit, FILENAME=filename, ACTION='read', IERR=ierr )
           IF ( ierr/=0 ) CALL errore(subname, 'opening dftdata file: '//TRIM(filename), ABS(ierr) )
           !
           CALL crio_read_header( CREATOR_VERSION=dftdata_fmt_version, IERR=ierr)
           IF ( ierr/=0 ) CALL errore(subname, 'reading CRIO header', ABS(ierr) )
           !
           CALL crio_close_file( ACTION='read', IERR=ierr )
           IF ( ierr/=0) CALL errore(subname,'closing crio datafile',ABS(ierr))
           !
      CASE ( 'wannier90' )
           !
           dftdata_fmt_version = 'unknown'
           !
      CASE ( 'internal' )
           !
           dftdata_fmt_version = 'unknown'
           !
      CASE DEFAULT
           !
           CALL errore(subname,'invalid dftdata_fmt = '//TRIM(dftdata_fmt),1)
      END SELECT
      !
      !
      ! init writing to logfile if required
      !
      CALL io_name ( "log", logfile )
      CALL log_init( log_unit, use_debug_mode, logfile, debug_level )
      CALL log_push( "main" )
      !
      alloc = .TRUE.
      !
   END SUBROUTINE io_init


!**********************************************************
   SUBROUTINE io_name( data_type, filename, lpostfix, postfix_loc, lbody, body, lpath, lproc)
   !**********************************************************
   !
   IMPLICIT NONE
      !
      CHARACTER(*),            INTENT(IN)  :: data_type
      CHARACTER(*),            INTENT(OUT) :: filename
      CHARACTER(*),  OPTIONAL, INTENT(IN)  :: body, postfix_loc
      LOGICAL,       OPTIONAL, INTENT(IN)  :: lpostfix, lbody, lpath, lproc   
      !
      CHARACTER(7)        :: subname="io_name"
      LOGICAL             :: lpostfix_, lbody_, lpath_, lproc_
      CHARACTER(nstrx)    :: path_, prefix_, body_, postfix_, suffix_, proc_
      INTEGER             :: length

      !
      ! DEFAULT
      lpostfix_   = .TRUE.
      lpath_      = .TRUE.
      lproc_      = .TRUE.
      lbody_      = .FALSE.
      !
      IF ( PRESENT(lbody) )       lbody_ = lbody
      IF ( PRESENT(body) )         body_ = TRIM(body)
      IF ( PRESENT(body) )        lbody_ = .TRUE.
      IF ( PRESENT(lpostfix) ) lpostfix_ = lpostfix
      IF ( PRESENT(lpath) )       lpath_ = lpath
      IF ( PRESENT(lproc) )       lproc_ = lproc


      !
      ! setting the base name
      path_    = " " 
      body_    = " "
      postfix_ = " " 
      IF ( lpath_ )           path_ = TRIM(work_dir)
      IF ( lpostfix_ )     postfix_ = TRIM(postfix)
      IF ( lpostfix_ .AND. PRESENT( postfix_loc) ) &    
                           postfix_ = TRIM(postfix_loc)


      !
      ! add the / if needed
      length = LEN_TRIM( path_ )
      IF ( length /= 0 ) THEN
         IF ( path_(length:length) /= "/"  ) path_ = TRIM(path_)//"/"
      ENDIF

      !
      ! set data for parallelism
      proc_ = " "
      !
      IF ( nproc > 1 .AND. lproc_ ) THEN
         CALL io_set_nd_nmbr( proc_, mpime, nproc )
         proc_ = "."//TRIM(proc_)
      ENDIF

      !
      ! redefine prefix
      prefix_ = TRIM(prefix)
          

      SELECT CASE( TRIM(data_type) )
      CASE ( "dft_data" ) 
           !
           SELECT CASE ( TRIM(dftdata_fmt) ) 
           CASE ( 'qexml', 'pw_export' )
               !
               suffix_  = TRIM(suffix_qe_data)
               postfix_ = " "
               !
           CASE ( 'etsf_io' ) 
               !
               suffix_  = TRIM(suffix_etsf_io_data)
               postfix_ = " "
               !
           CASE ( 'crystal', 'wannier90', 'internal' ) 
               !
               path_    = " "
               prefix_  = " " 
               postfix_ = " "
               suffix_  = TRIM( dftdata_file )
               !
           CASE DEFAULT
               CALL errore(subname,'invalid DFTDATA_FMT = '//TRIM(dftdata_fmt),1)
           END SELECT
           !
      CASE ( "space" ) 
           !
           suffix_ = TRIM(suffix_space)
           !
      CASE ( "overlap_projection" ) 
           !
           suffix_ = TRIM(suffix_ovp)
           !
      CASE ( "translations" ) 
           !
           suffix_ = TRIM(suffix_transl)
           !
      CASE ( "wannier" ) 
           !
           suffix_ = TRIM(suffix_wannier)
           !
      CASE ( "hamiltonian", "ham" )
           !
           suffix_ = TRIM(suffix_hamiltonian) 
           !
      CASE ( "save" )
           !
           suffix_ = TRIM(suffix_save) // TRIM(proc_)
           !
      CASE ( "log" )
           !
           suffix_ = TRIM(suffix_log) // TRIM(proc_)
           IF ( lbody_ ) body_ = "debug"
           !
      CASE ( "conductance", "cond" )
           !
           body_   = "cond" 
           suffix_ = ".dat"
           !
      CASE ( "doscond" )
           !
           body_   = "doscond" 
           suffix_ = ".dat"
           !
      CASE ( "eigenchannels", "eigchn" )
           !
           body_   = "eigchn" 
           suffix_ = ".dat"
           !
      CASE ( "dos" )
           !
           body_   = "dos" 
           suffix_ = ".dat"
           !
      CASE ( "sgm" )
           !
           body_   = "sgm" 
           IF ( PRESENT(body) )  body_ = TRIM(body)
           suffix_ = TRIM(suffix_sgm) // TRIM(proc_)
           !
      CASE ( "gf" )
           !
           body_   = "greenf" 
           IF ( PRESENT(body) )  body_ = TRIM(body)
           suffix_ = TRIM(suffix_gf) // TRIM(proc_)
           !
      CASE ( "free" )
           !
           body_   = "" 
           IF ( PRESENT(body) )  body_ = TRIM(body)
           suffix_ = TRIM(proc_)
           !
      CASE DEFAULT
           CALL errore('io_name','Unknown DATA type in input: '//TRIM(data_type),1)
      END SELECT

      filename = TRIM(path_)//TRIM(prefix_)//TRIM(body_)//TRIM(postfix_)//TRIM(suffix_)

  END SUBROUTINE io_name
   

!**********************************************************
   SUBROUTINE io_open_dftdata( lserial )
   !**********************************************************
   !
   ! open the dftdata files, either for a serial or a parallel read 
   !
   IMPLICIT NONE
      !
      LOGICAL,     INTENT(IN) :: lserial
      !
      CHARACTER(15)    :: subname='io_open_dftdata'
      CHARACTER(nstrx) :: filename
      INTEGER          :: ierr
      
      
      IF ( lserial .AND. .NOT. ionode ) RETURN
      !
      CALL log_push(subname) 
      ! 
      !
      CALL io_name('dft_data',filename)
      !
      SELECT CASE ( TRIM(dftdata_fmt) )
      CASE ( 'qexml', 'pw_export' )
          !
          CALL file_open(dft_unit,TRIM(filename),PATH="/",ACTION="read", IERR=ierr )
          IF ( ierr/=0 ) CALL errore(subname, 'QEXML-PWEXP: opening '//TRIM(filename), ABS(ierr)) 
          !
      CASE ( 'etsf_io' )
          !
#ifdef __ETSF_IO
          !
          CALL etsf_io_low_open_read(ncid, TRIM(filename), lstat,  &
                                     ERROR_DATA=error_data,        &
                                     VERSION_MIN=etsf_io_version_min)   
          IF (.NOT. lstat) CALL etsf_error(error_data,subname,'ETSF_IO: opening '//TRIM(filename), 10) 
          !
#else
          CALL errore(subname,'ETSF_IO not configured',10)
#endif
          !
      CASE ( 'crystal' )
          !
          CALL crio_open_file( dft_unit, TRIM(filename), ACTION='read', IERR=ierr)
          IF ( ierr/=0 ) CALL errore(subname, 'CRIO: opening '//TRIM(filename), ABS(ierr)) 
          !
      CASE ( 'wannier90', 'internal' )
          !
          ! nothing to do
          !
      CASE DEFAULT 
          CALL errore(subname,'invalid dftdata_fmt = '//TRIM(dftdata_fmt),10)
      END SELECT
      !
      CALL log_pop(subname)
      RETURN
      !
  END SUBROUTINE io_open_dftdata

          
!**********************************************************
   SUBROUTINE io_close_dftdata( lserial )
   !**********************************************************
   !
   ! close the dftdata files, either after a serial or a parallel read 
   !
   IMPLICIT NONE
      !
      LOGICAL,     INTENT(IN) :: lserial
      !
      CHARACTER(16)    :: subname='io_close_dftdata'
      INTEGER          :: ierr
      
      
      IF ( lserial .AND. .NOT. ionode ) RETURN
      !
      CALL log_push(subname) 
      ! 
      !
      SELECT CASE ( TRIM(dftdata_fmt) )
      CASE ( 'qexml', 'pw_export' )
          !
          CALL file_close(dft_unit,PATH="/",ACTION="read", IERR=ierr )
          IF ( ierr/=0 ) CALL errore(subname, 'QEXML-PWEXP: closing DFT datafile', ABS(ierr)) 
          !
      CASE ( 'etsf_io' )
          !
#ifdef __ETSF_IO
          !
          CALL etsf_io_low_close(ncid, lstat, error_data)
          IF (.NOT. lstat) CALL etsf_error(error_data,subname,'ETSF_IO: closing DFT datafile', 10) 
          !
#else
          CALL errore(subname,'ETSF_IO not configured',10)
#endif
          !
      CASE ( 'crystal' )
          !
          CALL crio_close_file( ACTION='read', IERR=ierr)
          IF ( ierr/=0 ) CALL errore(subname, 'CRIO: closing DFT datafile', ABS(ierr)) 
          !
      CASE ( 'wannier90', 'internal' )
          !
          ! nothing to do
          !
      CASE DEFAULT 
          CALL errore(subname,'invalid dftdata_fmt = '//TRIM(dftdata_fmt),10)
      END SELECT
      !
      CALL log_pop(subname) 
      RETURN
      !
  END SUBROUTINE io_close_dftdata


!**********************************************************
   SUBROUTINE io_read_data(unit,name,prefix_,postfix_,work_dir_,title_,found)
   !**********************************************************
   IMPLICIT NONE
      INTEGER,         INTENT(in)   :: unit
      CHARACTER(*),    INTENT(in)   :: name
      LOGICAL,         INTENT(out)  :: found
      CHARACTER(*),    INTENT(out)  :: prefix_, postfix_, work_dir_, title_ 

      CHARACTER(12)                 :: sub_name='io_read_data'
      CHARACTER(nstrx)              :: attr
      INTEGER                       :: ierr

      CALL iotk_scan_empty(unit,name,FOUND=found,ATTR=attr,IERR=ierr)
      IF ( .NOT. found ) RETURN
      IF ( ierr > 0 ) CALL errore(sub_name,'Wrong format in tag '//TRIM(name),ierr)
      found = .TRUE.

      CALL iotk_scan_attr(attr,'prefix',prefix_,IERR=ierr)
      IF (ierr /= 0) CALL errore(sub_name,'Wrong input format in PREFIX',ABS(ierr))
      !
      CALL iotk_scan_attr(attr,'postfix',postfix_,IERR=ierr)
      IF (ierr /= 0) CALL errore(sub_name,'Wrong input format in POSTFIX',ABS(ierr))
      !
      CALL iotk_scan_attr(attr,'work_dir',work_dir_,IERR=ierr)
      IF (ierr /= 0) CALL errore(sub_name,'Wrong input format in WORK_DIR',ABS(ierr))
      !
      CALL iotk_scan_attr(attr,'title',title_,IERR=ierr)
      IF (ierr /= 0) CALL errore(sub_name,'Wrong input format in TITLE',ABS(ierr))

   END SUBROUTINE io_read_data


!**********************************************************
   SUBROUTINE io_write_data(unit,name,prefix_,postfix_,work_dir_,title_)
   !**********************************************************
      IMPLICIT NONE
      INTEGER,         INTENT(in)   :: unit
      CHARACTER(*),    INTENT(in)   :: name
      CHARACTER(*),    INTENT(in)   :: prefix_, postfix_, work_dir_, title_ 

      CHARACTER(nstrx)              :: attr

      CALL iotk_write_attr(attr,"prefix",TRIM(prefix_),FIRST=.TRUE.)
      CALL iotk_write_attr(attr,"postfix",TRIM(postfix_),FIRST=.TRUE.)
      CALL iotk_write_attr(attr,"work_dir",TRIM(work_dir_))
      CALL iotk_write_attr(attr,"title",TRIM(title_))
      CALL iotk_write_empty(unit,name,ATTR=attr)

   END SUBROUTINE io_write_data

!
! Copyright (C) 2001-2008 Quantum-Espresso group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!

!**********************************************************
   SUBROUTINE io_set_nd_nmbr( nd_nmbr, node_number, nproc_image )
   !**********************************************************
     !
     IMPLICIT NONE
     !
     CHARACTER(LEN=6), INTENT(OUT) :: nd_nmbr
     INTEGER, INTENT(IN) :: node_number
     INTEGER, INTENT(IN) :: nproc_image
     !
     INTEGER :: nmax, nleft, nfact, n
     !
     nd_nmbr = '      '
     nmax = INT ( LOG10 ( nproc_image + 1.0D-8 ) )
     !
     ! nmax+1=number of digits of nproc_image (number of processors)
     ! 1.0D-8 protects from rounding error if nproc_image is a power of 10
     !
     IF ( nmax+1 > LEN (nd_nmbr) ) &
        CALL errore ( "io_set_nd_nmbr", 'insufficient size for nd_nmbr', nmax)
     IF ( nmax < 0) &
        CALL errore ( "io_set_nd_nmbr", 'incorrect value for nproc_image', nmax)
     !
     nleft = node_number
     !
     DO n = nmax, 0, -1
        !
        ! decompose node_number (index of this process) into powers of 10:
        !    node_number = i*10^nmax+j*10^(nmax-1)+k*10^(nmax-2)...
        ! i,j,k,... can be equal to 0
        !
        nfact = INT ( nleft/10**n )
        IF ( nfact > 9 ) CALL errore ( "io_set_nd_nmbr", 'internal error', 1 )
        nleft = nleft - nfact*10**n
        !
        WRITE( nd_nmbr(nmax-n+1:nmax-n+1), '(I1)' ) nfact
        !
     END DO
     !
     IF ( nleft > 0 ) CALL errore ( "io_set_nd_nmbr", 'internal error', 2 )
     !
     RETURN
     !
  END SUBROUTINE io_set_nd_nmbr
  !
END MODULE io_module

