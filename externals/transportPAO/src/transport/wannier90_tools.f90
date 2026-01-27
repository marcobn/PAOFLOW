!
! Copyright (C) 2009 WanT Group, 2017 ERMES Group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License\'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
! 
! Important contributions to this module have been given
! by Tonatiuh Rangel Gordillo
!
!*********************************************
   MODULE wannier90_tools_module
!*********************************************
   !
   USE kinds,              ONLY : dbl
   USE constants,          ONLY : BOHR => bohr_radius_angs, ZERO, ONE, &
                                  CZERO, CONE, TWO, RYD
   USE parameters,         ONLY : nstrx
   USE timing_module,      ONLY : timing
   USE log_module,         ONLY : log_push, log_pop
   USE converters_module,  ONLY : cart2cry, cry2cart
   USE parser_module,      ONLY : change_case
   USE grids_module,       ONLY : grids_get_rgrid
   USE iotk_module
   !
   IMPLICIT NONE 
   PRIVATE
   SAVE

   !
   ! global variables of the module
   !
   CHARACTER(nstrx)   :: prefix
   CHARACTER(nstrx)   :: file_chk
   CHARACTER(nstrx)   :: file_eig
   !
   LOGICAL            :: init = .FALSE.

   !
   ! contains:
   ! SUBROUTINE  wannier90_tools_init( prefix_ )
   ! SUBROUTINE  wannier90_tools_get_prefix( filein, prefix_ )
   ! SUBROUTINE  wannier90_tools_get_dims( [nbnd, nkpts, dimwann] )
   ! SUBROUTINE  wannier90_tools_get_eig(  nbnd, nkpts, eig )
   ! SUBROUTINE  wannier90_tools_get_data( nbnd, nkpts, dimwann, ... )
   ! SUBROUTINE  wannier90_tools_get_lattice( alat, avec, bvec )
   ! SUBROUTINE  wannier90_tools_get_kpoints( nkpts[, vkpt, wk, bvec] )
   ! SUBROUTINE  wannier90_to_internal( filein, fileout, filetype )
   ! FUNCTION    file_is_wannier90( filein )
   !
   PUBLIC :: wannier90_tools_get_dims
   PUBLIC :: wannier90_tools_get_eig
   PUBLIC :: wannier90_tools_get_data
   PUBLIC :: wannier90_tools_get_lattice
   PUBLIC :: wannier90_tools_get_kpoints
   PUBLIC :: wannier90_to_internal
   PUBLIC :: file_is_wannier90

CONTAINS


!**********************************************************
   SUBROUTINE wannier90_tools_init( prefix_ )
   !**********************************************************
   !
   ! define module global variables
   !
   IMPLICIT NONE
   CHARACTER(*),   INTENT(IN) :: prefix_ 
   !
   prefix   = TRIM( prefix_ )
   file_chk = TRIM(prefix)//'.chk'
   file_eig = TRIM(prefix)//'.eig'
   !
   init     = .TRUE.
   !
END SUBROUTINE wannier90_tools_init
   

!**********************************************************
   SUBROUTINE wannier90_tools_get_prefix( filein, prefix_ )
   !**********************************************************
   !
   ! extract the prefix (basename) of the input file.
   ! If the extension of the file is not ".chk" an
   ! empty prefix is issued
   !
   IMPLICIT NONE
   CHARACTER(*),   INTENT(IN)  :: filein
   CHARACTER(*),   INTENT(OUT) :: prefix_ 
   !
   INTEGER      :: ilen
   CHARACTER(4) :: suffix='.chk'
   !
   prefix_  = ' '
   !
   ilen = LEN_TRIM( filein )
   !
   IF ( filein(ilen-3:ilen) == suffix ) THEN
       !
       prefix_ = filein(1:ilen-4)
       !
   ENDIF
   !
END SUBROUTINE wannier90_tools_get_prefix
   

!**********************************************************
   SUBROUTINE wannier90_tools_get_dims( nbnd, nkpts, dimwann )
   !**********************************************************
   !
   ! get the dimensions of the problem
   ! need to have the module initialized
   !
   IMPLICIT NONE
   ! 
   INTEGER, OPTIONAL, INTENT(OUT) :: nbnd, nkpts, dimwann
   !
   CHARACTER(24) :: subname='wannier90_tools_get_dims'
   LOGICAL       :: lread
   INTEGER       :: nkpts_, dimwann_, iunit
   INTEGER       :: i_old, i_new, j, ierr
   
   CALL log_push( subname )
   !
   IF ( .NOT. init ) CALL errore(subname,'module not init',10)
   CALL iotk_free_unit( iunit )

   !
   ! check kind compatibility
   !
   IF ( dbl /= KIND(1.0D0)) CALL errore(subname,'internal dbl kind incompatible',10)

   !
   ! get dimensions from .chk
   !
   OPEN( iunit, FILE=file_chk, STATUS='old', FORM='unformatted', IOSTAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'opening '//TRIM(file_chk),ABS(ierr) )
   !
   READ(iunit, IOSTAT=ierr) 
   IF ( ierr/=0 ) CALL errore(subname,'skipping header',ABS(ierr) )
   !
   READ(iunit, IOSTAT=ierr) 
   IF ( ierr/=0 ) CALL errore(subname,'skipping lattice',ABS(ierr) )
   READ(iunit, IOSTAT=ierr) 
   IF ( ierr/=0 ) CALL errore(subname,'skipping recipr lattice',ABS(ierr) )
   !
   READ(iunit, IOSTAT=ierr) nkpts_
   IF ( ierr/=0 ) CALL errore(subname,'reading nkpts',ABS(ierr) )
   !
   READ(iunit, IOSTAT=ierr) 
   IF ( ierr/=0 ) CALL errore(subname,'skipping vkpt',ABS(ierr) )
   !
   READ(iunit, IOSTAT=ierr) 
   IF ( ierr/=0 ) CALL errore(subname,'skipping nntot',ABS(ierr) )
   READ(iunit, IOSTAT=ierr) dimwann_
   IF ( ierr/=0 ) CALL errore(subname,'reading dimwann',ABS(ierr) )
   !
   CLOSE( iunit, IOSTAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'closing '//TRIM(file_chk),ABS(ierr) )
   !
   !
   IF ( PRESENT( nkpts ) )      nkpts = nkpts_
   IF ( PRESENT( dimwann ) )  dimwann = dimwann_
  
   !
   ! get dimensions from .eig
   !
   IF ( PRESENT( nbnd ) ) THEN
       !
       OPEN( iunit, FILE=file_eig, STATUS='old', FORM='formatted', IOSTAT=ierr )
       IF ( ierr/=0 ) CALL errore(subname,'opening '//TRIM(file_eig),ABS(ierr) )
       !
       lread = .TRUE.
       i_old = 0
       i_new = 0
       DO WHILE( lread )
           !
           i_old = i_new
           READ(iunit, *, IOSTAT=ierr) i_new, j
           IF ( ierr/=0 ) CALL errore(subname,'reading indexes', ABS(ierr) )
           !
           IF ( j == 2 ) lread = .FALSE.
           !
       ENDDO
       !
       nbnd = i_old
       !
       !
       CLOSE( iunit, IOSTAT=ierr )
       IF ( ierr/=0 ) CALL errore(subname,'closing '//TRIM(file_eig),ABS(ierr) )
       !
   ENDIF
   !
   CALL log_pop( subname )
   RETURN
   !
END SUBROUTINE wannier90_tools_get_dims


!**********************************************************
   SUBROUTINE wannier90_tools_get_eig( nbnd, nkpts, eig )
   !**********************************************************
   !
   ! read eigenvalues
   !
   IMPLICIT NONE
   ! 
   INTEGER,   INTENT(IN)   :: nbnd, nkpts
   REAL(dbl), INTENT(OUT)  :: eig(nbnd,nkpts)
   !
   CHARACTER(23) :: subname='wannier90_tools_get_eig'
   INTEGER       :: iunit
   INTEGER       :: i, j, ib, ik, ierr
   
   CALL log_push( subname )
   !
   IF ( .NOT. init ) CALL errore(subname,'module not init',10)
   CALL iotk_free_unit( iunit )

   !
   ! get eigenvalues from .eig
   !
   OPEN( iunit, FILE=file_eig, STATUS='old', FORM='formatted', IOSTAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'opening '//TRIM(file_eig),ABS(ierr) )
   !
   DO ik = 1, nkpts
   DO ib = 1, nbnd
       !
       READ(iunit, *, IOSTAT=ierr) i, j, eig(ib,ik)
       IF ( ierr/=0 ) CALL errore(subname,'reading eig', ABS(ierr) )
       !
       IF ( ib /= i .OR. ik /= j) &
              CALL errore(subname,'something went bananas about indexes', 71)
       !
   ENDDO
   ENDDO
   !
   !
   CLOSE( iunit, IOSTAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'closing '//TRIM(file_eig),ABS(ierr) )
   !
   CALL log_pop( subname )
   RETURN
   !
END SUBROUTINE wannier90_tools_get_eig

   
!**********************************************************
   SUBROUTINE wannier90_tools_get_data( nbnd,  nkpts, dimwann,         &
                                        avec, bvec, vkpt, wk, lwindow, &
                                        ndimwin, u_matrix_opt, u_matrix )
   !**********************************************************
   !
   ! read main data from wannier90 files
   !
   IMPLICIT NONE
   ! 
   INTEGER,      INTENT(IN)   :: nbnd, nkpts, dimwann
   REAL(dbl),    INTENT(OUT)  :: avec(3,3), bvec(3,3), vkpt(3,*), wk(*)
   LOGICAL,      INTENT(OUT)  :: lwindow(nbnd,*)
   INTEGER,      INTENT(OUT)  :: ndimwin(*)
   COMPLEX(dbl), INTENT(OUT)  :: u_matrix_opt(nbnd,dimwann,*),  &
                                 u_matrix(dimwann,dimwann,*)
   !
   CHARACTER(24) :: subname='wannier90_tools_get_data'
   !CHARACTER(20) :: checkpoint
   INTEGER       :: iunit
   INTEGER       :: nkpts_, dimwann_
   LOGICAL       :: have_disentangle
   INTEGER       :: i, ik, ierr
   

   CALL log_push( subname )
   !
   IF ( .NOT. init ) CALL errore(subname,'module not init',10)
   CALL iotk_free_unit( iunit )

   !
   ! check kind compatibility
   !
   IF ( dbl /= KIND(1.0D0)) CALL errore(subname,'internal dbl kind incompatible',10)

   !
   ! get data from .chk
   !
   OPEN( iunit, FILE=file_chk, STATUS='old', FORM='unformatted', IOSTAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'opening '//TRIM(file_chk),ABS(ierr) )
   !
   READ(iunit, IOSTAT=ierr)
   IF ( ierr/=0 ) CALL errore(subname,'skipping header',ABS(ierr) )
   !
   READ(iunit, IOSTAT=ierr)  avec(1:3,1:3)
   IF ( ierr/=0 ) CALL errore(subname,'reading lattice',ABS(ierr) )
   READ(iunit, IOSTAT=ierr)  bvec(1:3,1:3)
   IF ( ierr/=0 ) CALL errore(subname,'reading recipr lattice',ABS(ierr) )
   !
   READ(iunit, IOSTAT=ierr) nkpts_
   IF ( ierr/=0 ) CALL errore(subname,'reading nkpts',ABS(ierr) )
   IF ( nkpts_ /= nkpts ) CALL errore(subname,'wrong nkpts from file',10)
   !
   READ(iunit, IOSTAT=ierr) vkpt(1:3,1:nkpts)
   IF ( ierr/=0 ) CALL errore(subname,'reading vkpt',ABS(ierr) )
   !
   wk(1:nkpts) = ONE
   !
   READ(iunit, IOSTAT=ierr) 
   IF ( ierr/=0 ) CALL errore(subname,'skipping nntot',ABS(ierr) )
   !
   READ(iunit, IOSTAT=ierr) dimwann_
   IF ( ierr/=0 ) CALL errore(subname,'reading dimwann',ABS(ierr) )
   IF ( dimwann_ /= dimwann ) CALL errore(subname,'wrong dimwann from file',10)
   !
   READ(iunit, IOSTAT=ierr) 
   IF ( ierr/=0 ) CALL errore(subname,'skipping checkpoint',ABS(ierr) )
   !
   READ(iunit, IOSTAT=ierr) have_disentangle
   IF ( ierr/=0 ) CALL errore(subname,'reading have_disentangle',ABS(ierr) )
   !
   IF ( have_disentangle ) THEN
       !
       READ(iunit, IOSTAT=ierr) 
       IF ( ierr/=0 ) CALL errore(subname,'skipping omega_invariant',ABS(ierr) )
       !
       READ(iunit, IOSTAT=ierr) lwindow(1:nbnd,1:nkpts)
       IF ( ierr/=0 ) CALL errore(subname,'reading lwindow',ABS(ierr) )
       !
       READ(iunit, IOSTAT=ierr) ndimwin(1:nkpts)
       IF ( ierr/=0 ) CALL errore(subname,'reading ndimwin',ABS(ierr) )
       !
       READ(iunit, IOSTAT=ierr) u_matrix_opt(1:nbnd,1:dimwann,1:nkpts)
       IF ( ierr/=0 ) CALL errore(subname,'reading u_matrix_opt',ABS(ierr) )
       !
   ELSE
       !
       IF ( nbnd /= dimwann) CALL errore(subname,'nbnd /= dimwann, case not impl',10)
       !
       lwindow(1:nbnd,1:nkpts)   = .TRUE.
       ndimwin(1:nkpts)          = nbnd
       !
       u_matrix_opt(:,:,1:nkpts) = CZERO
       !
       DO ik = 1, nkpts
       DO i  = 1, dimwann
           u_matrix_opt(i,i,ik) = CONE
       ENDDO
       ENDDO
       !
   ENDIF
   !
   READ(iunit, IOSTAT=ierr) u_matrix(1:dimwann,1:dimwann,1:nkpts)
   IF ( ierr/=0 ) CALL errore(subname,'reading u_matrix',ABS(ierr) )
   
   !
   ! units conversion 
   !
   ! avec:  is given in Angs     -->   bohr
   ! bvec:  is given in Ang^-1   -->   bohr^-1
   ! eig:   is given in eV       -->   eV
   ! vkpt:  are given in cryst 
   !
   avec = avec / bohr
   bvec = bvec * bohr
   !
   !
   CLOSE( iunit, IOSTAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'closing '//TRIM(file_chk),ABS(ierr) )
   !
   CALL log_pop( subname )
   RETURN
   !
END SUBROUTINE wannier90_tools_get_data


!**********************************************************
   SUBROUTINE wannier90_tools_get_lattice( alat, avec, bvec )
   !**********************************************************
   !
   ! read lattice data from wannier90 files
   !
   IMPLICIT NONE
   ! 
   REAL(dbl),    INTENT(OUT)  :: alat, avec(3,3), bvec(3,3)
   !
   CHARACTER(27) :: subname='wannier90_tools_get_lattice'
   INTEGER       :: iunit
   INTEGER       :: ierr
   

   CALL log_push( subname )
   !
   IF ( .NOT. init ) CALL errore(subname,'module not init',10)
   CALL iotk_free_unit( iunit )
   
   !
   ! check kind compatibility
   !
   IF ( dbl /= KIND(1.0D0)) CALL errore(subname,'internal dbl kind incompatible',10)

   !
   ! get data from .chk
   !
   OPEN( iunit, FILE=file_chk, STATUS='old', FORM='unformatted', IOSTAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'opening '//TRIM(file_chk),ABS(ierr) )
   !
   READ(iunit, IOSTAT=ierr)
   IF ( ierr/=0 ) CALL errore(subname,'skipping header',ABS(ierr) )
   !
   READ(iunit, IOSTAT=ierr)  avec(1:3,1:3)
   IF ( ierr/=0 ) CALL errore(subname,'reading lattice',ABS(ierr) )
   READ(iunit, IOSTAT=ierr)  bvec(1:3,1:3)
   IF ( ierr/=0 ) CALL errore(subname,'reading recipr lattice',ABS(ierr) )

   !
   ! move to bohr and bohr^-1 units
   !
   avec = avec / bohr
   bvec = bvec * bohr

   !
   ! init alat
   !
   alat = DOT_PRODUCT( avec(:,1), avec(:,1) )
   alat = SQRT( alat )
   !
   CLOSE( iunit, IOSTAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'closing '//TRIM(file_chk),ABS(ierr) )
   !
   CALL log_pop( subname )
   RETURN
   !
END SUBROUTINE wannier90_tools_get_lattice


!**********************************************************
   SUBROUTINE wannier90_tools_get_kpoints( nkpts, vkpt, wk, bvec )
   !**********************************************************
   !
   ! read kpts data. vkpt is in crystal units
   !
   IMPLICIT NONE
   ! 
   INTEGER,             INTENT(IN)  :: nkpts
   REAL(dbl), OPTIONAL, INTENT(OUT) :: vkpt(3,nkpts)
   REAL(dbl), OPTIONAL, INTENT(OUT) :: wk(nkpts)
   REAL(dbl), OPTIONAL, INTENT(OUT) :: bvec(3,3)
   !
   CHARACTER(27) :: subname='wannier90_tools_get_kpoints'
   REAL(dbl)     :: bvec_(3,3)
   INTEGER       :: iunit, nkpts_
   INTEGER       :: ierr
   
   CALL log_push( subname )
   !
   IF ( .NOT. init ) CALL errore(subname,'module not init',10)
   CALL iotk_free_unit( iunit )

   !
   ! check kind compatibility
   !
   IF ( dbl /= KIND(1.0D0)) CALL errore(subname,'internal dbl kind incompatible',10)

   !
   ! get data from .chk
   !
   OPEN( iunit, FILE=file_chk, STATUS='old', FORM='unformatted', IOSTAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'opening '//TRIM(file_chk),ABS(ierr) )
   !
   READ(iunit, IOSTAT=ierr) 
   IF ( ierr/=0 ) CALL errore(subname,'skipping header',ABS(ierr) )
   !
   READ(iunit, IOSTAT=ierr) 
   IF ( ierr/=0 ) CALL errore(subname,'skipping lattice',ABS(ierr) )
   !
   READ(iunit, IOSTAT=ierr) bvec_(1:3,1:3)
   IF ( ierr/=0 ) CALL errore(subname,'reading recipr lattice',ABS(ierr) )
   !
   READ(iunit, IOSTAT=ierr) nkpts_
   IF ( ierr/=0 ) CALL errore(subname,'reading nkpts',ABS(ierr) )
   !
   IF ( nkpts_ /= nkpts ) CALL errore(subname,'invalid nkpts',10)
   !
   IF ( PRESENT( vkpt ) ) THEN
       READ(iunit, IOSTAT=ierr) vkpt(1:3,1:nkpts)
       IF ( ierr/=0 ) CALL errore(subname,'reading vkpt',ABS(ierr) )
   ENDIF
   !
   CLOSE( iunit, IOSTAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'closing '//TRIM(file_chk),ABS(ierr) )
   ! 
   ! init wk
   !
   IF ( PRESENT( wk ) ) THEN
       wk(1:nkpts) = ONE / REAL(nkpts, dbl)
   ENDIF
   !
   bvec_ = bvec_ * bohr
   !
   IF ( PRESENT( bvec) )    bvec = bvec_
   !
   CALL log_pop( subname )
   RETURN
   !
END SUBROUTINE wannier90_tools_get_kpoints


!**********************************************************
   SUBROUTINE wannier90_to_internal( filein, fileout, filetype )
   !**********************************************************
   !
   ! Convert the datafile written by the wannier90 program to
   ! the internal representation.
   !
   ! FILETYPE values are:
   !  - ham, hamiltonian
   !  - space, subspace
   !
   IMPLICIT NONE

   LOGICAL, PARAMETER :: binary = .TRUE.
   
   !
   ! input variables
   !
   CHARACTER(*), INTENT(IN) :: filein
   CHARACTER(*), INTENT(IN) :: fileout
   CHARACTER(*), INTENT(IN) :: filetype
   
   !
   ! local variables
   !
   INTEGER                     :: ounit
   !
   CHARACTER(21)               :: subname="wannier90_to_internal"
   !
   CHARACTER(nstrx)            :: attr, filetype_, prefix_
   INTEGER                     :: dimwann, nkpts, nk(3), shift(3), nrtot, nr(3)
   INTEGER                     :: nspin, nbnd, dimwinx 
   INTEGER                     :: ierr, i, j, m, ir, ik, isp
   !
   LOGICAL                     :: write_ham, write_space
   !
   REAL(dbl)                   :: avec(3,3), bvec(3,3)
   REAL(dbl)                   :: norm, arg, fact
   COMPLEX(dbl)                :: phase
   LOGICAL,        ALLOCATABLE :: lwindow(:,:)
   INTEGER,        ALLOCATABLE :: ndimwin(:), imin(:), imax(:), ivr(:,:)
   REAL(dbl),      ALLOCATABLE :: vkpt_cry(:,:), vkpt(:,:), wk(:), wr(:), vr(:,:)
   REAL(dbl),      ALLOCATABLE :: eig(:,:,:), eig_aux(:), eig_wan(:,:,:)
   COMPLEX(dbl),   ALLOCATABLE :: rham(:,:,:,:)
   COMPLEX(dbl),   ALLOCATABLE :: kham(:,:,:,:)
   COMPLEX(dbl),   ALLOCATABLE :: u_matrix_opt(:,:,:), u_matrix(:,:,:)

!
!------------------------------
! main body
!------------------------------
!
   CALL timing( subname, OPR='start' )
   CALL log_push( subname )


   !
   ! search for units indipendently of io_module
   !
   CALL iotk_free_unit( ounit )

   !
   ! select the operation to do
   !
   write_ham     = .FALSE.
   write_space   = .FALSE.
   !
   filetype_ = TRIM( filetype )
   CALL change_case( filetype_, 'lower' )
   !
   SELECT CASE( TRIM( filetype_ ) )
   !
   CASE( 'ham', 'hamiltonian' )
      write_ham   = .TRUE.
   CASE( 'space', 'subspace' )
      write_space = .TRUE.
   CASE DEFAULT
      CALL errore(subname, 'invalid filetype: '//TRIM(filetype_), 71 )
   END SELECT


!
!---------------------------------
! read from filein (WANNIER90 fmt)
!---------------------------------
!

   CALL wannier90_tools_get_prefix( TRIM(filein), prefix_ )
   CALL wannier90_tools_init( prefix_ )

   CALL wannier90_tools_get_dims( nbnd, nkpts, dimwann )
   !
   ! assume spin unpolarized calculation
   !
   nspin = 1
   isp   = 1
   !
   !WRITE(0,*) "Reading dims"
   !WRITE(0,*) "     nbnd : ", nbnd
   !WRITE(0,*) "    nkpts : ", nkpts
   !WRITE(0,*) "  dimwann : ", dimwann

   !
   ! allocate main quantities to be readed
   !
   ALLOCATE( vkpt_cry(3, nkpts), STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'allocating vkpt_cry', ABS(ierr) )
   ALLOCATE( vkpt(3, nkpts), STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'allocating vkpt', ABS(ierr) )
   !
   ALLOCATE( wk(nkpts), STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'allocating wk', ABS(ierr) )
   !
   ALLOCATE( lwindow(nbnd,nkpts), STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'allocating lwindow', ABS(ierr) )
   !
   ALLOCATE( ndimwin(nkpts), STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'allocating ndimwin', ABS(ierr) )
   !
   ALLOCATE( u_matrix_opt(nbnd,dimwann,nkpts), STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'allocating u_matrix_opt', ABS(ierr) )
   !
   ALLOCATE( u_matrix(dimwann,dimwann,nkpts), STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'allocating u_matrix', ABS(ierr) )
   !
   ALLOCATE( eig(nbnd, nkpts, nspin), STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname,'allocating eig', ABS(ierr))
   !
   ALLOCATE( imin(nkpts), imax(nkpts), STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'allocating imin, imax', ABS(ierr) )
  

   !
   ! read main dataset
   !
   CALL wannier90_tools_get_eig ( nbnd, nkpts, eig(:,:,isp) )
   !
   CALL wannier90_tools_get_data( nbnd, nkpts, dimwann, &
         avec, bvec, vkpt_cry, wk, lwindow, ndimwin, u_matrix_opt, u_matrix )

   !
   ! units are converted to internal units during read-in.
   ! Now:  avec, bvec    in bohr and bohr^-1
   !       vkpt            in crystal units
   !       eig             eV
   !
   vkpt = vkpt_cry
   CALL cry2cart( vkpt, bvec )

   !
   ! initialization
   !
   dimwinx = MAXVAL( ndimwin(:) )
   !
   DO ik = 1, nkpts
       !
       imin(ik) = 0
       imax(ik) = nbnd
       !
       ! In the following we assume that imin and imax are
       ! set according to a connected window, i.e. all the
       ! states between imin and imax are part of the window.
       ! We check this and, if the condition is not fulfilled,
       ! and error is issued.
       !
       DO i = 1, nbnd
           IF ( lwindow(i,ik) ) THEN 
               imin(ik) = i
               EXIT
           ENDIF
       ENDDO
       !
       DO i = nbnd, 1, -1
           IF ( lwindow(i,ik) ) THEN 
               imax(ik) = i
               EXIT
           ENDIF
       ENDDO
       !
       ! check
       !
       IF ( ANY( .NOT. lwindow(imin(ik):imax(ik),ik) ) ) &
           CALL errore(subname,'e-window not simply connected',10)
       !
   ENDDO

   !
   ! check kpts are given as a monkhorst-pack grid
   !
   CALL get_monkpack( nk, shift, nkpts, vkpt_cry, "crystal", bvec, ierr )
   IF ( ierr/=0 ) CALL errore(subname,'kpt grid not Monkhorst-Pack',ABS(ierr))

   !
   ! check the normalization of the weights
   !
   norm   = SUM( wk )
   wk(:)  = wk(:) / norm


   !
   !
   ! real-space lattice vectors
   !
   nr(1:3) = nk(1:3)
   !
   CALL grids_get_rgrid(nr, NRTOT=nrtot )
   !
   ALLOCATE( ivr(3, nrtot), vr(3, nrtot), STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'allocating ivr, vr', ABS(ierr) )
   !
   ALLOCATE( wr(nrtot), STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'allocating wr', ABS(ierr) )
   !
   CALL grids_get_rgrid(nr, WR=wr, IVR=ivr )

   vr(:,:) = REAL( ivr, dbl)
   CALL cry2cart( vr, avec )
   

   !
   ! define the Hamiltonian on the WF basis
   !
   IF ( write_ham ) THEN
       ALLOCATE( rham(dimwann, dimwann, nrtot, nspin), STAT=ierr )
       IF ( ierr/=0 ) CALL errore(subname, 'allocating rham', ABS(ierr) )
   ENDIF
   !
   ALLOCATE( kham(dimwann, dimwann, nkpts, nspin), STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'allocating kham', ABS(ierr) )
   !
   ALLOCATE( eig_aux(dimwinx), STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'allocating eig_aux', ABS(ierr) )
   ALLOCATE( eig_wan(dimwann,nkpts,nspin), STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'allocating eig_wan', ABS(ierr) )
   
   !
   ! for each kpt and spin, 
   !  * get the eigenvalues inside the selected energy window
   !  * apply the rotation due to disentanglement
   !  * apply the rotation due to the wannierization
   !
   DO isp = 1, nspin 
   DO ik  = 1, nkpts

       !
       ! set the eigenvalues
       !
       j = 0
       !
       DO i = 1, nbnd
           ! 
           IF ( lwindow(i,ik) ) THEN
               j = j+1
               eig_aux(j) = eig(i,ik,isp)
           ENDIF
           !
       ENDDO

       !
       ! apply the rotation due to disentanglement
       !
       DO j = 1, dimwann
           !
           eig_wan(j,ik,isp) = ZERO
           !
           DO m = 1, ndimwin(ik)
               !
               eig_wan(j,ik,isp) = eig_wan(j,ik,isp) + eig_aux(m) * REAL( &
                            CONJG( u_matrix_opt(m,j,ik)) * u_matrix_opt(m,j,ik), dbl)
               !
           ENDDO
           !
       ENDDO
       
       !
       ! apply the rotation due to wannierization
       !
       DO j = 1, dimwann
       DO i = 1, j
           !
           kham(i,j,ik,isp) = CZERO
           !
           DO m = 1, dimwann
               !
               kham(i,j,ik,isp) = kham(i,j,ik,isp) + eig_wan(m,ik,isp) * &
                         CONJG(u_matrix(m,i,ik)) * u_matrix(m,j,ik)
               !
           ENDDO
           !
           IF ( i < j ) kham(j,i,ik,isp) = CONJG( kham(i,j,ik,isp) )
           !
       ENDDO
       ENDDO
       !
   ENDDO
   ENDDO

   IF ( write_ham ) THEN

       !
       ! apply the transformation from k to R space
       !
       DO isp = 1, nspin
           !
           ! 
           ! Fourier transform it: H_ij(k) --> H_ij(R) = (1/N_kpts) sum_k e^{-ikR}
           ! H_ij(k)
           !
           fact = ONE / REAL(nkpts, dbl)
           !
           DO ir = 1, nrtot
               !
               DO j = 1, dimwann
               DO i = 1, dimwann
                   !
                   rham(i,j,ir,isp) = CZERO
                   !
                   DO ik = 1, nkpts
                       !
                       arg = DOT_PRODUCT( vkpt(:,ik), vr(:,ir) )
                       phase = CMPLX( COS(arg), -SIN(arg), dbl )
                       !
                       rham(i,j,ir,isp) = rham(i,j,ir,isp) + phase * kham(i,j,ik,isp)
                       !
                   ENDDO
                   !
                   rham(i,j,ir,isp) = rham(i,j,ir,isp) * fact
                   !
               ENDDO
               ENDDO
               !   
           ENDDO
           !
       ENDDO
       ! 
       ! 
       DEALLOCATE( kham, STAT=ierr )
       IF ( ierr/=0 ) CALL errore(subname, 'deallocating kham', ABS(ierr) )
       !
       DEALLOCATE( eig_aux, STAT=ierr )
       IF ( ierr/=0 ) CALL errore(subname, 'deallocating eig_aux', ABS(ierr) )
       !
   ENDIF


!
!---------------------------------
! write to fileout (internal fmt)
!---------------------------------
!
   IF ( write_ham ) THEN
       !
       CALL iotk_open_write( ounit, FILE=TRIM(fileout), BINARY=binary )
       CALL iotk_write_begin( ounit, "HAMILTONIAN" )
       !
       !
       CALL iotk_write_attr( attr,"dimwann",dimwann,FIRST=.TRUE.)
       CALL iotk_write_attr( attr,"nkpts",nkpts)
       CALL iotk_write_attr( attr,"nspin",nspin)
       CALL iotk_write_attr( attr,"nk",nk)
       CALL iotk_write_attr( attr,"shift",shift)
       CALL iotk_write_attr( attr,"nrtot",nrtot)
       CALL iotk_write_attr( attr,"nr",nr)
       CALL iotk_write_attr( attr,"have_overlap", .FALSE. )
       CALL iotk_write_attr( attr,"fermi_energy", 0.0 )
       CALL iotk_write_empty( ounit,"DATA",ATTR=attr)
       !
       CALL iotk_write_attr( attr,"units","bohr",FIRST=.TRUE.)
       CALL iotk_write_dat( ounit,"DIRECT_LATTICE", avec, ATTR=attr, COLUMNS=3)
       !
       CALL iotk_write_attr( attr,"units","bohr^-1",FIRST=.TRUE.)
       CALL iotk_write_dat( ounit,"RECIPROCAL_LATTICE", bvec, ATTR=attr, COLUMNS=3)
       !
       CALL iotk_write_attr( attr,"units","crystal",FIRST=.TRUE.)
       CALL iotk_write_dat( ounit,"VKPT", vkpt_cry, ATTR=attr, COLUMNS=3)
       CALL iotk_write_dat( ounit,"WK", wk)
       !
       CALL iotk_write_dat( ounit,"IVR", ivr, ATTR=attr, COLUMNS=3)
       CALL iotk_write_dat( ounit,"WR", wr)
       !
       !
       spin_loop: & 
       DO isp = 1, nspin
          !
          IF ( nspin == 2 ) THEN
              !
              CALL iotk_write_begin( ounit, "SPIN"//TRIM(iotk_index(isp)) )
              !
          ENDIF
          !
          CALL iotk_write_begin( ounit,"RHAM")
          !
          DO ir = 1, nrtot
              !
              CALL iotk_write_dat( ounit,"VR"//TRIM(iotk_index(ir)), rham( :, :, ir, isp) )
              !
          ENDDO
          !
          CALL iotk_write_end( ounit,"RHAM")
          !
          IF ( nspin == 2 ) THEN
              !
              CALL iotk_write_end( ounit, "SPIN"//TRIM(iotk_index(isp)) )
              !
          ENDIF
          !
       ENDDO spin_loop
       !
       CALL iotk_write_end( ounit, "HAMILTONIAN" )
       CALL iotk_close_write( ounit )
       !
   ENDIF

   IF ( write_space ) THEN
       !
       CALL iotk_open_write( ounit, FILE=TRIM(fileout), BINARY=binary )
       !
       !
       CALL iotk_write_begin( ounit, "WINDOWS" )
       !
       !
       CALL iotk_write_attr( attr,"nbnd",nbnd,FIRST=.TRUE.)
       CALL iotk_write_attr( attr,"nkpts",nkpts)
       CALL iotk_write_attr( attr,"nspin",nspin)
       CALL iotk_write_attr( attr,"spin_component","none")
       CALL iotk_write_attr( attr,"efermi", 0.0 )
       CALL iotk_write_attr( attr,"dimwinx", dimwinx )
       CALL iotk_write_empty( ounit,"DATA",ATTR=attr)
       !
       CALL iotk_write_dat( ounit, "DIMWIN", ndimwin, COLUMNS=8 )
       CALL iotk_write_dat( ounit, "IMIN", imin, COLUMNS=8 )
       CALL iotk_write_dat( ounit, "IMAX", imax, COLUMNS=8 )
       !
       DO isp = 1, nspin
           !
           IF ( nspin == 2 ) THEN
               CALL iotk_write_begin( ounit, "SPIN"//TRIM(iotk_index(isp)) )
           ENDIF
           !
           CALL iotk_write_dat( ounit, "EIG", eig(:,:,isp), COLUMNS=4)
           !
           IF ( nspin == 2 ) THEN
               CALL iotk_write_end( ounit, "SPIN"//TRIM(iotk_index(isp)) )
           ENDIF
           !
       ENDDO
       !
       CALL iotk_write_end( ounit, "WINDOWS" )
       !
       !
       CALL iotk_write_begin( ounit, "SUBSPACE" )
       !
       CALL iotk_write_attr( attr,"dimwinx",dimwinx,FIRST=.TRUE.)
       CALL iotk_write_attr( attr,"nkpts",nkpts)
       CALL iotk_write_attr( attr,"dimwann", dimwann)
       CALL iotk_write_empty( ounit,"DATA",ATTR=attr)
       !
       CALL iotk_write_dat( ounit, "DIMWIN", ndimwin, COLUMNS=8 )
       CALL iotk_write_dat( ounit, "WAN_EIGENVALUES", eig_wan, COLUMNS=8 )
       !
       CALL iotk_write_end( ounit, "SUBSPACE" )
       !
       !
       CALL iotk_close_write( ounit )
       !
   ENDIF

!
! local cleaning
!

   DEALLOCATE( vkpt_cry, wk, STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'deallocating vkpt_cry, wk', ABS(ierr) )
   DEALLOCATE( vkpt, STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'deallocating vkpt', ABS(ierr) )
   !
   DEALLOCATE( ivr, vr, wr, STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'deallocating ivr, wr', ABS(ierr) )
   !
   DEALLOCATE( eig, STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'deallocating eig', ABS(ierr) )
   !
   DEALLOCATE( lwindow, STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'deallocating lwindow', ABS(ierr) )
   !
   DEALLOCATE( ndimwin, imin, imax, STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'deallocating ndimwin, imin, imax', ABS(ierr) )
   !
   DEALLOCATE( u_matrix_opt, u_matrix, STAT=ierr )
   IF ( ierr/=0 ) CALL errore(subname, 'deallocating u_matrices', ABS(ierr) )
   !
   IF( ALLOCATED( rham ) ) THEN
       DEALLOCATE( rham, STAT=ierr )
       IF ( ierr/=0 ) CALL errore(subname, 'deallocating rham', ABS(ierr) )
   ENDIF
   
   CALL log_pop( subname )
   CALL timing( subname, OPR='stop' )
   !
   RETURN
   !
END SUBROUTINE wannier90_to_internal


!**********************************************************
   LOGICAL FUNCTION file_is_wannier90( filename )
   !**********************************************************
   !
   IMPLICIT NONE
   !
   ! Check for wannier90 fmt
   ! To do this, we check that the main datafile
   ! is called $prefix.chk, and that a second file
   ! named $prefix.eig exists
   !
   CHARACTER(*), INTENT(IN) :: filename
   !
   CHARACTER(nstrx) :: prefix_ 
   INTEGER          :: iunit
   LOGICAL          :: lerror, lexist
     !
     !
     CALL iotk_free_unit( iunit )
     !
     file_is_wannier90 = .FALSE.
     lerror = .FALSE.
     !
     INQUIRE( FILE=filename, EXIST=lexist ) 
     IF ( .NOT. lexist ) lerror = .TRUE.
     
     CALL wannier90_tools_get_prefix( filename, prefix_ )
     !
     IF ( LEN_TRIM( prefix_ ) /= 0 ) THEN
         CALL wannier90_tools_init( prefix_ )
     ELSE
         lerror = .TRUE.
         RETURN
     ENDIF

     !
     ! check the existence of the second needed datafile
     ! even if redundant, we re-check also the first file
     !
     INQUIRE( FILE=file_chk, EXIST=lexist )
     IF ( .NOT. lexist ) lerror = .TRUE.
     !
     INQUIRE( FILE=file_eig, EXIST=lexist )
     IF ( .NOT. lexist ) lerror = .TRUE.
     
     !
     ! further check on the content of the files 
     ! can be performed if the case
     !

     IF ( lerror ) THEN
         RETURN
     ENDIF
     !
     file_is_wannier90 = .TRUE.
     !
  END FUNCTION file_is_wannier90

END MODULE wannier90_tools_module

