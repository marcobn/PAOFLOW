!
!      Copyright (C) 2012 WanT Group, 2017 ERMES Group
!
!      This file is distributed under the terms of the
!      GNU General Public License. See the file `License'
!      in the root directory of the present distribution,
!      or http://www.gnu.org/copyleft/gpl.txt .

!============================
PROGRAM write_ham
!============================

!
! Write external (not wannier.x generated) hamiltonian data file.ham to be used by conductor.x:
!
!...  Matrix definition
!
!     Given a conductor (C) bonded to a right lead (A) and a left lead (B)
!
!       H01_L    H00_L   H_LC    H00_C     H_CR   H00_R   H01_R
!       S01_L    S00_L   S_LC    S00_C     S_CR   S00_R   S01_R
!   ...--------------------------------------------------------------...
!         |                |                   |                | 
!         |     LEAD L     |    CONDUCTOR C    |     LEAD R     |
!         |                |                   |                | 
!   ...--------------------------------------------------------------...
!
!     H00_L, H00_R    = on site hamiltonian of the leads (from bulk calculation)
!     H01_L, H01_R    = hopping hamiltonian of the leads (from bulk calculation)
!     H00_C           = on site hamiltonian of the conductor (from supercell calculation)
!     H_LC, H_CR  = coupling matrix between leads and conductor 
!                       (from supercell calculation)
!
!     S00_L, S00_R, S00_C  = on site overlap matrices
!     S01_L, S01_R         = hopping overlap matrices
!     S_LC, S_CR           = coupling overlap matrices
!
!...  Units
!     energies are supposed to be in eV
!
! compile with "gfortran -o write_ham.x write_ham.f90 -I ../extlibs/iotk/include -L ../extlibs/iotk/lib -liotk"
!
USE kinds,            ONLY : dbl
USE io_global_module, ONLY : stdin, stdout
USE iotk_module
!
IMPLICIT NONE

INTEGER        :: dimwann
INTEGER        :: nkpts
INTEGER        :: nrtot
!
CHARACTER(256) :: filein 
CHARACTER(256) :: fileout 
CHARACTER(600) :: attr, card
CHARACTER(10)  :: subname="write_ham"
!
INTEGER        :: i, j, ik, ir, ierr, nsize
LOGICAL        :: have_overlap, htype
REAL(dbl)      :: fermi_energy
INTEGER        :: nk(3), nr(3)
!
INTEGER,      ALLOCATABLE :: ivr(:,:)
REAL(dbl),    ALLOCATABLE :: wr(:)
COMPLEX(dbl), ALLOCATABLE :: rham(:,:,:), ovp(:,:,:)
REAL(dbl),    ALLOCATABLE :: r_rham(:,:,:), r_ovp(:,:,:)


!
! input namelist
!
NAMELIST /INPUT/ dimwann, nkpts, nrtot, have_overlap, &
                 fermi_energy, nk, nr, filein, fileout, htype
!
! end of declariations
!   

!   
! init input namelist   
!   
have_overlap  = .false.
dimwann = 0
nkpts  = 1
nrtot  = 3
nk(:)  = 1
nr(:)  = 0
fermi_energy = 0.0_dbl
htype = .true.
fileout = ' '

ALLOCATE( wr(nkpts) )
wr(:)  = 1

READ(stdin, INPUT, IOSTAT=ierr)
IF ( ierr/=0 ) CALL errore(subname,"reading INPUT namelist",ABS(ierr))

ALLOCATE( ivr(3,nrtot), STAT=ierr )
IF ( ierr/= 0 ) CALL errore(subname,"allocating ivr", ABS(ierr))
!
ALLOCATE( rham(dimwann,dimwann,nrtot), STAT=ierr )
IF ( ierr/= 0 ) CALL errore(subname,"allocating rham", ABS(ierr))
ALLOCATE( ovp(dimwann,dimwann,nrtot), STAT=ierr )
IF ( ierr/= 0 ) CALL errore(subname,"allocating ovp", ABS(ierr))
!
IF( .NOT. htype) THEN
    !
    ALLOCATE( r_rham(dimwann,dimwann,nrtot), STAT=ierr )
    IF ( ierr/= 0 ) CALL errore(subname,"allocating r_rham", ABS(ierr))
    ALLOCATE( r_ovp(dimwann,dimwann,nrtot), STAT=ierr )
    IF ( ierr/= 0 ) CALL errore(subname,"allocating r_ovp", ABS(ierr))
    !
ENDIF

!
! further I/O
!
READ(stdin, *, IOSTAT=ierr) card
IF ( ierr/=0 ) CALL errore(subname,"reading card I", ABS(ierr))
!
DO ik = 1, nrtot
    !
    READ(stdin, *, IOSTAT=ierr) ivr(:,ik)
    IF ( ierr/=0 ) CALL errore(subname,"reading ivr", ABS(ierr))
    !
ENDDO

IF (nkpts /= 1) THEN
    !
    READ(stdin, *, IOSTAT=ierr) card
    IF ( ierr/=0 ) CALL errore(subname,"reading card II", ABS(ierr))
    !
    READ(stdin, *, IOSTAT=ierr) wr(:)
    IF ( ierr/=0 ) CALL errore(subname,"reading wr", ABS(ierr))
    !
ENDIF

IF (htype) THEN
    !
    DO ir = 1, nrtot
        !
        CALL iotk_scan_dat(stdin,"VR"//TRIM(iotk_index(ir)), rham(:,:,ir), IERR=ierr)
        IF ( ierr/=0 ) CALL errore(subname,"scanning VR", ir )
        ! 
        IF ( have_overlap ) THEN
            CALL iotk_scan_dat(stdin,"OVERLAP"//TRIM(iotk_index(ir)), ovp(:,:,ir), IERR=ierr)
            IF ( ierr/=0 ) CALL errore(subname,"scanning OVERLAP", ir )
        ENDIF
        !
    ENDDO
    !
ELSE
    !
    DO ir = 1, nrtot
        !
        CALL iotk_scan_dat(stdin,"VR"//TRIM(iotk_index(ir)), r_rham(:,:,ir), IERR=ierr)
        IF ( ierr/=0 ) CALL errore(subname,"scanning VR", ir )
        !
        IF ( have_overlap ) THEN
            CALL iotk_scan_dat(stdin,"OVERLAP"//TRIM(iotk_index(ir)), r_ovp(:,:,ir), IERR=ierr)
            IF ( ierr/=0 ) CALL errore(subname,"scanning OVERLAP", ir )
        ENDIF
        !
        rham(:,:,ir) = CMPLX( r_rham(:,:,ir), 0.0_dbl, KIND=dbl)
        ovp(:,:,ir)  = CMPLX( r_ovp(:,:,ir),  0.0_dbl, KIND=dbl)
        !
    ENDDO
    !
ENDIF

! 
! write to file 
! 

CALL iotk_open_write( stdout, FILE=TRIM(fileout))

CALL iotk_write_begin(stdout,"HAMILTONIAN")

CALL iotk_write_attr( attr, "dimwann", dimwann, FIRST=.TRUE. )
CALL iotk_write_attr( attr, "nkpts", nkpts )
CALL iotk_write_attr( attr, "nk", nk )
CALL iotk_write_attr( attr, "nrtot", nrtot )
CALL iotk_write_attr( attr, "nr", nr )
CALL iotk_write_attr( attr, "have_overlap", have_overlap )
CALL iotk_write_attr( attr, "fermi_energy", fermi_energy )

CALL iotk_write_empty( stdout, "DATA", ATTR=attr)

nsize=3*nrtot
CALL iotk_write_attr( attr, "type", "integer", FIRST=.TRUE. )
CALL iotk_write_attr( attr, "size", nsize )
CALL iotk_write_attr( attr, "columns", 3 )
CALL iotk_write_attr( attr, "units", "crystal" )
CALL iotk_write_dat( stdout, "IVR", ivr, COLUMNS=3, ATTR=attr )

CALL iotk_write_attr( attr, "type", "real", FIRST=.TRUE. )
CALL iotk_write_attr( attr, "size", nkpts )
CALL iotk_write_dat( stdout, "WR", wr, ATTR=attr )

CALL iotk_write_begin(stdout,"RHAM")
DO ir = 1, nrtot
    CALL iotk_write_dat(stdout,"VR"//TRIM(iotk_index(ir)), rham(:,:,ir))
    IF ( have_overlap ) THEN
        CALL iotk_write_dat(10,"OVERLAP"//TRIM(iotk_index(ir)), ovp(:,:,ir))
    ENDIF
ENDDO
CALL iotk_write_end(stdout,"RHAM")
CALL iotk_write_end(stdout,"HAMILTONIAN")

CALL iotk_close_write( stdout )

!
! cleaning
!

DEALLOCATE( ivr )
DEALLOCATE( rham )
DEALLOCATE( ovp )
DEALLOCATE( wr )
IF ( ALLOCATED( r_rham ) )  DEALLOCATE( r_rham )
IF ( ALLOCATED( r_ovp ) )   DEALLOCATE( r_ovp )

END PROGRAM write_ham

