!
! Copyright (C) 2000-2013 A. Marini and the YAMBO team 
!              http://www.yambo-code.org
! 
! This file is distributed under the terms of the GNU 
! General Public License. You can redistribute it and/or 
! modify it under the terms of the GNU General Public 
! License as published by the Free Software Foundation; 
! either version 2, or (at your option) any later version.
!
! This program is distributed in the hope that it will 
! be useful, but WITHOUT ANY WARRANTY; without even the 
! implied warranty of MERCHANTABILITY or FITNESS FOR A 
! PARTICULAR PURPOSE.  See the GNU General Public License 
! for more details.
!
! You should have received a copy of the GNU General Public 
! License along with this program; if not, write to the Free 
! Software Foundation, Inc., 59 Temple Place - Suite 330,Boston, 
! MA 02111-1307, USA or visit http://www.gnu.org/copyleft/gpl.txt.
!
subroutine fft_desc_init(n,dfft)
 !
 ! init dffts as required by the QE FFT lib
 !
 use kinds,      ONLY:DP=>dbl
 use constants,  ONLY:pi
 use fft_types,  ONLY:fft_dlay_descriptor
 use fft_scalar, ONLY:good_fft_dimension
 use stick_set,  ONLY:pstickset
 use lattice_module,  ONLY:alat
 use R_lattice,  ONLY:b,E_of_shell,n_g_shells
 !
 implicit none
 integer,       intent(in) :: n(3)
 type(fft_dlay_descriptor) :: dfft
 !
 type(fft_dlay_descriptor) :: dfftp_dum
 integer  :: ngw, ngm, ngs 
 real(DP) :: tpiba, bg(3,3)
 real(DP) :: gcutm, gkcut
    

 dfft%nr1=n(1)
 dfft%nr2=n(2)
 dfft%nr3=n(3)
 ! some FFT libraries are more efficient with 
 ! a slightly larger workspace
 dfft%nr1x  = good_fft_dimension( dfft%nr1 )
 dfft%nr2x  = dfft%nr2
 dfft%nr3x  = good_fft_dimension( dfft%nr3 )
 !
 dfftp_dum%nr1=dfft%nr1
 dfftp_dum%nr2=dfft%nr2
 dfftp_dum%nr3=dfft%nr3
 dfftp_dum%nr1x=dfft%nr1x
 dfftp_dum%nr2x=dfft%nr2x
 dfftp_dum%nr3x=dfft%nr3x
 !
 tpiba=2.0_DP*PI/alat(1)
 bg=b/tpiba
 !
 ! QE routines use Ry-au units (then converted to internal QE units
 ! by taking tpiba into account)
 !
 gcutm=2.0_DP*MAXVAL(E_of_shell(1:n_g_shells))/tpiba**2 
 gkcut=gcutm/4.0_DP
 !
 ! set up fft descriptors, including parallel stuff: sticks, planes, etc.
 !
!XXX
!WRITE(0,*) 
!WRITE(0,*) "gcutm", gcutm
!WRITE(0,*) "gkcut", gkcut
!WRITE(0,*)

    CALL pstickset( .FALSE., bg, gcutm, gkcut, gcutm, &
                    dfftp_dum, dfft, ngw, ngm, ngs, 0, 0, 1, 0, 1 )
    !
!WRITE(0,*) "ngw_", ngw
!WRITE(0,*) "ngm_", ngm
!WRITE(0,*) "ngs_", ngs
!WRITE(0,*)
!WRITE(0,*) "dffts%nst", dfft%nst
!WRITE(0,*) "dffts%nsp", dfft%nsp
!WRITE(0,*) "dffts%nsw", dfft%nsw
!WRITE(0,*) "dffts%npl", dfft%npl
!WRITE(0,*) "dffts%nnp", dfft%nnp
!WRITE(0,*) "dffts%nnr", dfft%nnr
!WRITE(0,*) "dffts%ngl", dfft%ngl
!WRITE(0,*) "dffts%nwl", dfft%nwl
!WRITE(0,*) "dffts%npp", dfft%npp
!WRITE(0,*) "dffts%ipp", dfft%ipp
!WRITE(0,*) "dffts%iss", dfft%iss
!WRITE(0,*) "dffts%isind", dfft%isind
!WRITE(0,*) "dffts%ismap", dfft%ismap
!WRITE(0,*) "dffts%iplp", dfft%iplp
!WRITE(0,*) "dffts%iplw", dfft%iplw
!WRITE(0,*)


return
end subroutine fft_desc_init

