!
! Copyright (C) 2005 WanT Group, 2017 ERMES Group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License\'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!*********************************************
   MODULE T_smearing_module
   !*********************************************
   !
   USE kinds,                ONLY : dbl
   USE constants,            ONLY : ZERO, ONE, TWO, PI, SQRTPI, SQRT2, CZERO, CONE, CI, EPS_m1
   USE timing_module,        ONLY : timing
   USE log_module,           ONLY : log_push, log_pop
   USE fft_scalar,           ONLY : cft_1z, good_fft_order_1dz
   USE smearing_base_module, ONLY : smearing_func
   !
   IMPLICIT NONE
   PRIVATE 
   SAVE
!
! Contains smearing data
! 
   
   !
   ! ... general parameters
   CHARACTER(30)             :: smearing_type    ! ("lorentzian" | "gaussian" | "fermi-dirac"  |
                                                 !  "marzari-vanderbilt" | "methfessel-paxton" )
   CHARACTER(30)             :: smearing_type_null 
   REAL(dbl)                 :: delta            ! actual smearing parameter
   REAL(dbl)                 :: delta_ratio      ! delta_pole / delta
                                                 ! (eg 10^-3)
   !
   ! ... grid parameters
   INTEGER                   :: nx               ! dimension of the energy grid
   REAL(dbl)                 :: dx, xmax         ! step and grid extrema (-xmax, xmax) 
   REAL(dbl), ALLOCATABLE    :: xgrid(:)         ! grid values
   COMPLEX(dbl), ALLOCATABLE :: g_smear(:)       ! numeric function for G_zero defined on the xgrid
   !
   LOGICAL :: alloc = .FALSE.

!
! end delcarations
!

   PUBLIC :: alloc
   PUBLIC :: delta, delta_ratio
   PUBLIC :: smearing_type, smearing_type_null
   PUBLIC :: nx, dx, xmax, xgrid
   PUBLIC :: g_smear
   !
   PUBLIC :: smearing_init
   PUBLIC :: smearing_deallocate
   PUBLIC :: smearing_memusage


CONTAINS

!
! subroutines
!

!**********************************************************
   SUBROUTINE smearing_init()
   !**********************************************************
   IMPLICIT NONE
       CHARACTER(13)          :: subname="smearing_init"
       REAL(dbl)              :: cost, x
       INTEGER                :: i, ierr
       !
       INTEGER                   :: is_start, is_end    ! index for eps_s grid
       REAL(dbl)                 :: eps_sx              ! eps_s grid extrema
       ! 
       INTEGER                   :: ip_start, ip_end    ! index for eps_p grid
       REAL(dbl)                 :: eps_px              ! eps_p grid extrema
       !
       INTEGER                   :: nfft                ! dim of the FFT grid
       REAL(dbl)                 :: Tmax                ! FFT grid extrema
       REAL(dbl), ALLOCATABLE    :: fft_grid(:)         ! actual smearing funKtion
       COMPLEX(dbl), ALLOCATABLE :: auxs_in(:)          ! cmplx smear for FFT
       COMPLEX(dbl), ALLOCATABLE :: auxp_in(:)          ! pole for FFT
       COMPLEX(dbl), ALLOCATABLE :: auxs_out(:)         ! FFT output for aux1
       COMPLEX(dbl), ALLOCATABLE :: auxp_out(:)         ! FFT input for aux2
       COMPLEX(dbl), ALLOCATABLE :: wrapped(:)          ! auxiliary vect
       !
       INTEGER                   :: ix_start, ix_end    ! index for g_smear grid


       CALL timing ( 'smearing_init', OPR='start')
       CALL log_push ( 'smearing_init' )
       !
       ! few checks
       IF ( alloc ) CALL errore(subname,'smearing already allocated',1)
       IF ( delta_ratio < ZERO  ) CALL errore(subname,'delta_ratio too small',1)
       IF ( delta_ratio > EPS_m1) CALL errore(subname,'delta_ratio too large',1)

       !
       ! define the xgrid
       nx = 2 * INT( TWO * xmax / delta_ratio )
       dx = TWO * xmax / REAL(nx, dbl)
       !
       ALLOCATE( g_smear(nx), xgrid(nx), STAT=ierr )
       IF ( ierr /=0 ) CALL errore(subname,'allocating g_smear, xgrid',ABS(ierr))
       !
       DO i = 1, nx
           !
           xgrid(i) = -REAL(nx/2,dbl)*dx + REAL(i-1, dbl) * dx
           !
       ENDDO

       !
       ! define the fft grid
       !
       ! eps_px (pole)   = xmax +   eps_sx
       ! Tmax (FFT extr) = xmax + 2*eps_sx 
       !

       ! define eps_sx  (half of the width of the smearing function)
       eps_sx = 15.0_dbl
       !
       eps_px = xmax + eps_sx
       Tmax   = xmax + TWO * eps_sx
       !
       nfft   = 1+ INT ( ( Tmax / xmax ) * nx )
       !
       ! find a "good" fft dimension (machine dependent)
       !
       nfft = good_fft_order_1dz( nfft ) 
       !
       !
       ALLOCATE( fft_grid( nfft ), STAT=ierr ) 
       IF (ierr/=0) CALL errore(subname, 'allocating fft_grid',ABS(ierr))
       !
       ALLOCATE( auxs_in( nfft ), auxp_in( nfft ), STAT=ierr ) 
       IF (ierr/=0) CALL errore(subname, 'allocating auxs_in, auxp_in',ABS(ierr))
       !
       ALLOCATE( auxs_out( nfft ), auxp_out( nfft ), STAT=ierr ) 
       IF (ierr/=0) CALL errore(subname, 'allocating auxs_out, auxp_out',ABS(ierr))
       !
       ALLOCATE( wrapped( nfft ), STAT=ierr ) 
       IF (ierr/=0) CALL errore(subname, 'allocating wrapped',ABS(ierr))
       !
       alloc = .true.
       !
       DO i=1,nfft
           !
           fft_grid(i) = -REAL(nfft/2, dbl)*dx + REAL(i-1,dbl) *dx
           !
       ENDDO

       !
       ! find the extrema of interest on the fft_grid
       CALL locate( fft_grid, nfft, -eps_sx, is_start)
       CALL locate( fft_grid, nfft,  eps_sx, is_end)
       !
       CALL locate( fft_grid, nfft, -eps_px, ip_start)
       CALL locate( fft_grid, nfft,  eps_px, ip_end)

       !
       ! define the smearing function
       !
       auxs_in(:)  = CZERO
       auxs_out(:) = CZERO

       cost = ONE / delta 
       !
       DO i= is_start, is_end
           !
           x          = fft_grid(i)
           auxs_in(i) = cost * smearing_func( x, TRIM(smearing_type) )
           !
       ENDDO
       
       !
       ! define the input pole function
       !
       auxp_in(:)  = CZERO
       auxp_out(:) = CZERO
       !
       cost = ONE 
       DO i = ip_start, ip_end
           !
           x          = fft_grid(i)
           auxp_in(i) = cost / ( x + CI * delta_ratio )
           !
       ENDDO

!
! perform the FFT
!
       !
       ! perform the smearing func wrapping
       !
       CALL locate( fft_grid, nfft, ZERO, i)
       IF ( fft_grid(i) < ZERO ) i = i+1
       !
       wrapped(:) = CSHIFT( auxs_in(:), i-1 )
       auxs_in(:) = wrapped(:)


       !
       ! freq to time FT
       !
       CALL timing ( 'cft_1z', OPR='start')
       CALL log_push ( 'cft_1z')
            !
            CALL cft_1z ( auxs_in, 1, nfft, nfft, -1, auxs_out)
            CALL cft_1z ( auxp_in, 1, nfft, nfft, -1, auxp_out)
            !
       CALL timing ( 'cft_1z', OPR='stop')
       CALL log_pop ( 'cft_1z')

       !
       ! perform the convolution
       !
       cost = TWO * Tmax
       DO i=1, nfft
           !
           auxp_out(i) = cost * auxp_out(i) * auxs_out(i) 
           !
       ENDDO

       !
       ! backwards fft
       !
       CALL timing ( 'cft_1z', OPR='start')
       CALL log_push ( 'cft_1z')
            !
            CALL cft_1z ( auxp_out, 1, nfft, nfft, 1, auxp_in)
            !
       CALL timing ( 'cft_1z', OPR='stop')
       CALL log_pop ( 'cft_1z')

       !
       ! smeared green function extraction
       !
       CALL locate( fft_grid, nfft, -xmax, ix_start )
       ix_end = ix_start + nx -1
       !
       g_smear(:) = auxp_in(ix_start:ix_end) 

       !
       ! local cleaning
       !
       DEALLOCATE( fft_grid, STAT=ierr ) 
       IF (ierr/=0) CALL errore(subname, 'deallocating fft_grid',ABS(ierr))
       !
       DEALLOCATE( auxs_in, auxp_in, STAT=ierr ) 
       IF (ierr/=0) CALL errore(subname, 'deallocating auxs_in, auxp_in',ABS(ierr))
       !
       DEALLOCATE( auxs_out, auxp_out, STAT=ierr ) 
       IF (ierr/=0) CALL errore(subname, 'deallocating auxs_out, auxp_out',ABS(ierr))
       !
       DEALLOCATE( wrapped, STAT=ierr ) 
       IF (ierr/=0) CALL errore(subname, 'deallocating wrapped',ABS(ierr))

       CALL timing ( 'smearing_init', OPR='stop')
       CALL log_pop ( 'smearing_init')

   END SUBROUTINE smearing_init


!**********************************************************
   SUBROUTINE smearing_deallocate()
   !**********************************************************
   IMPLICIT NONE
       CHARACTER(19)          :: subname="smearing_deallocate"
       INTEGER :: ierr

       IF ( ALLOCATED(xgrid) ) THEN
            DEALLOCATE(xgrid, STAT=ierr)
            IF (ierr/=0) CALL errore(subname,'deallocating xgrid',ABS(ierr))
       ENDIF
       IF ( ALLOCATED(g_smear) ) THEN
            DEALLOCATE(g_smear, STAT=ierr)
            IF (ierr/=0) CALL errore(subname,'deallocating g_smear',ABS(ierr))
       ENDIF
       alloc = .FALSE.
   END SUBROUTINE smearing_deallocate


!**********************************************************
   REAL(dbl) FUNCTION smearing_memusage()
   !**********************************************************
   IMPLICIT NONE
       !
       REAL(dbl) :: cost
       !
       cost = ZERO
       IF ( ALLOCATED(xgrid) )    cost = cost + REAL(SIZE(xgrid))      *  8.0_dbl
       IF ( ALLOCATED(g_smear) )  cost = cost + REAL(SIZE(g_smear))    * 16.0_dbl
       !
       smearing_memusage = cost / 1000000.0_dbl
       !
   END FUNCTION smearing_memusage


END MODULE T_smearing_module

