!
! Copyright (C) 2001-2004 ESPRESSO group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!
!--------------------------------------------------------------------
subroutine sph_bes (msh, r, q, l, jl)
  !--------------------------------------------------------------------
  !
  ! ... input:
  ! ...   msh     = number of grid points points
  ! ...   r(1:msh)= radial grid
  ! ...   q       = q
  ! ...   l       = angular momentum (-1 <= l <= 6)
  ! ... output:
  ! ...   jl(1:msh) = j_l(q*r(i))  (j_l = spherical bessel function)
  !
  use kinds, only: dbl
  USE constants, ONLY : ONE, ZERO, EPS_m8
  !
  implicit none
  !
  integer :: msh, l
  real(dbl) :: r (msh), q, jl (msh)
  !
  integer :: ir0
  !
#if defined (__MASS)
  real(dbl) :: qr(msh), sin_qr(msh), cos_qr(msh)
#endif
  !
  if (abs (q) < EPS_m8 ) then
     if (l == -1) then
        call errore ('sph_bes', 'j_{-1}(0) ?!?', 1)
     elseif (l == 0) then
        jl(:) = ONE
     else
        jl(:) = ZERO
     endif
  else
     if (abs (q * r (1) ) > EPS_m8 ) then
        ir0 = 1
     else
        if (l == -1) then
           call errore ('sph_bes', 'j_{-1}(0) ?!?', 2)
        elseif (l == 0) then
           jl (1) = ONE
        else
           jl (1) = ZERO
        endif
        ir0 = 2
     endif

     if (l == - 1) then

#if defined (__MASS)

        qr = q * r
        call vcos( cos_qr, qr, msh)
        jl(ir0:) = cos_qr(ir0:) / ( q * r(ir0:) )

#else

        jl (ir0:) = cos (q * r (ir0:) ) / (q * r (ir0:) )

#endif

     elseif (l == 0) then

#if defined (__MASS)

        qr = q * r
        call vsin( sin_qr, qr, msh)
        jl (ir0:) = sin_qr(ir0:) / (q * r (ir0:) )

#else

        jl (ir0:) = sin (q * r (ir0:) ) / (q * r (ir0:) )

#endif

     elseif (l == 1) then

#if defined (__MASS)

        qr = q * r
        call vcos( cos_qr, qr, msh)
        call vsin( sin_qr, qr, msh)
        jl (ir0:) = ( sin_qr(ir0:) / (q * r (ir0:) ) - &
                      cos_qr(ir0:) ) / (q * r (ir0:) )

#else

        jl (ir0:) = (sin (q * r (ir0:) ) / (q * r (ir0:) ) - &
                     cos (q * r (ir0:) ) ) / (q * r (ir0:) )

#endif

     elseif (l == 2) then

#if defined (__MASS)

        qr = q * r
        call vcos( cos_qr, qr, msh)
        call vsin( sin_qr, qr, msh)
        jl (ir0:) = ( (3.0_dbl / (q*r(ir0:)) - (q*r(ir0:)) ) * sin_qr(ir0: ) - &
                       3.0_dbl * cos_qr(ir0:) ) / (q*r(ir0:))**2

#else

        jl (ir0:) = ( (3.0_dbl / (q*r(ir0:)) - (q*r(ir0:)) ) * sin (q*r(ir0:)) - &
                       3.0_dbl * cos (q*r(ir0:)) ) / (q*r(ir0:))**2

#endif

     elseif (l == 3) then

#if defined (__MASS)

        qr = q * r
        call vcos( cos_qr, qr, msh)
        call vsin( sin_qr, qr, msh)
        jl (ir0:) = (sin_qr (ir0:) * &
                     (15.0_dbl / (q*r(ir0:)) - 6.0_dbl * (q*r(ir0:)) ) + &
                     cos_qr (ir0:) * ( (q*r(ir0:))**2 - 15.0_dbl ) ) / &
                     (q*r(ir0:))**3

#else

        jl (ir0:) = (sin (q*r(ir0:)) * &
                     (15.0_dbl / (q*r(ir0:)) - 6.0_dbl * (q*r(ir0:)) ) + &
                     cos (q*r(ir0:)) * ( (q*r(ir0:))**2 - 15.0_dbl ) ) / &
                     (q*r(ir0:)) **3

#endif

     elseif (l == 4) then

#if defined (__MASS)

        qr = q * r
        call vcos( cos_qr, qr, msh)
        call vsin( sin_qr, qr, msh)
        jl (ir0:) = (sin_qr (ir0:) * &
                     (105.0_dbl - 45.0_dbl * (q*r(ir0:))**2 + (q*r(ir0:))**4) + &
                     cos_qr (ir0:) * &
                     (10.0_dbl * (q*r(ir0:))**3 - 105.0_dbl * (q*r(ir0:))) ) / &
                       (q*r(ir0:))**5

#else

        jl (ir0:) = (sin (q*r(ir0:)) * &
                     (105.0_dbl - 45.0_dbl * (q*r(ir0:))**2 + (q*r(ir0:))**4) + &
                     cos (q*r(ir0:)) * &
                     (10.0_dbl * (q*r(ir0:))**3 - 105.0_dbl * (q*r(ir0:))) ) / &
                        (q*r(ir0:))**5
#endif

     elseif (l == 5) then

#if defined (__MASS)
        qr = q * r
        call vcos( cos_qr, qr, msh)
        call vsin( sin_qr, qr, msh)
        jl (ir0:) = (-cos_qr(ir0:) - &
                     (945.0_dbl *cos_qr(ir0:)) / (q*r(ir0:)) ** 4 + &
                     (105.0_dbl *cos_qr(ir0:)) / (q*r(ir0:)) ** 2 + &
                     (945.0_dbl *sin_qr(ir0:)) / (q*r(ir0:)) ** 5 - &
                     (420.0_dbl *sin_qr(ir0:)) / (q*r(ir0:)) ** 3 + &
                     ( 15.0_dbl *sin_qr(ir0:)) / (q*r(ir0:)) ) / (q*r(ir0:))
#else
        jl (ir0:) = (-cos(q*r(ir0:)) - &
                     (945.0_dbl *cos(q*r(ir0:))) / (q*r(ir0:)) ** 4 + &
                     (105.0_dbl *cos(q*r(ir0:))) / (q*r(ir0:)) ** 2 + &
                     (945.0_dbl *sin(q*r(ir0:))) / (q*r(ir0:)) ** 5 - &
                     (420.0_dbl *sin(q*r(ir0:))) / (q*r(ir0:)) ** 3 + &
                     ( 15.0_dbl *sin(q*r(ir0:))) / (q*r(ir0:)) ) / (q*r(ir0:))
#endif

     elseif (l == 6) then

#if defined (__MASS)

        qr = q * r
        call vcos( cos_qr, qr, msh)
        call vsin( sin_qr, qr, msh)
        jl (ir0:) = ((-10395.0_dbl *cos_qr(ir0:)) / (q*r(ir0:))**5 + &
                     (  1260.0_dbl *cos_qr(ir0:)) / (q*r(ir0:))**3 - &
                     (    21.0_dbl *cos_qr(ir0:)) / (q*r(ir0:))    - &
                                sin_qr(ir0:)                   + &
                     ( 10395.0_dbl *sin_qr(ir0:)) / (q*r(ir0:))**6 - &
                     (  4725.0_dbl *sin_qr(ir0:)) / (q*r(ir0:))**4 + &
                     (   210.0_dbl *sin_qr(ir0:)) / (q*r(ir0:))**2 ) / (q*r(ir0:))
#else

        jl (ir0:) = ((-10395.0_dbl *cos(q*r(ir0:))) / (q*r(ir0:))**5 + &
                     (  1260.0_dbl *cos(q*r(ir0:))) / (q*r(ir0:))**3 - &
                     (    21.0_dbl *cos(q*r(ir0:))) / (q*r(ir0:))    - &
                                sin(q*r(ir0:))                   + &
                     ( 10395.0_dbl *sin(q*r(ir0:))) / (q*r(ir0:))**6 - &
                     (  4725.0_dbl *sin(q*r(ir0:))) / (q*r(ir0:))**4 + &
                     (   210.0_dbl *sin(q*r(ir0:))) / (q*r(ir0:))**2 ) / (q*r(ir0:))
#endif

     else

        call errore ('sph_bes', 'not implemented', l)

     endif

  endif
  !
  return
end subroutine sph_bes
