!      Copyright (C) 2007 WanT Group, 2017 ERMES Group
!
!      This file is distributed under the terms of the
!      GNU General Public License. See the file `License\'
!      in the root directory of the present distribution,
!      or http://www.gnu.org/copyleft/gpl.txt .
!
!***********************************************
   PROGRAM current
   !***********************************************
   USE kinds,                ONLY : dbl
   USE parameters,           ONLY : nstrx 
   USE version_module,       ONLY : version_number
   USE timing_module,        ONLY : timing, timing_upto_now
   USE io_module,            ONLY : stdout, stdin, curr_unit => aux_unit,   &
                                    cond_unit => aux1_unit,                 &
                                    work_dir, prefix, postfix
!   USE T_smearing_module,    ONLY : smearing_init

   IMPLICIT NONE

   !
   REAL(dbl), ALLOCATABLE   :: curr(:)                 ! current
   REAL(dbl)                :: mu_L, mu_R              ! chemical potentials
   REAL(dbl)                :: mu_L_aux, mu_R_aux      ! 
   REAL(dbl)                :: sigma                   ! broadening
   REAL(dbl), ALLOCATABLE   :: ftemp_L(:), ftemp_R(:)  ! temperature smearing functions
   REAL(dbl), ALLOCATABLE   :: transm(:)               ! transmittance from data file
   !
   CHARACTER(nstrx)         :: filein                  ! input  filename (transmittance)
   CHARACTER(nstrx)         :: fileout                 ! output filename (current)

   !
   ! energy grid
   !
   INTEGER                  :: ne                      ! dimension of the energy grid
   REAL(dbl)                :: de         
   REAL(dbl)                :: de_old         
   REAL(dbl), ALLOCATABLE   :: egrid(:)                ! energy grid

   !
   ! bias grid
   !
   INTEGER                  :: nV                      ! dimension of the bias grid
   REAL(dbl)                :: Vmin                    ! Vgrid extrema
   REAL(dbl)                :: Vmax                    !
   REAL(dbl)                :: dV         
   REAL(dbl), ALLOCATABLE   :: Vgrid(:)                ! bias grid

   !
   ! interpolation variables
   !
   INTEGER                  :: ndiv, ndim_new          ! interpolation grid dimension
   REAL(dbl)                :: de_new                  ! interpolation grid step 
   REAL(dbl), ALLOCATABLE   :: egrid_new(:)            ! interpolation energy grid
   REAL(dbl), ALLOCATABLE   :: transm_new(:)           ! interpolated transmittance
   INTEGER                  :: i_min, i_max, inew      ! 

   !
   ! local variables
   !
   INTEGER                  :: ie, iv, ierr, ios
   INTEGER                  :: i_start, i_end, ndim    ! integration extrema
   REAL(dbl), ALLOCATABLE   :: funct(:)                ! auxiliary vectors for integration

   !
   ! input namelist
   !
   NAMELIST /INPUT/ prefix, postfix, work_dir, mu_L, mu_R, sigma, filein, fileout, &
                    Vmin, Vmax, nV

!
!------------------------------
! main body
!------------------------------
!
   CALL startup(version_number,'current')

!
! ... Read INPUT namelist from stdin
!
   prefix                      = ' '
   postfix                     = ' '
   work_dir                    = './'
   filein                      = ' '
   fileout                     = ' '
   mu_L                        =  -0.5
   mu_R                        =   0.5
   sigma                       =   0.1    ! eV
   Vmin                        =  -1.0
   Vmax                        =   1.0
   nV                          =  1000
                                                                                                                  
   CALL input_from_file ( stdin )
   !
   READ(stdin, INPUT, IOSTAT=ierr)
   IF ( ierr /= 0 )  CALL errore('current','Unable to read namelist INPUT',ABS(ierr))

!
! init
!
   !
   ! get energy grid and transmittance from data file
   !
   IF( LEN_TRIM( filein ) == 0 ) &
      filein = TRIM(work_dir)//'/'//TRIM(prefix)//'cond'//TRIM(postfix)//'.dat'
   !
   OPEN ( cond_unit, FILE=TRIM(filein), FORM='formatted', IOSTAT=ierr )
   IF ( ierr/=0 ) CALL errore('current','opening file = '//TRIM(filein), ABS(ierr) )
   !
   ie = 0
   !
   DO WHILE ( .TRUE. ) 
      !
      READ ( cond_unit, *, IOSTAT=ierr )
      !
      IF ( ierr /= 0 ) EXIT
      ie = ie + 1
      !
   ENDDO
   !
   ne = ie 
   !
   ALLOCATE ( egrid(ne), STAT=ierr )
   IF( ierr /=0 ) CALL errore('current','allocating egrid', ABS(ierr) )
   !
   ALLOCATE ( transm(ne), STAT=ierr )
   IF( ierr /=0 ) CALL errore('current','allocating transmittance', ABS(ierr) )
   !
   REWIND ( cond_unit )
   !
   DO ie = 1, ne
       READ ( cond_unit, *, IOSTAT=ios ) egrid(ie), transm(ie)
       IF ( ios/=0 ) CALL errore('current','reading T',ie)  
   ENDDO
   !
   CLOSE( cond_unit )

   !
   ! allocate
   !
   ALLOCATE ( Vgrid(nV), STAT=ierr )
   IF( ierr /=0 ) CALL errore('current','allocating Vgrid', ABS(ierr) )
   !
   ALLOCATE ( curr(nV), STAT=ierr )
   IF( ierr /=0 ) CALL errore('current','allocating current', ABS(ierr) )

   !
   ! bias grid
   !
   dV = (Vmax - Vmin)/REAL(nV-1, dbl)
   !
   DO iv = 1, nV
      Vgrid(iv) = Vmin + REAL(iv-1, dbl) * dV
   ENDDO 
   !

!
! current calculation
!
   !
   curr(:) = 0.0
   de_old = (egrid(ne) - egrid(1))/REAL(ne-1, dbl)
   !
   DO iv = 1, nV
      !
      mu_L_aux = mu_L * Vgrid(iv)
      mu_R_aux = mu_R * Vgrid(iv)
      !
      ! integration extrema
      CALL locate( egrid, ne, MIN( mu_L_aux, mu_R_aux ) -sigma -3.0_dbl*de_old, i_start)
      IF ( i_start == 0 .OR. i_start == ne ) CALL errore('current','invalid i_start',4)
      !
      CALL locate( egrid, ne, MAX( mu_R_aux, mu_L_aux ) +sigma +3.0_dbl*de_old, i_end)
      IF ( i_end == 0 .OR. i_end == ne ) CALL errore('current','invalid i_end',5)

      !
      ! egrid
      !
      ndim = i_end - i_start + 1
      !
      ! simpson routine requires that ndim is an odd number
      !
      IF ( MOD(ndim, 2) == 0 ) THEN
          i_end = i_end - 1
          ndim  = ndim -1
      ENDIF
      !
      de = (egrid(i_end) - egrid(i_start))/REAL(ndim-1, dbl)

      !
      ! redefinition of the integration mesh for a better interpolation
      !
      ndiv = NINT( de / (2.0_dbl*sigma) )
      IF (ndiv == 0) ndiv = 1      
      !
      de_new   = de / REAL(ndiv, dbl)
      ndim_new = (ndim - 1) * ndiv + 1 
      !
      ALLOCATE ( transm_new(ndim_new), STAT=ierr )
      IF( ierr /=0 ) CALL errore('current','allocating transm_new', ABS(ierr) )
      !
      ALLOCATE ( egrid_new(ndim_new), STAT=ierr )
      IF( ierr /=0 ) CALL errore('current','allocating transm_new', ABS(ierr) )
      !
      ! new integration mesh
      egrid_new(1) = egrid(i_start)
      !
      DO inew = 2, ndim_new   
         egrid_new(inew) = egrid_new(1) + de_new * REAL( inew -1, dbl)
      ENDDO
      !
      ! Transmittance interpolated on the new grid
      !
      IF (ndiv /= 1) THEN
         !
         DO ie = i_start, i_end - 1
            !
            i_min = (ie-i_start) * ndiv + 1 
            i_max = (ie-i_start+1) * ndiv + 1 
            !
            transm_new(i_min) = transm(ie)
            transm_new(i_max) = transm(ie+1)
            !
            DO inew = i_min+1, i_max-1
               !
               transm_new(inew) = transm_new(i_max)*(egrid_new(inew)-egrid_new(i_min))/  &
                                  (egrid_new(i_max)-egrid_new(i_min)) -                  &
                                  transm_new(i_min)*(egrid_new(inew)-egrid_new(i_max))/  &
                                  (egrid_new(i_max)-egrid_new(i_min))
            ENDDO
         ENDDO
         !
      ELSE 
         !
         ! ndiv == 1
         !
         transm_new(:) = transm(i_start:i_end)
      ENDIF
      !
      ! auxiliary vectors for integral calculation
      !
      ALLOCATE ( funct(ndim_new), STAT=ierr )
      IF( ierr /=0 ) CALL errore('current','allocating funct', ABS(ierr) )
      !
      ALLOCATE ( ftemp_L(ndim_new), ftemp_R(ndim_new), STAT=ierr )
      IF( ierr /=0 ) CALL errore('current','allocating ftemp', ABS(ierr) )
      !
      !
      DO ie = 1, ndim_new
          !
          ftemp_L(ie) = 1.0 / ( EXP( -(egrid_new( ie ) -mu_L_aux ) / sigma) + 1.0 )
          ftemp_R(ie) = 1.0 / ( EXP( -(egrid_new( ie ) -mu_R_aux ) / sigma) + 1.0 )
          !
      ENDDO
      !
      ! perform the integration
      !
      ! if you want to use the simpson routine for integration 
      ! uncomment the following line and comment the next one   
      !
!      funct(:) = ( ftemp_L(:) - ftemp_R(:) ) * transm(i_start:i_end)
      funct(:) = transm_new(1:ndim_new)
      !
      ! if you want to use the simpson routine for integration 
      ! uncomment the following line and comment the next ones   
      !
!      CALL simpson (ndim, funct, rab, curr(iv) )
      DO ie = 1, ndim_new-1
         curr(iv) = curr(iv) + ( ftemp_L(ie) - ftemp_R(ie) )*funct(ie)*de_new/3.0 + &
                    ( ftemp_L(ie+1) - ftemp_R(ie+1) )*funct(ie+1)*de_new/3.0 +      &
                    ( ftemp_L(ie) - ftemp_R(ie) )*funct(ie+1)*de_new/6.0 +          &
                    ( ftemp_L(ie+1) - ftemp_R(ie+1) )*funct(ie)*de_new/6.0
      ENDDO
      !
      DEALLOCATE ( transm_new, STAT=ierr )
      IF( ierr /=0 ) CALL errore('current','deallocating transm_new', ABS(ierr) )
      !
      DEALLOCATE ( egrid_new, STAT=ierr )
      IF( ierr /=0 ) CALL errore('current','deallocating transm_new', ABS(ierr) )
      DEALLOCATE ( funct, STAT=ierr )
      IF( ierr /=0 ) CALL errore('current','deallocating funct', ABS(ierr) )
      DEALLOCATE ( ftemp_L, ftemp_R, STAT=ierr )
      IF( ierr /=0 ) CALL errore('current','deallocating ftemp', ABS(ierr) )
      !
   ENDDO
   !

   !
   ! write input data on the output file
   !
   IF ( LEN_TRIM(fileout) == 0 ) &
        fileout = TRIM(work_dir)//'/'//TRIM(prefix)//'current'//TRIM(postfix)//'.dat'
   !
   OPEN ( curr_unit, FILE=TRIM(fileout), FORM='formatted' )
   !
   DO iv = 1, nV
       WRITE ( curr_unit, '(2(f15.9))' ) Vgrid(iv), curr(iv)
   ENDDO
   !
   CLOSE( curr_unit )
   !

   
   !
   ! deallocate
   !
   DEALLOCATE ( egrid, STAT=ierr )
   IF( ierr /=0 ) CALL errore('current','deallocating egrid', ABS(ierr) )
   !
   DEALLOCATE ( Vgrid, STAT=ierr )
   IF( ierr /=0 ) CALL errore('current','deallocating Vgrid', ABS(ierr) )
   !
   DEALLOCATE ( curr, STAT=ierr )
   IF( ierr /=0 ) CALL errore('current','deallocating current', ABS(ierr) )
   !
   DEALLOCATE ( transm, STAT=ierr )
   IF( ierr /=0 ) CALL errore('current','deallocating transmittance', ABS(ierr) )

   CALL cleanup()
   !
   CALL shutdown('current')

END PROGRAM current
  
