! 
! Copyright (C) 2005 WanT Group, 2017 ERMES Group
! 
! This file is distributed under the terms of the 
! GNU General Public License. See the file `License' 
! in the root directory of the present distribution, 
! or http://www.gnu.org/copyleft/gpl.txt . 
! 
!********************************************
   MODULE T_input_parameters_module
!********************************************
   !
   USE kinds,         ONLY : dbl
   USE constants,     ONLY : ZERO, EPS_m1, EPS_m2, EPS_m5, EPS_m4, EPS_m3 
   USE parameters,    ONLY : nstrx
   USE parser_module, ONLY : change_case
   USE io_module,     ONLY : ionode, ionode_id
   USE log_module,    ONLY : log_push, log_pop
   USE mp,            ONLY : mp_bcast
   !
   IMPLICIT NONE
   PRIVATE
   SAVE
!
! This module contains the definitions of the parameters in the
! input file and of thier default values (when any). 
! These data are then exported to the input module where the
! main routine controls the IO and after that exports to the
! final modules where internal data are stored.
!
! The T character in front of the module name stands there
! in order to distinguish from wannier modules.
!
! Here are also the routine reading and checking the NAMELIST
!
! routines in this module:
! SUBROUTINE  read_namelist_input_conductor(unit)
!

!
! ... declarations

!
!======================================== 
! INPUT_CONDUCTOR Namelist parameters
!======================================== 

   INTEGER :: dimL = 0
       ! WF number in lead L

   INTEGER :: dimR = 0
       ! WF number in lead R

   INTEGER :: dimC = 0
       ! WF number in the central conductor region

   INTEGER :: transport_dir = 0
       ! transport direction 

   CHARACTER(nstrx) :: calculation_type = 'conductor'
       ! ( 'conductor' | 'bulk' )
       ! wheter full conductor or bulk calculation is performed

   CHARACTER(nstrx) :: calculation_type_allowed(2)
   DATA calculation_type_allowed / 'conductor',  'bulk' /
       ! the allowed values for calculation_type

   CHARACTER(nstrx) :: conduct_formula = 'landauer'
       ! ( 'landauer' | 'generalized' )
       ! wheter use landauer of correlation corrected formula

   CHARACTER(nstrx) :: conduct_formula_allowed(2)
   DATA conduct_formula_allowed / 'landauer',  'generalized' /
       ! the allowed values for conduct_formula

   CHARACTER(nstrx) :: carriers = 'electrons'
       ! ( 'electrons' | 'phonons' )
       ! electron or phonon case

   CHARACTER(nstrx) :: carriers_allowed(2)
   DATA carriers_allowed / 'electrons',  'phonons' /
       ! the allowed values for the carriers

   INTEGER :: ne = 1000  
       ! the dimension of the energy grid

   INTEGER :: ne_buffer = 1
       ! dimension of the energy buffering for correlation sgm

   REAL(dbl) :: emin = -10.0
       ! lower bound of the energy range

   REAL(dbl) :: emax =  10.0
       ! upper bound of the energy range

   REAL(dbl) :: delta =  EPS_m5
       ! i\delta broadening of green functions

   CHARACTER(30) :: smearing_type = 'lorentzian'
       ! type of smearing technique

   CHARACTER(30) :: smearing_type_allowed(8)
   DATA smearing_type_allowed / 'lorentzian',  'gaussian', 'fermi-dirac', 'fd',         &
                                'methfessel-paxton', 'mp', 'marzari-vanderbilt', 'mv' /
       ! the allowed values for smearing_type
       
   REAL(dbl) :: delta_ratio =  5.0_dbl * EPS_m3
       ! ratio between delta (used for convolution with the pole) and
       ! smearing delta

   REAL(dbl) :: xmax = 25.0    
        ! grid extrema  (-xmax, xmax)

   REAL(dbl) :: bias =  ZERO
       ! effective bias between the leads
       ! not fully implemented at the moment

   INTEGER :: nk(2) =  0
       ! dimension of the 2D kpt mesh on which the input 
       ! Hamiltonian will be interpolated

   INTEGER :: s(2) =  0
       ! shifts for the generation of the 2D mesh of kpts
 
   LOGICAL :: use_symm = .TRUE.
       ! whether to use symmetry to reduce the numberof kpts
       ! only Time-Rev is implemented at the moment

   INTEGER :: nprint = 20 
       ! every nprint energy step write to stdout

   INTEGER :: niterx = 200
       ! max number of iterations in the calculation of
       ! lead self-energies

   INTEGER :: nfailx = 5
       ! max allowed number of failures during lead-sgm calculation

   REAL(dbl) :: transfer_thr = 1.0d-7
       ! Threshold for the convergence of the iterative procedure
       ! to compute the transfer matrices (lead self-energies)

   LOGICAL :: write_kdata = .FALSE.
       ! write kpoint-resolved dos an transmittance to output

   LOGICAL :: write_lead_sgm = .FALSE.
       ! write the computed lead self-energies to disk.
       ! ACTUNG: at the moment it needs use_symm=.FALSE.

   LOGICAL :: write_gf = .FALSE.
       ! write the green's function in the conductor region
       ! ACTUNG: at the moment it needs use_symm=.FALSE.

   LOGICAL :: do_eigenchannels = .FALSE.
       ! compute eigenchannels

   INTEGER :: neigchnx = 200000
       ! maximum number of eigenchannels printed out
       ! Let unset means all of them are written.

   LOGICAL :: do_eigplot = .FALSE.
       ! compute and write auxiliary data to plot the eigenchannels
 
   INTEGER :: ie_eigplot = 0
       ! write auxdata for eigechannel analysis at this energy

   INTEGER :: ik_eigplot = 0
       ! write auxdata for eigechannel analysis at this kpt

   CHARACTER(nstrx) :: work_dir   = './'
       ! working directory where to write datafiles

   CHARACTER(nstrx) :: prefix     = ' '
       ! prefix used for the names of files

   CHARACTER(nstrx) :: postfix     = ' '
       ! postfix used for the names of files

   CHARACTER(nstrx) :: datafile_L = ' '
       ! the name of the file containing L lead data

   CHARACTER(nstrx) :: datafile_C = ' '
       ! the name of the file containing central conductor data

   CHARACTER(nstrx) :: datafile_R = ' '
       ! the name of the file containing R lead data

   CHARACTER(nstrx) :: datafile_sgm = ' '
       ! the name of the file containing correlation self-energy
       ! If a valid file is provided, correlation is taken into account.
       ! This var is kept for back-compatibility, and it refers to
       ! datafile_C_sgm

   CHARACTER(nstrx) :: datafile_C_sgm = ' '
       ! the name of the file containing correlation self-energy
       ! C, CR, LC regions

   CHARACTER(nstrx) :: datafile_L_sgm = ' '
       ! the name of the file containing correlation self-energy
       ! 00L, 01L regions

   CHARACTER(nstrx) :: datafile_R_sgm = ' '
       ! the name of the file containing correlation self-energy
       ! 00R, 01R regions

   LOGICAL :: do_orthoovp = .FALSE.
       ! if a non-orthogonal set is used, it is lowdin-orthogonalized
       ! during conversion

   REAL(dbl) :: atmproj_sh = 5.0d0
       ! atmproj shifthing: energy shift when computing the proj Hamiltonian

   REAL(dbl) :: atmproj_thr = 0.9d0
       ! atmproj filtering: thr on projections

   INTEGER :: atmproj_nbnd = 0
       ! atmproj filtering: number of bands used for filtering

   REAL(dbl) :: shift_L = 0.0
       ! global energy shift [eV] to be applied to the matrix elements
       ! of the left lead (H00_L, H01_L)

   REAL(dbl) :: shift_C = 0.0
       ! global energy shift [eV] to be applied to the matrix elements
       ! of the conductor region (H00_C, H_LC, H_CR)
       
   REAL(dbl) :: shift_R = 0.0
       ! global energy shift [eV] to be applied to the matrix elements
       ! of the right lead (H00_R, H01_R)

   REAL(dbl) :: shift_corr = 0.0
       ! global energy shift [eV] to be applied to the matrix elements
       ! of the correlation self-energy operator.

   INTEGER :: debug_level = 0
       ! level of debug report; values <= 0 switch the debug_mode off

   INTEGER :: ispin = 0
       ! define which spin component has to be used in the calculation.
       ! This variable is intended to exist temporarily until a full
       ! treatment of the spin degrees of freedom is not implemented.
       ! Currently it is used only within the interface with CRYSTAL09.
   LOGICAL   :: surface = .FALSE.  ! Marcio surface bandstructure
   REAL(dbl) :: efermi_bulk=  0.0  ! Marcio surface bandstructure

   NAMELIST / INPUT_CONDUCTOR / dimL, dimC, dimR, calculation_type,             &
                 conduct_formula, niterx, ne, ne_buffer, emin, emax,            &
                 nprint, delta, bias,                                           &
                 datafile_L,     datafile_C,     datafile_R, datafile_sgm,      &
                 datafile_L_sgm, datafile_C_sgm, datafile_R_sgm,                &
                 transport_dir, smearing_type, do_eigenchannels, do_eigplot,    &
                 ie_eigplot, ik_eigplot, neigchnx, carriers,                    &
                 delta_ratio, xmax, nk, s, use_symm, debug_level,               &
                 work_dir, prefix, postfix, ispin,                              &
                 write_kdata, write_lead_sgm, write_gf,                         &
                 do_orthoovp, atmproj_sh, atmproj_thr, atmproj_nbnd,            &
                 shift_L, shift_C, shift_R, shift_corr, nfailx, transfer_thr,   &
                 surface,efermi_bulk


   PUBLIC :: dimL, dimC, dimR, calculation_type, conduct_formula, niterx, smearing_type
   PUBLIC :: ne, ne_buffer, emin, emax, nprint, delta, bias, delta_ratio, xmax 
   PUBLIC :: datafile_L,     datafile_C,     datafile_R,     datafile_sgm, transport_dir, carriers
   PUBLIC :: datafile_L_sgm, datafile_C_sgm, datafile_R_sgm
   PUBLIC :: nk, s, use_symm, debug_level
   PUBLIC :: do_orthoovp, atmproj_sh, atmproj_thr, atmproj_nbnd
   PUBLIC :: work_dir, prefix, postfix, ispin, write_kdata, write_lead_sgm, write_gf
   PUBLIC :: do_eigenchannels, neigchnx, do_eigplot, ie_eigplot, ik_eigplot
   PUBLIC :: shift_L, shift_C, shift_R, shift_corr, nfailx, transfer_thr
   PUBLIC :: surface, efermi_bulk
   PUBLIC :: INPUT_CONDUCTOR


!
! ... subroutines
!

   PUBLIC :: read_namelist_input_conductor

CONTAINS

!**********************************************************
   SUBROUTINE read_namelist_input_conductor(unit)
   !**********************************************************
   !
   ! reads INPUT_CONDUCTOR namelist
   !
   IMPLICIT NONE
      INTEGER, INTENT(in)   :: unit

      CHARACTER(29) :: subname='read_namelist_input_conductor'
      LOGICAL :: allowed, exists
      INTEGER :: i, ios

      REAL(dbl) :: rydcm1 = 13.6058d0*8065.5d0
      REAL(dbl) :: amconv = 1.66042d-24/9.1095d-28*0.5d0

      CALL log_push( 'read_namelist_input_conductor' )

      IF ( ionode ) THEN
         !
         READ(unit, INPUT_CONDUCTOR, IOSTAT=ios )
         IF (ios/=0) CALL errore(subname,'reading INPUT_CONDUCTOR namelist',ABS(ios))
         !
      ENDIF


      !
      ! scale energies depending of carriers
      !
      CALL change_case(carriers,'lower')
      allowed=.FALSE.
      DO i=1,SIZE(carriers_allowed)
          IF ( TRIM(carriers) == carriers_allowed(i) ) allowed=.TRUE. 
      ENDDO
      IF (.NOT. allowed) &
          CALL errore(subname,'Invalid carriers ='//TRIM(carriers),10)

      IF ( TRIM(carriers) == 'phonons') THEN
          emin=emin**2/(rydcm1/dsqrt(amconv))**2
          IF ( emin < 0.0) CALL errore(subname,'Invalid emin',1)
          emax=emax**2/(rydcm1/dsqrt(amconv))**2
      ENDIF

      !
      ! variable bcasting
      !
      CALL mp_bcast( dimL,               ionode_id)      
      CALL mp_bcast( dimC,               ionode_id)      
      CALL mp_bcast( dimR,               ionode_id)      
      CALL mp_bcast( transport_dir,      ionode_id)      
      CALL mp_bcast( calculation_type,   ionode_id)      
      CALL mp_bcast( conduct_formula,    ionode_id)      
      CALL mp_bcast( ne,                 ionode_id)      
      CALL mp_bcast( ne_buffer,          ionode_id)      
      CALL mp_bcast( emin,               ionode_id)      
      CALL mp_bcast( emax,               ionode_id)      
      CALL mp_bcast( delta,              ionode_id)      
      CALL mp_bcast( smearing_type,      ionode_id)      
      CALL mp_bcast( delta_ratio,        ionode_id)
      CALL mp_bcast( carriers,           ionode_id)
      CALL mp_bcast( xmax,               ionode_id)      
      CALL mp_bcast( bias,               ionode_id)      
      CALL mp_bcast( nprint,             ionode_id)      
      CALL mp_bcast( niterx,             ionode_id)      
      CALL mp_bcast( write_kdata,        ionode_id)      
      CALL mp_bcast( write_lead_sgm,     ionode_id)      
      CALL mp_bcast( write_gf,           ionode_id)      
      CALL mp_bcast( nk,                 ionode_id)      
      CALL mp_bcast( s,                  ionode_id)      
      CALL mp_bcast( use_symm,           ionode_id)      
      CALL mp_bcast( debug_level,        ionode_id)      
      CALL mp_bcast( do_eigenchannels,   ionode_id)      
      CALL mp_bcast( neigchnx,           ionode_id)      
      CALL mp_bcast( do_eigplot,         ionode_id)      
      CALL mp_bcast( ie_eigplot,         ionode_id)      
      CALL mp_bcast( ik_eigplot,         ionode_id)      
      CALL mp_bcast( ispin,              ionode_id)      
      CALL mp_bcast( work_dir,           ionode_id)      
      CALL mp_bcast( prefix,             ionode_id)      
      CALL mp_bcast( postfix,            ionode_id)      
      CALL mp_bcast( datafile_L,         ionode_id)      
      CALL mp_bcast( datafile_C,         ionode_id)      
      CALL mp_bcast( datafile_R,         ionode_id)      
      CALL mp_bcast( datafile_sgm,       ionode_id)      
      CALL mp_bcast( datafile_L_sgm,     ionode_id)      
      CALL mp_bcast( datafile_C_sgm,     ionode_id)      
      CALL mp_bcast( datafile_R_sgm,     ionode_id)      
      CALL mp_bcast( do_orthoovp,        ionode_id)      
      CALL mp_bcast( atmproj_sh,         ionode_id)      
      CALL mp_bcast( atmproj_thr,        ionode_id)      
      CALL mp_bcast( atmproj_nbnd,       ionode_id)      
      CALL mp_bcast( shift_L,            ionode_id)      
      CALL mp_bcast( shift_C,            ionode_id)      
      CALL mp_bcast( shift_R,            ionode_id)      
      CALL mp_bcast( shift_corr,         ionode_id)      
      CALL mp_bcast( nfailx,             ionode_id)
      CALL mp_bcast( transfer_thr,       ionode_id)
      CALL mp_bcast( surface,            ionode_id)
      CALL mp_bcast( efermi_bulk,        ionode_id)

      !
      ! ... checking parameters
      !
      IF ( transport_dir < 1 .OR. transport_dir > 3) &
           CALL errore(subname,'Invalid transport_dir',1)

      IF ( dimC <= 0) CALL errore(subname,'Invalid dimC',1)

      IF ( LEN_TRIM(datafile_C) == 0 ) &
           CALL errore(subname,'datafile_C unspecified',1)

      INQUIRE( FILE=datafile_C, EXIST=exists )
      IF ( .NOT. exists ) CALL errore(subname,'unable to find '//TRIM(datafile_C),1)
      !
      IF ( emax <= emin )   CALL errore(subname,'Invalid EMIN EMAX',1)
      IF ( ne <= 1 )        CALL errore(subname,'Invalid NE',1)
      IF ( ne_buffer <= 0 ) CALL errore(subname,'Invalid NE_BUFFER',1)
      IF ( niterx <= 0 )    CALL errore(subname,'Invalid NITERX',1)
      IF ( nprint <= 0)     CALL errore(subname, ' nprint must be > 0 ', -nprint+1 )
      IF ( delta < ZERO )   CALL errore(subname,'Invalid DELTA',1)

      IF ( delta > 3.0_dbl* EPS_m1 ) CALL errore(subname,'DELTA too large',1)

      CALL change_case(calculation_type,'lower')
      allowed=.FALSE.
      DO i=1,SIZE(calculation_type_allowed)
          IF ( TRIM(calculation_type) == calculation_type_allowed(i) ) allowed=.TRUE. 
      ENDDO
      IF (.NOT. allowed) &
          CALL errore(subname,'Invalid calculation_type ='//TRIM(calculation_type),10)

      CALL change_case(conduct_formula,'lower')
      allowed=.FALSE.
      DO i=1,SIZE(conduct_formula_allowed)
          IF ( TRIM(conduct_formula) == conduct_formula_allowed(i) ) allowed=.TRUE. 
      ENDDO
      IF (.NOT. allowed) &
          CALL errore(subname,'Invalid conduct_formula ='//TRIM(conduct_formula),10)

      CALL change_case(smearing_type,'lower')
      allowed=.FALSE.
      DO i=1,SIZE(smearing_type_allowed)
          IF ( TRIM(smearing_type) == smearing_type_allowed(i) ) allowed=.TRUE. 
      ENDDO
      IF (.NOT. allowed) &
          CALL errore(subname,'Invalid smearing_type ='//TRIM(smearing_type),10)

      CALL change_case(carriers,'lower')
      allowed=.FALSE.
      DO i=1,SIZE(carriers_allowed)
          IF ( TRIM(carriers) == carriers_allowed(i) ) allowed=.TRUE. 
      ENDDO
      IF (.NOT. allowed) &
          CALL errore(subname,'Invalid carriers ='//TRIM(carriers),10)

      IF ( TRIM(calculation_type) == 'conductor' ) THEN
           IF ( dimL <= 0) CALL errore(subname,'Invalid dimL',1)
           IF ( dimR <= 0) CALL errore(subname,'Invalid dimR',1)
           !
           IF ( LEN_TRIM(datafile_L) == 0 ) &
                CALL errore(subname,'datafile_L unspecified',1)
           IF ( LEN_TRIM(datafile_R) == 0 ) &
                CALL errore(subname,'datafile_R unspecified',1)

           !
           INQUIRE( FILE=datafile_L, EXIST=exists )
           IF ( .NOT. exists ) CALL errore(subname,'unable to find '//TRIM(datafile_L),1)
           ! 
           INQUIRE( FILE=datafile_R, EXIST=exists )
           IF ( .NOT. exists ) CALL errore(subname,'unable to find '//TRIM(datafile_R),1)
           !
      ELSE
           !
           ! bulk case
           !
           IF ( dimL /= 0) CALL errore(subname,'dimL should not be specified',1)
           IF ( dimR /= 0) CALL errore(subname,'dimR should not be specified',1)
           dimL = dimC
           dimR = dimC
           IF ( LEN_TRIM(datafile_L) /= 0 ) &
                CALL errore(subname,'datafile_L should not be specified',1)
           IF ( LEN_TRIM(datafile_R) /= 0 ) &
                CALL errore(subname,'datafile_R should not be specified',1)
      ENDIF

      IF ( ANY( nk(:) < 0 ) ) CALL errore(subname,'invalid nk', 10 )
      IF ( ANY( s(:)  < 0 .OR.  s(:) > 1 ) ) CALL errore(subname,'invalid s', 10 )

      IF ( xmax < 10.0_dbl )      CALL errore(subname,'xmax too small',1)
      IF ( delta_ratio < ZERO )   CALL errore(subname,'delta_ratio is negative',1)
      IF ( delta_ratio > EPS_m1 ) CALL errore(subname,'delta_ratio too large',1)

      IF ( TRIM(conduct_formula) /= 'landauer' .AND. &
                ( LEN_TRIM (datafile_sgm) == 0 .AND. LEN_TRIM (datafile_C_sgm) == 0 ) ) &
           CALL errore(subname,'invalid conduct formula',1)

!      IF ( LEN_TRIM (datafile_sgm) == 0 .AND.  &
!           ( LEN_TRIM (datafile_C_sgm) /= 0  .OR. LEN_TRIM (datafile_L_sgm) /= 0 ) .OR. &
!             LEN_TRIM (datafile_R_sgm) /= 0  ) &
!           CALL errore(subname,'use datafile_C_sgm instead of datafile_sgm',1)

      IF ( ispin < 0 ) CALL errore(subname, 'ispin too small', 1) 
      IF ( ispin > 2 ) CALL errore(subname, 'ispin too large', 2) 

      IF ( do_eigplot .AND. .NOT. do_eigenchannels ) &
          CALL errore(subname,'eigplot needs eigchannels',1)

      IF ( neigchnx < 0 )   CALL errore(subname,'invalid eigchnx < 0',1)

      IF ( ie_eigplot < 0 ) CALL errore(subname,'invalid ie_eigplot < 0',1)
      IF ( ik_eigplot < 0 ) CALL errore(subname,'invalid ik_eigplot < 0',1)
      !
      IF ( ie_eigplot > 0 .AND. .NOT. do_eigplot ) &
          CALL errore(subname,'ie_eigplot needs do_eigplot',1)
      IF ( ik_eigplot > 0 .AND. .NOT. do_eigplot ) &
          CALL errore(subname,'ik_eigplot needs do_eigplot',1)
      !
      !
      IF ( ( write_lead_sgm .OR. write_gf ) .AND. use_symm ) &
           CALL errore(subname,'use_symm and write_sgm or write_gf not implemented',1)

      IF ( transfer_thr <= ZERO ) CALL errore(subname,'invalid value for transfer_thr',10)

      IF ( atmproj_thr > 1.0d0 .OR. atmproj_thr < 0.0d0) &
                                  CALL errore(subname, 'invalid atmproj_thr', 10 )
      IF ( atmproj_nbnd < 0)      CALL errore(subname, 'invalid atmproj_nbnd', 10 )

      CALL log_pop( 'read_namelist_input_conductor' )

   END SUBROUTINE read_namelist_input_conductor


END MODULE T_input_parameters_module

