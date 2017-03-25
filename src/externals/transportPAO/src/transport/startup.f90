! 
! Copyright (C) 2004 WanT Group, 2017 ERMES Group
! 
! This file is distributed under the terms of the 
! GNU General Public License. See the file `License' 
! in the root directory of the present distribution, 
! or http://www.gnu.org/copyleft/gpl.txt . 
! 
#include "configure.h"
#include "build_date.h"
!
!**********************************************************
   SUBROUTINE startup(version,main_name)
   !**********************************************************
   !
   ! This routine initializes the code.
   ! Parallel MPI initializations should also
   ! be handled by this routines
   !
   USE io_global_module,     ONLY : stdout, ionode, ionode_id, &
                                    io_global_start, io_global_getionode
   USE timing_module,        ONLY : nclockx, timing, timing_allocate
   USE mp,                   ONLY : mp_start, mp_env
   USE mp_global,            ONLY : mpime, nproc, root, group, mp_global_start
   !
#if defined __CUDA
   USE cuda_env
#endif
   !
   IMPLICIT NONE

   !
   ! input variables
   !
   CHARACTER(*), INTENT(in) :: version
   CHARACTER(*), INTENT(in) :: main_name

   !
   ! local variables
   !
   CHARACTER(9)        :: cdate, ctime
   !
#if defined __OPENMP
    INTEGER, EXTERNAL  :: omp_get_max_threads
#endif
      
!--------------------------------------------------

   !
   ! MPI initializations
   !
   root = 0
   CALL mp_start()
   CALL mp_env(nproc,mpime,group)
   CALL mp_global_start( root, mpime, group, nproc)
   !  mpime = procesor number, starting from 0
   !  nproc = number of processors
   !  group = group index
   !  root  = index of the root processor

   !
   ! GPU/CUDA, phiGEMM
   !
#if ( defined __CUDA || defined __PHIGEMM )
   !CALL selfPhigemmInit()
   CALL InitCudaEnv()
#endif

   !
   ! IO initializations
   !
   CALL io_global_start( mpime, root )
   CALL io_global_getionode( ionode, ionode_id )

   !
   ! initilize clocks and timing
   !
   CALL timing_allocate(nclockx)
   CALL timing(TRIM(main_name),OPR="start")
   !
   ! description
   ! 
   CALL date_and_tim(cdate,ctime)
   !
   IF ( ionode ) THEN
       !
       WRITE( stdout, "(/,2x,70('=') )" ) 
       WRITE( stdout, "(a)" ) '              =                                            ='
       WRITE( stdout, "(a)" ) '              =     *** WanT *** Wannier Transport Code    ='
       WRITE( stdout, "(a)" ) '              =        (www.wannier-transport.org)         ='
       WRITE( stdout, "(a)" ) '              =      Ultra Soft Pseudopotential Implem.    ='
       WRITE( stdout, "(a)" ) '              =                                            ='
       WRITE( stdout, "(2x,70('='),2/ )" ) 
       !
       WRITE( stdout, FMT='(2x,"Program <",a,">  v. ",a,"  starts ..." )') &
                      TRIM(main_name),TRIM(version) 
       WRITE( stdout, FMT='(2x,"Date ",A9," at ",A9,/ )') cdate, ctime
       !
#ifdef __PARA
       WRITE( stdout, '(5x,"Number of MPI processes:    ",i4,/ )') nproc
#else
       WRITE( stdout, '(5x,"Serial run.",/ )')
#endif
       !
#ifdef __OPENMP
       WRITE( stdout, '(5X,"Threads/MPI process:        ",i4)' ) &
            omp_get_max_threads()
#endif
       !
#ifdef __CUDA
       !
       WRITE( stdout, '(5X,"Number of GPUs detected:    ",i4)' ) ngpus_detected
       !
       WRITE( stdout, '(5X,"Number of GPUs used:        ",i4)' ) ngpus_used
       !
#endif
       WRITE( stdout, "(/)")
   ENDIF

   !
   ! architecture / compilation details
   !
   IF ( ionode ) THEN
       !
       WRITE( stdout, "(2x,'        BUILT :',4x,a)" ) &
           TRIM( ADJUSTL( __CONF_BUILD_DATE  ))
       !
#ifdef __HAVE_CONFIG_INFO
       !
       WRITE( stdout, "(2x,'         HOST :',4x,a)" ) &
           TRIM( ADJUSTL( __CONF_HOST        ))
       WRITE( stdout, "(2x,'         ARCH :',4x,a)" ) &
           TRIM( ADJUSTL( __CONF_ARCH        ))
       WRITE( stdout, "(2x,'           CC :',4x,a)" ) &
           TRIM( ADJUSTL( __CONF_CC          ))
       WRITE( stdout, "(2x,'          CPP :',4x,a)" ) &
           TRIM( ADJUSTL( __CONF_CPP         ))
       WRITE( stdout, "(2x,'          F90 :',4x,a)" ) &
           TRIM( ADJUSTL( __CONF_MPIF90      ))
       WRITE( stdout, "(2x,'          F77 :',4x,a)" ) &
           TRIM( ADJUSTL( __CONF_F77         ))
       WRITE( stdout, "(2x,'       DFLAGS :',4x,a)" ) &
           TRIM( ADJUSTL( __CONF_DFLAGS      ))
       WRITE( stdout, "(2x,'    BLAS LIBS :',4x,a)" ) &
           TRIM( ADJUSTL( __CONF_BLAS_LIBS   ))
       WRITE( stdout, "(2x,'  LAPACK LIBS :',4x,a)" ) &
           TRIM( ADJUSTL( __CONF_LAPACK_LIBS ))
       WRITE( stdout, "(2x,'     FFT LIBS :',4x,a)" ) &
           TRIM( ADJUSTL( __CONF_FFT_LIBS    ))
       WRITE( stdout, "(2x,'    MASS LIBS :',4x,a)" ) &
           TRIM( ADJUSTL( __CONF_MASS_LIBS   ))
#endif
       !
       WRITE( stdout, "(/)")
       !
   ENDIF

END SUBROUTINE startup

