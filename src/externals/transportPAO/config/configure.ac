#
AC_INIT(WANT, 2.6, , want)

AC_CONFIG_AUX_DIR(config)

# -----------------------------
# options:
# parallel compilation
AC_ARG_ENABLE(parallel, 
   [AS_HELP_STRING([--enable-parallel],
       [compile for parallel execution if possible (default: yes)])],
   [set_use_parallel=1
    if   test "$enableval" = "yes" ; then
      use_parallel=1
   else
      use_parallel=0
   fi],
   [set_use_parallel=0 use_parallel=1])
# openMP compilation
AC_ARG_ENABLE(openmp, 
   [AS_HELP_STRING([--enable-openmp],
       [compile for openmp execution if possible (default: no)])],
   [if   test "$enableval" = "yes" ; then
      use_openmp=1
   else
      use_openmp=0
   fi],
   [use_openmp=0])
# shared libraries (limited implementation, maybe obsolete?)
AC_ARG_ENABLE(shared, 
   [AS_HELP_STRING([--enable-shared],
       [use shared libraries if available (default: yes)])],
   [if   test "$enableval" = "yes" ; then
      use_shared=1
   else
      use_shared=0
   fi],
   [use_shared=1])
# check C to Fortran wrappers (may cause trouble on some machines)
AC_ARG_ENABLE(wrappers, 
   [AS_HELP_STRING([--disable-wrappers],
       [disable C to fortran wrapper check (default: enabled)])],
   [if   test "$enableval" = "yes" ; then
      check_wrappers=1
   else
      check_wrappers=0
   fi],
   [check_wrappers=1])
# force internal blas switch
AC_ARG_WITH(internal-blas, 
   [AS_HELP_STRING([--with-internal-blas],
       [compile with internal blas (default: no)])],
    [if   test "$withval" = "yes" ; then
      use_internal_blas=1
   else
      use_internal_blas=0
   fi],
   [use_internal_blas=0])
# force internal lapack switch
AC_ARG_WITH(internal-lapack, 
   [AS_HELP_STRING([--with-internal-lapack],
       [compile with internal lapack (default: no)])],
    [if   test "$withval" = "yes" ; then
      use_internal_lapack=1
   else
      use_internal_lapack=0
   fi],
   [use_internal_lapack=0])
# AF
# force internal lapack switch
AC_ARG_WITH(internal-libs, 
   [AS_HELP_STRING([--with-internal-libs],
       [compile with internal blas, lapack, etc (default: no)])],
    [if   test "$withval" = "yes" ; then
      use_internal_blas=1
      use_internal_lapack=1
   else
      use_internal_blas=0
      use_internal_lapack=0
   fi],
   [])

# enable signals
AC_ARG_ENABLE(signals, 
   [AS_HELP_STRING([--enable-signals],
       [enable signal trapping (default: no)])],
   [if   test "$enableval" = "yes" ; then
      use_signals=1
   else
      use_signals=0
   fi],
   [use_signals=0])

# AF
# use netcdf
AC_ARG_WITH(netcdf,
   [AS_HELP_STRING([--with-netcdf],
       [use netcdf if available (default: no)])],
   [if  test "$withval" = "yes" ; then
      use_netcdf=1
   else
      use_netcdf=0
   fi],
   [use_netcdf=0])

## use etsf-io
#AC_ARG_ENABLE(etsf-io,
#   [AS_HELP_STRING([--enable-etsf-io],
#       [enable the use of etsf-io interface (default: no)])],
#   [if   test "$enableval" = "yes" ; then
#      use_etsf_io=1
#      use_netcdf=1
#   else
#      use_etsf_io=0
#   fi],
#   [use_etsf_io=0])

# use etsf-io
AC_ARG_WITH(etsf-io,
   [AS_HELP_STRING([--with-etsf-io],
       [use etsf-io lib (default: no)])],
   [if  test "$withval" = "yes" ; then
      use_etsf_io=1
      use_netcdf=1
   else
      use_etsf_io=0
   fi],
   [use_etsf_io=0])

# iotk lib
AC_ARG_WITH(external-iotk,
   [AS_HELP_STRING([--with-external-iotk],
       [compile with external iotk lib (default: no)])],
    [if   test "$withval" = "yes" ; then
      use_internal_iotk=0
   else
      use_internal_iotk=1
   fi],
   [use_internal_iotk=1])


AC_ARG_ENABLE(cuda,
   [AS_HELP_STRING([--enable-cuda],
       [enable CUDA (default: no)])],
   [if  test "$enableval" = "yes" ; then
      use_cuda=1
   else
      use_cuda=0
   fi],
   [use_cuda=0])

AC_ARG_ENABLE(magma,
   [AS_HELP_STRING([--enable-magma],
       [enable MAGMA (default: no)])],
   [if  test "$enableval" = "yes" ; then
      use_magma=1
   else
      use_magma=0
   fi],
   [use_magma=0])

AC_ARG_ENABLE(phigemm,
   [AS_HELP_STRING([--enable-phigemm],
       [enable phiGEMM (default: no)])],
   [if  test "$enableval" = "yes" ; then
      use_phigemm=1
   else
      use_phigemm=0
   fi],
   [use_phigemm=0])


AC_ARG_WITH(cuda_dir,
   [AS_HELP_STRING([--with-cuda-dir=<path>],
    [specify CUDA installation directory (default is /usr/local/cuda/)])],
    [cuda_path="$withval"],
    [cuda_path="/usr/local/cuda/"])

AC_ARG_WITH(gpu_arch,
   [AS_HELP_STRING([--with-gpu-arch=<arch>],
        [specify the GPU target architecture (default: 20)])],
        [cuda_arch="$withval"],
        [cuda_arch=20])

AC_ARG_WITH(scalapack,
   [AS_HELP_STRING([--with-scalapack],
       [use scalapack if available (default: yes)])],
   [if  test "$withval" = "yes" ; then
      with_scalapack=1
   else
      with_scalapack=0
   fi],
   [with_scalapack=1])


# -----------------------------
# -----------------------------
# packages
#

AC_ARG_WITH(scalapack, 
   [AS_HELP_STRING([--with-scalapack],
       [use scalapack if available (default: yes)])],
   [if  test "$withval" = "yes" ; then
      with_scalapack=1
   else
      with_scalapack=0
   fi],
   [with_scalapack=1])
# -----------------------------
# store variables from the environment, if set (may be or not be set)
# If set, they take precedence over configure internal choice. 
# Flags and libraries are accepted without further testing;
# compilers are tested. Specify compiler name only, not the full path
# (i.e. F90=/usr/local/bin/f90 may not work, use F90=f90)

topdir=$TOPDIR     # current directory 
arch=$ARCH         # see below for recognized architectures, plus
                   # "ppc64-mn" for MareNostrum IBM parallel machine
env_cc=$CC         # C compiler (must be in the execution path)
cpp=$CPP           # C preprocessor (as above)
cflags=$CFLAGS     # Flags for C compiler
cppflags=$CPPFLAGS # Flags for C preprocessor
dflags=$DFLAGS     # Fortran file preprocessing options, e.g. -D__DEFINE_THIS
iflags=$IFLAGS     # Location of include files - shouldn't be needed
f77=$F77           # Fortran 77 serial compiler (must be in execution path)
f90=$F90           # Fortran 90 serial compiler (must be in execution path)
mpif90=$MPIF90     # Fortran 90 parallel compiler (must be in execution path)
fflags=$FFLAGS     # Flags for Fortran 77 and 90 compilers
f90flags=$F90FLAGS # Flags for Fortran 90 compiler only
ld=$LD             # Loader (must be in the execution path)
ldflags=$LDFLAGS   # Flags for loader
ld_libs=$LD_LIBS   # Additional libraries
blas_libs=$BLAS_LIBS     # blas library - specify e.g. /my/blas/lib/libmyblas.a
                         # or -L/my/blas/lib -lmyblas
lapack_libs=$LAPACK_LIBS # lapack library, similar to above
fft_libs=$FFT_LIBS       # FFT libraries - may depend upon DFLAGS
mpi_libs=$MPI_LIBS       # MPI libraries - shouldn't be needed
mass_libs=$MASS_LIBS     # MASS libraries (IBM only)
libdirs=$LIBDIRS         # Where to look for libraries (e.g. /my/blas/lib)
scalapack_dir=$SCALAPACK_LIB  # Where to look for libscalapack.a
blacs_dir=$BLACS_LIB          # Where to look for libblacs.a
ar=$AR                   # ar (shouldn't be needed)
arflags=$ARFLAGS         # Flags for ar (as above)
pgplot_libs=$PGPLOT_LIBS  # PGPLOT libraries - used only by metadyn_pp.x 

# AF
iotk_include=$IOTK_INCLUDE
iotk_libs=$IOTK_LIBS     # iotk library, similar to above
#
netcdf_include=$NETCDF_INCLUDE
netcdf_libs=$NETCDF_LIBS
#
etsf_io_include=$ETSF_IO_INCLUDE
etsf_io_libs=$ETSF_IO_LIBS


# configure for current directory by default
if test "$topdir" = "" ; then topdir="`pwd`" ; fi

# check system type (no cross-compilation for now)
AC_CANONICAL_BUILD

# many HPC systems are configured so that running parallel programs
# interactively is disabled: on those systems, AC_PROG_F77 and AC_PROG_CC
# would fail because they can't run the compiled executables.
# to work around that, let's pretend we are cross-compiling even if we aren't
# !!! this relies on undocumented Autoconf behavior !!! 

# This is used to distinguish between true and fake cross compilation
# (only on NEC SX8 actually)
if test "$host" != "" ; then ranlib=echo; fi

cross_compiling=yes
if test "$host" = "" ; then host=$build; fi

# identify architecture
if test "$arch" = ""
then
	case $host in
		ia64-*-linux-gnu )      arch=ia64   ;;
		x86_64-*-linux-gnu )    arch=x86_64 ;;
		*-pc-linux-gnu )        arch=ia32   ;;
		*-ibm-aix* )            arch=aix    ;;
		sparc-sun-* )           arch=sparc  ;;
		i386-pc-solaris* )      arch=solaris;;
		powerpc-apple-darwin* ) arch=macppc ;;
		i686-apple-darwin* )    arch=mac686 ;;
		*-pc-cygwin )           arch=cygwin ;;
		sx*-nec* )              arch=necsx  ;;
		powerpc64-*-linux-gnu ) arch=ppc64  ;;
                * )                     AC_MSG_WARN(incorrect host name ?)
        ;;

	esac
fi
echo checking architecture... $arch

# candidate archiver and archiver flags
try_ar="ar"
try_arflags="ruv"
try_iflags=" "

# candidate fortran compilers good for all cases
try_mpif90="mpif90"
try_f90="g95 gfortran f90"

# add all needed -D options to try_dflags
try_dflags=""

# candidate compilers and flags based on architecture
case $arch in
ia32 | ia64 | x86_64 )
	try_f90="ifort pgf90 pathf95 sunf95 openf95 $try_f90"
	;;
solaris )
	try_f90="sunf95 $try_f90"
	;;
aix )
	try_mpif90="mpxlf90_r mpxlf90"
	try_f90="xlf90_r xlf90 $try_f90"
	try_arflags="-X64 ruv"
	try_dflags="-D__AIX -D__XLF"
	;;
crayxt | crayxt4 )
	try_f90="ftn $try_f90"
	;;
macppc )
	try_mpif90="mpxlf90 mpf90"
	try_f90="xlf90 $try_f90"
	try_dflags="-D__XLF"
# due to bugs in xlf 8.1 for powerpc mac os-x, the iotk library must be
# created as a dynamical library, using "libtool" - OBSOLETE

# try_arflags_dynamic="-dynamic -arch_only ppc -all_load -single_module -L/opt/ibmcmp/xlf/8.1/lib -lxlf90 -lxl -lc -o"
# try_ar="libtool"
# try_arflags="-static -o"
	;;
mac686 | cygwin )
	try_f90="ifort $try_f90"
	;;
necsx )
	sxopt=`echo $host|awk '{print substr($1,1,3)}'`
	echo $sxopt $host
	try_mpif90="sxmpif90"
	try_f90="sxf90"
	try_ar="sxar"
	try_arflags="rv"
        try_dflags='-D__SX6 ' 
        use_fft_asl=0
        use_fft_mathkeisan=1
        use_fft_para=0
# default for Nec: no parallel unless explicitly required
        if test "$set_use_parallel" -ne 1 ; then use_parallel=0 ; fi
        if test "$use_parallel" -eq 1 ; then use_fft_para=1 ; fi
        try_dflags_fft_asl='-DASL'
        try_dflags_fft_mathkeisan=' '
        try_dflags_fft_para='-D__USE_3D_FFT'
	;;
ppc64 )
	try_mpif90="mpxlf90_r mpf90_r"
	try_f90="xlf90_r $try_f90"
	try_dflags="-D__XLF"
	try_arflags="ruv"
	;;
# PowerPC MareNostrum
ppc64-mn )
	try_f90="xlf90_r"
	try_dflags="-D__XLF"
	try_arflags="ruv"
	;;
* )
	AC_MSG_WARN(unsupported architecture?)
	;;
esac

# check serial Fortran 90 compiler. This must be done before performing
# the check for the parallel compiler (section below) because option
# --disable-parallel will do very strange things otherwise. The reason
# seems to be that autoconf does not repeat all tests for the second
# occurrence of AC_PROG_F77. So the first occurrence is the one that
# must always be performed, the second is optional. PG & CC sep.2006

# use F90 if set, ignore contents of F77 (is stored in f77 for later usage)
unset F77
if test "$f90" = "" ; then f90="$try_f90" ; fi
AC_PROG_F77($f90)
f90=$F77

# check parallel Fortran 90 compiler
if test "$use_parallel" -eq 0 ; 
then
	mpif90=$f90
else
        # clear cached values (not sure when and why this is needed)
        unset F77 ac_cv_prog_ac_ct_F77 ac_cv_f77_compiler_gnu ac_cv_prog_f77_g
	if test "$mpif90" = "" ; then mpif90="$try_mpif90 $f90" ; fi
	AC_PROG_F77($mpif90)
	mpif90=$F77
fi

# check which compiler does mpif90 wrap
 
case "$arch" in
	ia32 | ia64 | x86_64 | mac686 )
	echo $ECHO_N "checking version of $mpif90... $ECHO_C"
	ifort_version=`$mpif90 -V 2>&1 | grep "Intel(R)"`
	sunf95_version=`$mpif90 -V 2>&1 | grep "^sunf95"`
	openf95_version=`$mpif90 -V 2>&1 | grep "^Open64"`
	pgf_version=`$mpif90 -V 2>&1 | grep "^pgf"`
	eko_version=`$mpif90 -v 2>&1 | grep "EKOPath"`
	g95_version=`$mpif90 -v 2>&1 | grep "g95"`
	gfortran_version=`$mpif90 -v 2>&1 | grep "gcc version"`
        #
	if test "$ifort_version" != ""
	then
		version=`$mpif90 -V 2>&1 | grep Version |
			 sed 's/.*Version//' | awk '{print $1}'`
		ifort_version=`echo $version | sed 's/\..*//'`
		echo "${ECHO_T}ifort $version"
                f90_in_mpif90="ifort"
		if test "$ifort_version" -gt 8; then
		# flags for MKL - ifort 9 and later
		   MKL_LIBS=""
	  	   if test "$ifort_version" -gt 9; then
			MKL_FLAGS="-static-intel"
		   else
			MKL_FLAGS="-i-static"
		   fi
                   if test "$use_openmp" -eq 1; then
			MKL_FLAGS="$MKL_FLAGS -openmp"
		   fi
	        else
		# flags for MKL - ifort 8 and earlier
                   MKL_LIBS="-lguide -lpthread"
                   MKL_FLAGS=""
		fi
	elif test "$sunf95_version" != ""
	then
		version=`echo $sunf95_version | awk '{print $5}'`
		echo "${ECHO_T}sunf95 $version"
                f90_in_mpif90="sunf95"
	elif test "$openf95_version" != ""
	then
		version=`echo $openf95_version | awk '{print $5}'`
		echo "${ECHO_T}openf95 $version"
                f90_in_mpif90="openf95"
	elif test "$pgf_version" != ""
	then
		version=`echo $pgf_version | awk '{print $2}'`
		echo "${ECHO_T}pgf90 $version"
                f90_in_mpif90="pgf90"
	elif test "$eko_version" != ""
	then
		version=`echo $eko_version | awk '{print $6}'`
		echo "${ECHO_T}pathf95 $version"
                f90_in_mpif90="pathf95"
	elif test "$g95_version" != ""
	then
		version=`echo $g95_version | awk '{print $3}'`
		echo "${ECHO_T}g95 $version"
                f90_in_mpif90="g95"
	elif test "$gfortran_version" != ""
	then
		version=`echo $gfortran_version | awk '{print $3}'`
		echo "${ECHO_T}gfortran $version"
                f90_in_mpif90="gfortran"
	else
		echo "${ECHO_T}unknown, assuming gfortran"
                f90_in_mpif90="gfortran"
	fi
        # check if serial and parallel compiler are the same
        if test "$f90" != "$f90_in_mpif90"; then
           AC_MSG_WARN([parallel compiler $mpif90 uses $f90_in_mpif90, but serial compiler $f90 was detected])
           AC_MSG_WARN([assuming F90=$f90_in_mpif90, discarding $f90])
        fi
        f90=$f90_in_mpif90
	;;
esac

echo setting F90... $f90
echo setting MPIF90... $mpif90

# candidate C and f77 compilers good for all cases
try_cc="cc gcc"
try_f77="$f90"

case "$arch:$f90" in
*:ifort* )
	try_cc="icc ecc $try_cc"
	;;
*:pgf90 )
	try_cc="pgcc $try_cc"
	try_f77="pgf77 $f90"
	;;
*:pathf95 )
	try_cc="pathcc $try_cc"
	;;
*:sunf95 )
	try_cc="suncc $try_cc"
	;;
*:openf95 )
	try_cc="opencc $try_cc"
	;;
aix:*xlf*_r )
	try_cc="xlc_r $try_cc"
	try_f77="xlf_r $f90"
	;;
aix:*xlf* )
	try_cc="xlc $try_cc"
	try_f77="xlf $f90"
	;;
sparc:* | solaris:sunf95 )
	try_f77="f77 $f90"
	;;
cray*:ftn )
	try_cc="cc"
	;;
macppc:xlf90 )
	try_f77="xlf $f90"
	;;
necsx:* )
	try_cc="sxcc"
	;;
ppc64:*xlf* | ppc64-mn:*xlf* )
	try_cc="xlc_r $try_cc"
	try_f77="xlf_r $f90"
	;;
esac

# check serial C compiler
if test "$env_cc" = "" ; then cc="$try_cc" ; else cc="$env_cc"; fi
AC_PROG_CC($cc)
cc=$CC

echo setting CC... $cc

# check size of pointers to int - needed to decide the size of integer
# arrays in fortran holding C pointers for FFTW

AC_CHECK_SIZEOF([int *])
SIZEOF_INT_P=$ac_cv_sizeof_int_p
AC_SUBST(SIZEOF_INT_P)
AC_CONFIG_FILES([include/fftqe_defs.h:include/fftqe_defs.h.in])

# check if the structure mallinfo is present in malloc.h
AC_CHECK_HEADER(malloc.h,have_malloc_h=1,have_malloc_h=0, )
if test "$have_malloc_h" -ne 0 
then
AC_CHECK_MEMBER([struct mallinfo.arena], 
                [AC_DEFINE(HAVE_MALLINFO)], 
                ,
	        [#include <malloc.h>])

fi

# clear cached values
unset F77 ac_cv_prog_ac_ct_F77 ac_cv_f77_compiler_gnu ac_cv_prog_f77_g

# check serial Fortran 77 compiler (use F77 if it was set)
if test "$f77" = "" ; then f77="$try_f77" ; fi
AC_PROG_F77($f77)
f77=$F77

echo setting F77... $f77

# check Fortran compiler flags
# have_cpp=0: use external C preprocessing for fortran code
# have_cpp=1: use C-like preprocessing in fortran compiler
have_cpp=1
xlf_flags=0

echo using F90... $f90

case "$arch:$f90" in
ia32:ifort* | ia64:ifort* | x86_64:ifort* | mac686:ifort*  )
	try_fflags="-O2 -assume byterecl -g -traceback"
	try_fflags_openmp="-openmp"
	try_f90flags="\$(FFLAGS) -nomodule"
	try_fflags_noopt="-O0 -assume byterecl -g -traceback"
	try_ldflags=""
	try_ldflags_static="-static"
	try_dflags="$try_dflags -D__INTEL"
	pre_fdflags="-fpp "
	;;
ia32:pgf* | ia64:pgf* | x86_64:pgf* )
# in try_fflags we use -O3 instead of -fast because some *.f files in Multigrid
# confuse the optimizer when option -fast is enabled
	try_fflags="-O3 -r8"
	try_fflags_openmp="-mp"
	try_f90flags="-fast -Mcache_align -r8"
	try_fflags_noopt="-O0"
	try_ldflags=""
	try_ldflags_static="-Bstatic"
	try_dflags="$try_dflags -D__PGI"
	have_cpp=0
	;;
ia32:path* | ia64:path* | x86_64:path* )
	try_fflags="-march=auto -O2"
	try_f90flags="\$(FFLAGS)"
	try_fflags_noopt="-O0"
	try_ldflags=""
	try_ldflags_static="-static"
	try_dflags="$try_dflags -D__EKO"
	have_cpp=0
	;;
*:g95 )
	try_fflags="-O2 -cpp"
	try_f90flags="\$(FFLAGS)"
	try_fflags_noopt="-O0 -cpp"
	try_ldflags=""
	try_ldflags_static="-static"
	;;
*:gfortran )
	try_fflags="-O2 -g"
	try_fflags_openmp="-fopenmp"
	try_f90flags="\$(FFLAGS) -x f95-cpp-input"
	try_fflags_noopt="-O0 -g"
	try_ldflags="-g"
	try_dflags="$try_dflags -D__GFORTRAN -D__STD_F95"
	try_ldflags_static="-static"
        ;;
*:sunf95 )
	try_fflags="-O4"
	try_fflags_openmp="-openmp"
	try_f90flags="\$(FFLAGS) -fpp"
	try_fflags_noopt="-O0"
	try_ldflags="-fast"
	try_ldflags_static="-Bstatic"
	imod="-M"
	;;
*:openf95 )
	try_fflags="-O3"
	try_f90flags="\$(FFLAGS) -ftpp"
	try_fflags_noopt="-O0"
	try_ldflags=""
	imod="-I"
	;;
aix:*xlf* )
	try_fflags="-q64 -qalias=noaryovrlp -O3 -qstrict \
-qarch=auto -qtune=auto -qdpc -Q -qalias=nointptr"
        try_fflags_openmp="-qsmp=omp"
	try_f90flags="\$(FFLAGS) -qsuffix=cpp=f90 -qfree=f90"
	try_fflags_noopt="-q64 -O0"
	try_ldflags="-q64"
	# try_ldflags_static="-bstatic"
	pre_fdflags="-WF,"
	xlf_flags=1
	;;
solaris:sunf95 )
	try_fflags="-fast -O2 -fpp"
	try_f90flags="\$(FFLAGS)"
	try_fflags_noopt="-O0 "
	try_ldflags=""
	imod="-M"
	;;
sparc:f90 )
	try_fflags="-fast -O1 -nodepend -xvector=no -xchip=ultra3 \
-xarch=v8plusb -xlic_lib=sunperf"
	try_f90flags="\$(FFLAGS)"
	try_fflags_noopt="-O0 -xlic_lib=sunperf"
	try_ldflags=""
	imod="-M"
        have_cpp=0
	;;
crayxt*:ftn )
# see comment above for pgf*
	try_fflags_openmp="-mp"
        try_fflags="-O3 -r8"
	try_f90flags="-fast -Mcache_align -r8"
        try_fflags_noopt="-O0"
        try_ldflags_openmp="-mp"
        try_ldflags="-v"
	try_dflags="$try_dflags -D__PGI"
	have_cpp=0
	;;
macppc:* )
	try_fflags="-O4 -qarch=auto -qtune=auto -qsuffix=cpp=f90 \
-qdpc -qalias=nointptr"
	try_f90flags="\$(FFLAGS) -qfree=f90"
	try_fflags_noopt="-O0"
	try_ldflags=""
	pre_fdflags="-WF,"
	xlf_flags=1
	;;

necsx:* )
        try_fflags='      -float0 -Cvopt -eab -R5 -Wf,-Ncont,-A dbl4,-P nh,-ptr byte,-pvctl noifopt loopcnt=9999999 expand=12 fullmsg vwork=stack,-fusion,-O noif,-init stack=nan heap=nan'
        try_f90flags='  -f2003  -float0 -Cvopt -eab -R5 -Wf,-Ncont,-A dbl4,-P nh,-ptr byte,-pvctl noifopt loopcnt=9999999 expand=12 fullmsg vwork=stack,-fusion,-O noif,-init stack=nan heap=nan'
        try_f90flags="-$sxopt $try_f90flags"
        try_fflags_noopt='-float0   '
        try_f90flags_noopt='-f2003 -float0 -eab -R5 -C debug  -Wf,-Ncont,-A dbl4,-P nh ,ptr byte,-init stack=nan heap=nan'
        try_f90flags_noopt="$try_f90flags_noopt"
        try_f90flags_inline='-f2003  -float0 -Cvopt -eab -R5 -pi noauto incdir exp=w0gauss -Wf,-Ncont,-A dbl4,-P nh,-ptr byte,-pvctl noifopt loopcnt=9999999 expand=12 fullmsg vwork=stack,-fusion,-O noif,-init stack=nan heap=nan'
        try_f90flags_inline="$try_f90flags_inline"
        try_ldflags_static='-P static'
        try_ldflags='-Wl,-f zero'
        try_ldflags="-p $try_ldflags"
        pre_fdflags=""
	;;

ppc64:*xlf* )
	try_fflags="-O4 -qsuffix=cpp=f90 -qdpc -qalias=nointptr -Q"
	try_f90flags="\$(FFLAGS) -qfree=f90"
	try_fflags_noopt="-O0"
	try_ldflags=""
	pre_fdflags="-WF,"
	xlf_flags=1
	;;

ppc64-mn:* )
	try_fflags="-O3 -q64 -qstrict -qsuffix=cpp=f90 -qdpc -qalias=nointptr -Q -qtune=ppc970 -qarch=ppc970 -qcache=auto -qhot=vector,simd -qenablevmx"
	try_f90flags="\$(FFLAGS) -qfree=f90"
	try_fflags_noopt="-O0 -q64"
	try_ldflags=""
	pre_fdflags="-WF,"
	xlf_flags=1
	;;

* )
	# unknown, try these
	try_fflags="-O"
	try_f90flags="\$(FFLAGS)"
	try_fflags_noopt="-O0"
	try_ldflags=""
        have_cpp=0
	;;

esac
if test "$use_shared" -eq 0 ; then
  try_ldflags="$try_ldflags $try_ldflags_static" ; fi

if test "$use_openmp" -eq 1 ; then 
  $try_fflags="$try_fflags $try_fflags_openmp"
  $try_ldlags="$try_ldlags $try_ldlags_openmp" 
fi

if test "$fflags" = ""   ; then fflags=$try_fflags     ; fi
if test "$f90flags" = "" ; then f90flags=$try_f90flags ; fi
if test "$fflags_noopt" = ""   ; then fflags_noopt=$try_fflags_noopt     ; fi
echo setting FFLAGS... $fflags
echo setting F90FLAGS... $f90flags
echo setting FFLAGS_NOOPT... $fflags_noopt
if test "$imod" = "" ; then imod="-I" ; fi

# tentative C and loader flags, good for many cases
try_cflags="-O3"
c_ldflags=""
try_cpp="cpp"

case "$arch:$cc" in
*:pgcc )
	try_cflags="-fast"
	;;
# pathcc -E seems to give problems when preprocessing iotk
#*:pathcc )
#	try_cpp="pathcc -E"
#	;;
aix:xlc* | aix:cc )
	try_cflags="-q64 -O2"
	c_ldflags="-q64"
	;;
*:suncc  )
	try_cflags="-fast -O"
	;;
sparc:cc )
	try_cflags="-fast -dalign -xchip=ultra3 -xarch=v8plusb \
-xlic_lib=sunperf"
	try_cpp="fpp"
	;;
crayxt*:cc )
        try_cflags="-fast"
        ;;
macppc:xlc* )
	try_cflags="-O4"
	;;
necsx:* )
	#try_cflags="-D__SX6 \$(IFLAGS) \$(MODFLAGS)"
        try_cflags=""
	try_ar="sxar"
	;;
ppc64-mn:* )
        try_cflags="-O3 -q64"
	;;
esac
if test "$cflags" = "" ; then cflags=$try_cflags ; fi
echo setting CFLAGS... $cflags

# preprocessor - try cpp in all cases; the preprocessor returned by
# AC_PROG_CPP
# may sometimes refuse to preprocess fortran files
if test "$cpp" = "" ; then cpp=$try_cpp; fi
# if test "$cpp" = "" ; then cpp=$CPP;     fi
echo setting CPP... $cpp

echo $ECHO_N "setting CPPFLAGS... $ECHO_C"
case $cpp in
	cpp)  try_cppflags="-P -traditional" ;;
	fpp)  try_cppflags="-P"              ;;
	*)    try_cppflags=""                ;;
esac
if test "$cppflags" = "" ; then cppflags=$try_cppflags ; fi
echo "${ECHO_T}$cppflags"

# linker and archiver
if test "$ld" = ""       ; then ld="$mpif90"           ; fi
if test "$ldflags" = ""  ; then ldflags="$try_ldflags" ; fi
if test "$ar" = ""       ; then ar="$try_ar"           ; fi
if test "$arflags" = ""  ; then arflags="$try_arflags" ; fi
echo setting LD... $ld
echo setting LDFLAGS... $ldflags
echo setting AR... $ar
echo setting ARFLAGS... $arflags

# compilation rules
AC_PROG_MAKE_SET
echo $ECHO_N "checking whether Fortran files must be preprocessed... $ECHO_C"
if test "$have_cpp" -ne 0
then
	f90rule="\$(MPIF90) \$(F90FLAGS) -c \$<"
	echo "${ECHO_T}no"
else
	f90rule="\$(CPP) \$(CPPFLAGS) \$< -o \$*.F90 ; \\
	\$(MPIF90) \$(F90FLAGS) -c \$*.F90 -o \$*.o"
	echo "${ECHO_T}yes"
fi

# compilation flags for all subsequent tests
# remove all $(...) because at least one compiler doesn't like them
# but if f90flags contains $(FFLAGS), substitute it
test_cflags="`echo $cflags | sed 's/\$([[^)]]*)//g'`"
test_cppflags="$test_cflags"
if test "`echo $f90flags | grep '$(FFLAGS)'`" != ""
then
	test_fflags="`echo $fflags $f90flags | sed 's/\$([[^)]]*)//g'`"
else
	test_fflags="`echo $f90flags | sed 's/\$([[^)]]*)//g'`"
fi
test_ldflags="`echo $ldflags | sed 's/\$([[^)]]*)//g'`"

# find Fortran to C wrappers

if test "$check_wrappers" -ne 0; then
   AC_F77_WRAPPERS
fi

AC_LANG_PUSH(Fortran 77)
F77=$f90 # use Fortran 90 actually 

# Search for libraries
# Flags for needed libraries
have_blas=0
have_lapack=0
have_fft=0
have_mpi=0
have_scalapack=0
# Flags for machine-specific libraries
have_acml=0
have_atlas=0
have_essl=0
have_mkl=0
have_cuda=0
have_phigemm=0


# check for blas
# supported vendor replacements:
#   mkl and acml on Intel/AMD architectures
#   essl on aix
#   SUNperf on sparc
# atlas is used over blas if available
# internal version is used if none is found
if test "$use_internal_blas" -eq 0 
then
   if test "$blas_libs" = "" 
   then
	# check directories in LD_LIBRARY_PATH too
	# (maybe they are already searched by default, but I'm not sure)
	ld_library_path=`echo $LD_LIBRARY_PATH | sed 's/:/ /g'`
	
	case "$arch:$f90" in

	x86_64:pgf* | x86_64:path* | x86_64:openf95)
		# check for acml - note that it contains lapack as well
                try_libdirs="/opt/acml*/pathscale64/lib/"
                try_libdirs="$libdirs $ld_library_path $try_libdirs"

		for dir in none $try_libdirs
		do
			unset ac_cv_search_dgemm # clear cached value
			if test "$dir" = "none"
			then
				try_loption=
			else
				echo $ECHO_N "in $dir: " $ECHO_C
				try_loption="-L$dir"
			fi
			FFLAGS="$test_fflags"
			LDFLAGS="$test_ldflags $try_loption"
 			LIBS=""
			AC_SEARCH_LIBS(dgemm, acml, have_blas=1 have_lapack=1
				have_acml=1 blas_libs="$try_loption $LIBS")
			if test "$ac_cv_search_dgemm" != "no"
			then break ; fi
		done
       	        ;;

	ia64:* )
		# check for mkl (in several directories)
		try_libdirs="/opt/intel/Compiler/*/*/mkl/lib/64
		             /opt/intel/mkl/*/lib/64
			     /opt/intel/mkl*/lib/64"
		try_libdirs="$libdirs $try_libdirs $ld_library_path"

		for dir in none $try_libdirs
		do
			unset ac_cv_search_dgemm # clear cached value
			if test "$dir" = "none"
			then
				try_loption="-L$topdir/install"
			else
				echo $ECHO_N "in $dir: " $ECHO_C
				try_loption="-L$topdir/install -L$dir"
			fi
			FFLAGS="$test_fflags"
			LDFLAGS="$MKL_FLAGS $test_ldflags $try_loption"
			LIBS="$MKL_LIBS"
			# 
			# unthreaded MKL: use install/mkl_ipf_nothread
			# multithreaded MKL: use install/mkl_ipf
			# (should work for MKL 10.2 and later as well)
			#
                        if test "$use_openmp" -eq 0; then
			   AC_SEARCH_LIBS(dgemm, mkl_ipf_nothread,
					 have_blas=1 have_mkl=1
				         blas_libs="$try_loption $LIBS"
                                         ldflags="$MKL_FLAGS $ldflags")
                        else
			   AC_SEARCH_LIBS(dgemm, mkl_ipf,
					 have_blas=1 have_mkl=1
				         blas_libs="$try_loption $LIBS"
                                         ldflags="$MKL_FLAGS $ldflags")
			fi
			if test "$ac_cv_search_dgemm" != "no"
			then break ; fi
		done
		;;

	*:sunf95 )
		# check for acml - note that it contains lapack as well
		if test "$arch" = "x86_64"
		then
			try_libdirs="/usr/local/sunstudio*/lib/amd64/"
		else
			try_libdirs="/usr/local/sunstudio*/lib/"
		fi
		try_libdirs="$libdirs $ld_library_path $try_libdirs"

		for dir in none $try_libdirs
		do
			unset ac_cv_search_dgemm # clear cached value
			if test "$dir" = "none"
			then
				try_loption=
			else
				echo $ECHO_N "in $dir: " $ECHO_C
				try_loption="-L$dir"
			fi
			FFLAGS="$test_fflags"
			LDFLAGS="$test_ldflags $try_loption"
			LIBS=""
			AC_SEARCH_LIBS(dgemm, sunperf, have_blas=1 have_lapack=1
				blas_libs="$try_loption $LIBS")
			if test "$ac_cv_search_dgemm" != "no"
			then break ; fi
		done
		;;

	x86_64:* )
                try_libdirs="/opt/intel/Compiler/*/*/mkl/lib/em64t
                             /opt/intel/mkl/*/lib/em64t
                             /opt/intel/mkl*/lib/em64t"
		try_libdirs="$libdirs $try_libdirs $ld_library_path"

		for dir in none $try_libdirs
		do
			unset ac_cv_search_dgemm # clear cached value
			if test "$dir" = "none"
			then
				try_loption="-L$topdir/install"
			else
				echo $ECHO_N "in $dir: " $ECHO_C
				try_loption="-L$topdir/install -L$dir"
			fi
			FFLAGS="$test_fflags"
			LDFLAGS="$MKL_FLAGS $test_ldflags $try_loption"
			LIBS="$MKL_LIBS"	
			# 
			# unthreaded MKL: use install/mkl_em64t_nothread
			# multithreaded MKL: use install/mkl_em64t
			# (should work for MKL 10.2 and later as well)
			#
                        if test "$use_openmp" -eq 0; then
   		           AC_SEARCH_LIBS(dgemm, mkl_em64t_nothread,
				have_blas=1 have_mkl=1
				blas_libs="$try_loption $LIBS"
                                ldflags="$MKL_FLAGS $ldflags" )
			else
			   AC_SEARCH_LIBS(dgemm, mkl_em64t, 
				have_blas=1 have_mkl=1
				blas_libs="$try_loption $LIBS"
                                ldflags="$MKL_FLAGS $ldflags")
			fi
			if test "$ac_cv_search_dgemm" != "no"
			then break ; fi
       		done
		;;

	ia32:* )
		# check for mkl (in several directories)
                try_libdirs="/opt/intel/Compiler/*/*/mkl/lib/32
                             /opt/intel/mkl/*/lib/32
                             /opt/intel/mkl*/lib/32"
		try_libdirs="$libdirs $try_libdirs $ld_library_path"

		for dir in none $try_libdirs
		do
			unset ac_cv_search_dgemm # clear cached value
			if test "$dir" = "none"
			then
				try_loption="-L$topdir/install"
			else
				echo $ECHO_N "in $dir: " $ECHO_C
				try_loption="-L$topdir/install -L$dir"
			fi
			FFLAGS="$test_fflags"
			LDFLAGS="$MKL_FLAGS $test_ldflags $try_loption"
			LIBS="$MKL_LIBS"
			# unthreaded MKL: use install/mkl_ia32_nothread
			# multithreaded MKL: use install/mkl_ia32
			# (should work for MKL 10.2 and later as well)
			#
                        if test "$use_openmp" -eq 0; then
   		           AC_SEARCH_LIBS(dgemm, mkl_ia32_nothread,
				have_blas=1 have_mkl=1
				blas_libs="$try_loption $LIBS"
                                ldflags="$MKL_FLAGS $ldflags" )
			else
			   AC_SEARCH_LIBS(dgemm, mkl_ia32, 
				have_blas=1 have_mkl=1
				blas_libs="$try_loption $LIBS"
                                ldflags="$MKL_FLAGS $ldflags")
			fi
			if test "$ac_cv_search_dgemm" != "no"
			then break ; fi 

		done
		;;

	aix:* )
		# check for essl
		unset ac_cv_search_dgemm # clear cached value
		FFLAGS="$test_fflags"
		LDFLAGS="$test_ldflags"
		LIBS=""
		AC_SEARCH_LIBS(dgemm, essl, have_blas=1
			       blas_libs="$LIBS" )
                # notice that some IBM machines may not need -lessl
                # to load blas so the above test may fail
                if test "`echo $blas_libs | grep essl`" != ""
                then
                    have_essl=1 
                    try_dflags="$try_dflags -D__ESSL"
                fi
		;;

	sparc:* | solaris:* )
		# check for SUNperf library
		unset ac_cv_search_dgemm # clear cached value
		FFLAGS="$test_fflags"
		LDFLAGS="$test_ldflags"
		LIBS=""
		AC_SEARCH_LIBS(dgemm, sunperf, have_blas=1 have_lapack=1
                               blas_libs="-xlic_lib=sunperf $LIBS")
		;;
	necsx:* )
                #sx5-nec or sx6-nec or sx8-nec: check in (/SX)/usr/lib 
                #sx8-nec-idris: check in /SX/opt/mathkeisan/inst/lib0
                try_libdirs="/SX/usr/lib /SX/opt/mathkeisan/inst/lib0" 
                for dir in none $try_libdirs
                do
                        unset ac_cv_search_dgemm # clear cached value
                        if test "$dir" = "none"
                        then
                                try_loption=
                        else
                                echo $ECHO_N "in $dir: " $ECHO_C
                                try_loption="-L$dir"
                        fi
                        FFLAGS="$test_fflags"
                        LDFLAGS="$test_ldflags $try_loption"
                        LIBS=""
                        AC_SEARCH_LIBS(dgemm, blas, have_blas=1 
                                       blas_libs="$try_loption $LIBS")
                        if test "$ac_cv_search_dgemm" != "no"
                        then break ; fi
                 done
                 ;;
	ppc64:* )
		# check for essl
		unset ac_cv_search_dgemm # clear cached value
		FFLAGS="$test_fflags"
		LDFLAGS="$test_ldflags"
		LIBS=""
		AC_SEARCH_LIBS(dgemm, essl, have_blas=1
			       blas_libs="$LIBS" )
                # notice that some IBM machines may not need -lessl
                # to load blas so the above test may fail
                if test "`echo $blas_libs | grep essl`" != ""
                then
                    have_essl=1 
                    try_dflags="$try_dflags -D__LINUX_ESSL"
                fi
		;;
        ppc64-mn:* )
                # assume essl
                unset ac_cv_search_dgemm # clear cached value
                FFLAGS="$test_fflags"
                LDFLAGS="$test_ldflags"
                have_blas=1
                have_essl=1
                try_dflags="$try_dflags -D__LINUX_ESSL"
                ;;

	esac
	# blas not (yet) found: look for more possibilities
	if test "$have_blas" -eq 0
	then
	case "$f90" in
		pgf* )
		# check for PGI blas
		unset ac_cv_search_dgemm # clear cached value
		FFLAGS="$test_fflags"
		LDFLAGS="$test_ldflags"
		LIBS=""
		AC_SEARCH_LIBS(dgemm, blas, have_blas=1 blas_libs="$LIBS")
		;;
	esac
	fi

	# macppc :  try_libdirs="/usr/local/lib /sw/lib"
	if test "$have_blas" -eq 0
	then
		# check for atlas (in several directories)
		try_libdirs="/usr/local/lib"
		try_libdirs="$libdirs $try_libdirs $ld_library_path"

		for dir in none $try_libdirs
		do
			unset ac_cv_search_dgemm # clear cached value
			if test "$dir" = "none"
			then
				try_loption=
			else
				echo $ECHO_N "in $dir: " $ECHO_C
				try_loption="-L$dir"
			fi
			FFLAGS="$test_fflags"
			LDFLAGS="$test_ldflags $try_loption" 
			LIBS="-latlas" 
			AC_SEARCH_LIBS(dgemm, f77blas, have_blas=1 have_atlas=1
				       blas_libs="$try_loption $LIBS", , -lg2c)
			if test "$ac_cv_search_dgemm" != "no"
			then break ; fi
		done
	fi

	# blas still not found

	if test "$have_blas" -eq 0
	then
		# check for blas (in several directories)
		try_libdirs="/usr/local/lib"
		try_libdirs="$libdirs $try_libdirs $ld_library_path"

		for dir in none $try_libdirs
		do
			unset ac_cv_search_dgemm # clear cached value
			if test "$dir" = "none"
			then
				try_loption=
			else
				echo $ECHO_N "in $dir: " $ECHO_C
				try_loption="-L$dir"
			fi
			FFLAGS="$test_fflags"
			LDFLAGS="$test_ldflags $try_loption"
			LIBS=""
			AC_SEARCH_LIBS(dgemm, blas, have_blas=1
				       blas_libs="$try_loption $LIBS")
			if test "$ac_cv_search_dgemm" != "no"
			then break ; fi
		done
	fi
   else
        # blas provided in BLAS_LIBS - not checked!
	have_blas=1
   fi
fi

# check for lapack
#
# same supported vendor replacements as for blas
# internal version is used if none is found
if test "$use_internal_lapack" -eq 0 
   then
   if test "$lapack_libs" = ""
   then
	# check directories in LD_LIBRARY_PATH too
	# (maybe they are already searched by default, but I'm not sure)
	ld_library_path=`echo $LD_LIBRARY_PATH | sed 's/:/ /g'`

	case "$arch:$f90" in

	ia32:* | ia64:*| x86_64:* )
		# check for mkl_lapack (if mkl found and acml not found)
                # OBSOLESCENT - recent versions of mkl contain lapack
		if test "$have_mkl" -ne 0 && test "$have_acml" -eq 0
		then
       			unset ac_cv_search_dspev ac_lib # clear cached value
               		FFLAGS="$test_fflags"
             	  	LIBS=" $blas_libs"
			LDFLAGS="$MKL_FLAGS $test_ldflags"
			AC_SEARCH_LIBS(dspev, mkl_lapack, have_lapack=1)
		        if test "$ac_lib" != "" ; then lapack_libs="-l$ac_lib"; fi
		fi
		;;

	sparc:* )
		# check for SUNperf library
		unset ac_cv_search_dspev # clear cached value
		FFLAGS="$test_fflags"
		LDFLAGS="$test_ldflags"
		LIBS="$blas_libs"
		AC_SEARCH_LIBS(dspev, sunperf, have_lapack=1
			       lapack_libs="-xlic_lib=sunperf $LIBS")
		;;
	aix:* )
		# check for essl
		unset ac_cv_search_dspev # clear cached value
		FFLAGS="$test_fflags"
		LDFLAGS="$test_ldflags"
		LIBS="$blas_libs"
		AC_SEARCH_LIBS(dspev, essl, have_lapack=1
		 		lapack_libs="-lessl" 
 				try_dflags="$try_dflags -D__ESSL")
		# essl may not have been found in previous test on blas
		if test "$have_lapack" -eq 1; then have_essl=1; fi
		;;
	necsx:* )
                #sx5-nec or sx6-nec or sx8-nec: check in (/SX)/usr/lib
                #sx8-nec-idris: check in /SX/opt/mathkeisan/inst/lib0
		try_libdirs="/SX/usr/lib /SX/opt/mathkeisan/inst/lib0"
                for dir in none $try_libdirs
                do
                        unset ac_cv_search_dspev # clear cached value
                        if test "$dir" = "none"
                        then
                                try_loption=
                        else
                                echo $ECHO_N "in $dir: " $ECHO_C
                                try_loption="-L$dir"
                        fi
                        FFLAGS="$test_fflags"
                        LDFLAGS="$test_ldflags $try_loption"
                        LIBS="$blas_libs"
                        AC_SEARCH_LIBS(dspev, lapack, have_lapack=1
                                       lapack_libs="$try_loption $LIBS")
                        if test "$ac_cv_search_dspev" != "no"
                        then break ; fi
                done
                ;;
	esac

	if test "$have_lapack" -eq 0
	then
		# check for lapack (in several directories)
	        # macppc : try_libdirs="/usr/local/lib /sw/lib"
		try_libdirs="/usr/local/lib"
		try_libdirs="$libdirs $try_libdirs $ld_library_path"

		for dir in none $try_libdirs
		do
			unset ac_cv_search_dspev # clear cached value
			if test "$dir" = "none"
			then
				try_loption=
			else
				echo $ECHO_N "in $dir: " $ECHO_C
				try_loption="-L$dir"
			fi
			FFLAGS="$test_fflags"
			LDFLAGS="$test_ldflags $try_loption"
			LIBS="$blas_libs"
			AC_SEARCH_LIBS(dspev, lapack-3 lapack, have_lapack=1
				       lapack_libs="$try_loption $LIBS")
			if test "$ac_cv_search_dspev" != "no"; then
			   # essl must precede lapack (if present)
			   if test "$have_essl" -ne 0 ; then
				lapack_libs="$blas_libs $lapack_libs"
			   fi
			   break
			fi
		done
	fi

   else
        # lapack provided in LAPACK_LIBS - not checked!
	have_lapack=1
   fi
fi

# no blas library found, or internal blas required: use the built-in blas
# (blas_libs is used in the above lapack tests: do not move the following
# settings above lapack tests, which would seem a more logical place)

if test "$have_blas" -eq 0 -o "$use_internal_blas" -eq 1 ; then
    blas_libs="-L$topdir/extlibs/blas/lib -lblas"    #AF
    blas_switch="internal"
else
    blas_switch="external"
fi

# no lapack library found, or incomplete lapack found (atlas, essl),
# or internal lapack esplicitly required

if test "$have_lapack" -eq 0 -o "$use_internal_lapack" -eq 1 ; then
    lapack_libs="-L$topdir/extlibs/lapack/lib -llapack"
    lapack_switch="internal"
else 
    if test "$have_essl" -eq 1 -o "$have_atlas" -eq 1 ; then
    # IBM essl or atlas: add missing lapack routines - must be loaded after lib
    # atlas: add missing lapack routines so as to complete atlas
    # note that some compilers do not like to have multiple symbols
      lapack_libs="$lapack_libs $topdir/extlibs/lapack/lib/liblapack.a"
      lapack_switch="internal"
    else
      lapack_switch="external"
    fi
fi

echo setting BLAS_LIBS... $blas_libs
echo setting LAPACK_LIBS... $lapack_libs


# AF
if test "$use_internal_iotk" -eq 1 ; then
   iotk_libs="-L$topdir/extlibs/iotk/lib -liotk"
   iotk_include="${imod}$topdir/extlibs/iotk/include"
   iotk_switch="internal"
else
   iotk_switch="external"
fi
#
echo setting IOTK_LIBS... $iotk_libs


# check for FFT libraries (no check for explicit openmp)
# supported vendor replacements:
#   essl on aix and some IBM linux machines
#   SUNperf on sparc
#   ASL/Mathkeisan on Nec
#   acml on amd
if test "$fft_libs" = "" && test "$use_openmp" -eq 0
then
	# check directories in LD_LIBRARY_PATH too
	# (maybe they are already searched by default, but I'm not sure)
	ld_library_path=`echo $LD_LIBRARY_PATH | sed 's/:/ /g'`

	case "$arch" in
	aix )
		# check for essl
		unset ac_cv_search_dcft # clear cached value
		FFLAGS="$test_fflags"
		LDFLAGS="$test_ldflags"
		LIBS="$fft_libs"
		AC_SEARCH_LIBS(dcft, essl, have_fft=1 fft_libs="$LIBS")
	    ;;     
	ppc64 | ppc64-mn )
		# check for essl
		unset ac_cv_search_dcft # clear cached value
		FFLAGS="$test_fflags"
		LDFLAGS="$test_ldflags"
		LIBS="$fft_libs"
		AC_SEARCH_LIBS(dcft, essl, have_fft=1 fft_libs="$LIBS")
	    ;;     
	sparc )
		# check for SUNperf FFT library on Sun Sparcs
                # but not on solaris PC! it is slower than FFTW
		unset ac_cv_search_zfft3i # clear cached value
		FFLAGS="$test_fflags"
		LDFLAGS="$test_ldflags"
		LIBS="$libs"
		AC_SEARCH_LIBS(zfft3i, sunperf, have_fft=1
			       try_dflags="$try_dflags -D__SUNPERF"
			       fft_libs="-xlic_lib=sunperf $LIBS")
		;;
	necsx )
                if test "$use_fft_mathkeisan" -ne 0
                then
                   #sx5-nec or sx6-nec or sx8-nec: check in (/SX)/usr/lib
                   #sx8-nec-idris: check in /SX/opt/mathkeisan/inst/lib0
                   try_libdirs="/SX/usr/lib /SX/opt/mathkeisan/inst/lib0" 
                   #check for Mathkeisan (Cray simple precision )
                   #search for initialization subroutine
                   echo $ECHO_N  "Searching in Mathkeisan" $ECHO_C
                   for dir in none $try_libdirs
                   do
                        unset ac_cv_search_zftfax # clear cached value
                        if test "$dir" = "none"
                        then
                                try_loption=
                        else
                                echo $ECHO_N "in $dir: " $ECHO_C
                                try_loption="-L$dir"
                        fi
                        FFLAGS="$test_fflags"
                        LDFLAGS="$test_ldflags $try_loption"
                        LIBS=""
                        AC_SEARCH_LIBS(zftfax, fft, have_fft=1
                             try_dflags="$try_dflags try_dflags_fft_mathkeisan"
                                       fft_libs="$try_loption $LIBS")
                        if test "$ac_cv_search_zftfax" != "no"
                        then break ; fi
                   done
                fi
                if test "$use_fft_asl" -ne 0
                then
                   #check for asl in (/SX)/usr/lib 
                   try_libdirs="/SX/usr/lib" 
                   #search for initialization subroutine
                   echo $ECHO_N  "Searching in Asl" $ECHO_C
                   for dir in none $try_libdirs
                   do
                        unset ac_cv_search_zfc3cl # clear cached value
                        if test "$dir" = "none"
                        then
                                try_loption=
                        else
                                echo $ECHO_N "in $dir: " $ECHO_C
                                try_loption="-L$dir"
                        fi
                        FFLAGS="$test_fflags"
                        LDFLAGS="$test_ldflags $try_loption"
                        LIBS=""
                        AC_SEARCH_LIBS(zfc3cl, asl, have_fft=1 
                             asl_libs="$try_loption $LIBS" 
                             try_dflags="$try_dflags $try_dflags_fft_asl"
                             fft_libs="$fft_libs $asl_libs")
                        if test "$ac_cv_search_zfc3cl" != "no"
                        then break ; fi
                   done
                fi
                if test "$use_fft_para" -ne 0
                then
                   try_dflags="$try_dflags $try_dflags_fft_para"
                fi
                ;;
	x86_64)
		# check for acml
                case $f90 in
                pgf* | path* )
			if test "$have_acml" -ne 0
			then
				unset ac_cv_search_zfft1mx # clear cached value
				FFLAGS="$test_fflags"
				LDFLAGS="$test_ldflags"
				LIBS="$fft_libs"
				AC_SEARCH_LIBS(zfft1mx, acml, have_fft=1
					fft_libs="$LIBS"
					try_dflags="$try_dflags -D__ACML" )
		
			fi
		;;
                esac
		;;
	esac

fi

if test "$have_fft" -eq 0 && test "$use_openmp" -eq 0 
then

	# check for fftw v.3 (in several directories)
	try_libdirs="/usr/local/lib"
	try_libdirs="$libdirs $try_libdirs $ld_library_path"

	for dir in none $try_libdirs
	do
		unset ac_cv_search_dfftw_execute_dft # clear cached value
		if test "$dir" = "none"
		then
			try_loption=
		else
			echo $ECHO_N "in $dir: " $ECHO_C
			try_loption="-L$dir"
		fi
		CFLAGS="$test_cflags"
		CPPFLAGS="$test_cppflags"
		LDFLAGS="$c_ldflags $try_loption"
		LIBS="$fft_libs"
		AC_SEARCH_LIBS(dfftw_execute_dft, fftw3, have_fft=1 
			       fft_libs="$try_loption $LIBS", , -lm)
		if test "$ac_cv_search_dfftw_execute_dft" != "no"
		then
		        try_dflags="$try_dflags -D__FFTW3"
			break
		fi
	done

fi
echo setting FFT_LIBS... $fft_libs

# if no valid FFT library was found, use the local copy
if test "$have_fft" -eq 0
then
	try_dflags="$try_dflags -D__FFTW"
fi

F77=$f90
FFLAGS="$test_fflags"
LDFLAGS="$test_ldflags"
# check for mass on aix
if test "$mass_libs" = ""
then
	# check directories in LD_LIBRARY_PATH too
	# (maybe they are already searched by default, but I'm not sure)
	ld_library_path=`echo $LD_LIBRARY_PATH | sed 's/:/ /g'`

	case "$arch" in
	aix )
		# check for mass (in several directories)
		try_libdirs="/usr/local/lib /cineca/lib /cineca/lib/mass"
		try_libdirs="$libdirs $try_libdirs $ld_library_path"

		for dir in none $try_libdirs
		do
			unset ac_cv_search_vexp # clear cached value
			if test "$dir" = "none"
			then
				try_loption=
			else
				echo $ECHO_N "in $dir: " $ECHO_C
				try_loption="-L$dir"
			fi
			FFLAGS="$test_fflags"
			LDFLAGS="$test_ldflags $try_loption"
			LIBS=""
			AC_SEARCH_LIBS(vexp, massvp4 massv, , , -lmass)
			if test "$ac_cv_search_vexp" = "-lmassvp4" \
				-o "$ac_cv_search_vexp" = "-lmassv"
			then mass_libs="$try_loption $ac_cv_search_vexp -lmass"
			fi
			if test "$ac_cv_search_vexp" != "no" ; then break ; fi
		done
		;;

	ppc64* )
		# check for mass (in several directories)
		try_libdirs="/usr/local/lib /opt/ibmcmp/xlmass/*/lib64"
		try_libdirs="$libdirs $try_libdirs $ld_library_path"

		for dir in none $try_libdirs
		do
			unset ac_cv_search_vexp # clear cached value
			if test "$dir" = "none"
			then
				try_loption=
			else
				echo $ECHO_N "in $dir: " $ECHO_C
				try_loption="-L$dir"
			fi
			FFLAGS="$test_fflags"
			LDFLAGS="$test_ldflags $try_loption"
			LIBS=""
			AC_SEARCH_LIBS(vexp, massvp4_64, , , -lmass_64)
			if test "$ac_cv_search_vexp" = "-lmassvp4_64"
			then mass_libs="$try_loption $ac_cv_search_vexp -lmass_64"
			fi
			if test "$ac_cv_search_vexp" != "no" ; then break ; fi
		done
		;;

	esac
fi

if test "$mass_libs" != ""; then try_dflags="$try_dflags -D__MASS" ; fi

echo setting MASS_LIBS... $mass_libs

# check for mpi
# some architectures require to link mpi libraries explicitly
F77=$mpif90 # use parallel compiler
if test "$mpi_libs" = ""
then
	# check directories in LD_LIBRARY_PATH too
	# (maybe they are already searched by default, but I'm not sure)
	ld_library_path=`echo $LD_LIBRARY_PATH | sed 's/:/ /g'`

	if test "$use_parallel" -ne 0
	then
		if test "$have_mpi" -eq 0
			# check for mpi
		then
			unset ac_cv_search_mpi_init # clear cached value
			LDFLAGS="$test_ldflags"
			LIBS="$mpi_libs"
			AC_SEARCH_LIBS(mpi_init, mpi,
				       have_mpi=1 mpi_libs="$LIBS")
		fi
	fi
fi
echo setting MPI_LIBS... $mpi_libs

# AF
if test "$use_cuda" -eq 1 
then 
    # check for CUDA bin
    AC_ARG_VAR([NVCC], [nvcc compiler to use])
    have_valid_cuda_compiler=1
    AC_PATH_PROG(NVCC, nvcc, )
    if test -n "$nvcc"; then
        have_valid_cuda_compiler=0
    fi
    
    # check for CUDA headers
    have_valid_cuda_header=0
    if test -e "$cuda_path/include/cuda.h"; then
        have_valid_cuda_header=1
    fi

    if test "$have_valid_cuda_header" == 1  && 
        test "$have_valid_cuda_compiler" == 1 
    then 
        # check for CUDA libs
        AC_LANG_PUSH(C)
    	nvcc=$NVCC
    	
    	# AC_EGREP_HEADER does not work... yet
    	# AC_EGREP_HEADER("4000", fileh, , try_dflags="$try_dflags -D__CUDA_3")
    	val="`egrep "CUDA_VERSION 40" $cuda_path/include/cuda.h`"
    	if test -z "$val";  then
    	   cudamajor=3
    	   AC_MSG_NOTICE([CUDA 3.x detected.]) 
    	   try_dflags="$try_dflags -D__CUDA_3"
    	else
    	   AC_MSG_NOTICE([CUDA 4.x detected.]) 
    	   cudamajor=4
    	fi
    	
    	try_libdirs="$cuda_path/lib64 /usr/lib64/nvidia /usr/local/cuda/lib64"
        try_libdirs="$try_libdirs $libdirs $ld_library_path"
        for dir in $try_libdirs
        do
            unset ac_cv_search_cudaGetDeviceCount # clear cached value
            unset ac_cv_search_cublasInit
            unset ac_cv_search_cufftPlan1d
            unset ac_cv_search_cuMemGetInfo_v2
            if test "$dir" = "none"
            then
                    try_loption=
            else                  
                    try_loption="-L$dir"
            fi			
			CFLAGS="$test_cflags"
			# $test_ldflags
            LDFLAGS="$try_loption"
            LIBS=""
            
            echo $ECHO_N "in $dir: " $ECHO_C				
			AC_SEARCH_LIBS(cudaGetDeviceCount,cudart, 
			      cuda_libs="$cuda_libs $LIBS",,)
			LIBS=""
			echo $ECHO_N "in $dir: " $ECHO_C	
			AC_SEARCH_LIBS(cublasInit,cublas, 
				  cuda_libs="$cuda_libs $LIBS",,)
			
			echo $ECHO_N "in $dir: " $ECHO_C
			AC_SEARCH_LIBS(cufftPlan1d,cufft, 
				  cuda_libs="$cuda_libs $LIBS",,)	 	 			  
		 	 			  
		 	cuda_libs="$try_loption $cuda_libs"
		 	
		 	LDFLAGS=""
		 	LIBS=""
            AC_SEARCH_LIBS(cuMemGetInfo_v2,cuda, 
                  cuda_libs="$cuda_libs $LIBS",,)
                  
			if test "$ac_cv_search_cudaGetDeviceCount" != "no" && 
				test "$ac_cv_search_cublasInit" != "no" && 
				test "$ac_cv_search_cufftPlan1d" != "no"
			then				 
				
				try_dflags="$try_dflags -D__CUDA"
				
				if test "$use_debugcuda" == 1 ; then
				    try_dflags="$try_dflags -D__CUDA_DEBUG"
				fi
				
				if test "$use_pinnedmem" -eq 1  ; then
                    try_dflags="$try_dflags -D__CUDA_MEM_PINNED"
                fi
                
				if test "$ac_cv_search_cuMemGetInfo_v2" == "no" 
                then
                    try_dflags="$try_dflags -D__CUDA_GET_MEM_HACK"
                fi
				have_cuda=1
				break
			fi
    	done   	
    	AC_LANG_POP(C)  	
    else
        AC_MSG_ERROR([Please check if CUDA is correctly installed.]) 
    fi
fi

if test "$use_cuda" -eq 1 && test "$have_cuda" -eq 0; then 
    AC_MSG_ERROR([Please check if CUDA is correctly installed.]) 
fi

if test "$have_cuda" -eq 1
then
     
    # Empty inits...
    # AF
    cuda_switch="external"
    #
    magma_include=""
    magma_libs=""
    magma_switch=""
    #
    phigemm_include=""
    phigemm_libs=""
    phigemm_switch=""
                    
    # GPU architecture
    if  test "$cuda_arch" = "20" ; then
        nvccflags="-O3 --ptxas-options=-v -arch sm_20 -gencode arch=compute_20,code=compute_20"
        try_dflags="$try_dflags -D__GPU_NVIDIA_20"
        magma_gpu_target=1
    fi
    if test "$cuda_arch" = "13" ; then
        nvccflags="-O3 --ptxas-options=-v -arch sm_13 -gencode arch=compute_13,code=compute_13"
        try_dflags="$try_dflags -D__GPU_NVIDIA_13"
        magma_gpu_target=0
    fi   
    
    cuda_include="-I$cuda_path/include"
    
    # phiGEMM 
    if test "$use_phigemm" -eq 1  ; then 
    	if test "$have_mkl" -eq 1  ; then
	        phigemm_switch="internal"
	        try_dflags="$try_dflags -D__PHIGEMM"
	        phigemm_opts="-D__PHIGEMM_WEAK_INTERFACES"	       
                # AF
	        phigemm_include="-I$topdir/extlibs/phiGEMM/include"
	        phigemm_libs="$topdir/extlibs/phiGEMM/lib/libphigemm.a "    

		    if test "$have_mpi" -eq 1 && test "$cudamajor" -eq 4 ; then
			    phigemm_opts="$phigemm_opts -D__PHIGEMM_PARA"
		    fi
		    
		    if test "$use_phigemmprofile" -eq 1; then
		        phigemm_opts="$phigemm_opts -D__PHIGEMM_PROFILE"
		        try_dflags="$try_dflags -D__PHIGEMM_PROFILE"
		    fi		    
	    else
	        AC_MSG_WARN([This version of Quantum ESPRESSO requires the Intel MKL libraries to use the phiGEMM library.]) 
		fi
    fi
    
	if test "$use_multigpu" -eq 1
	then
	   if test "$cudamajor" -eq 4
	   then
            try_dflags="$try_dflags -D__MULTI_GPU"
       else
            AC_MSG_WARN([The multi-GPU features are available only using CUDA 4.x.]) 
       fi
    fi
      
    # By default show all CUDA timing    
    # try_dflags="$try_dflags -D__CUDA_QE_TIMING"
            
    if test "$use_magma" -eq 1; then
        magma_switch="internal"
        # AF
        magma_include="-I$topdir/extlibs/magma/include -I$topdir/quark/include"
        magma_libs="$topdir/extlibs/magma/lib/libmagma.a $topdir/extlibs/magma/lib/libmagmablas.a $topdir/extlibs/magma/lib/libmagma.a" 
        try_dflags="$try_dflags -D__MAGMA"
        
        # new MAGMA (1.1.0) requires this flag...
        ldflags="$ldflags -zmuldefs"
    fi
    
    try_iflags="$try_iflags $magma_include $phigemm_include $cuda_include"
    
    lapack_libs="$magma_libs $lapack_libs"
    blas_libs="$phigemm_libs $blas_libs"
        
    ld_libs="$cuda_libs $ld_libs"
	
    echo setting CUDA_LIBS... $cuda_libs

fi

# Always show CUDA timing (if CUDA is not enable, only GEMM are tracked)    
if test "$have_cuda" -eq 1
then
    try_dflags="$try_dflags -D__CUDA_QE_TIMING"
else
    try_dflags="$try_dflags"
fi
    

# final check on availability of parallel environment
parallel=0
if test "$use_parallel" -ne 0
then
	F77=$mpif90
	LIBS="$mpi_libs"
	AC_SEARCH_LIBS(mpi_init, "", parallel=1
		       try_dflags="$try_dflags -D__MPI -D__PARA")
# look for scalapack if required
        if test "$parallel" -eq 1
        then
           if test "$with_scalapack" -ne 0
           then
# no additional libraries needed
	     scalapack_libs=""
	     AC_SEARCH_LIBS(pdgemr2d, "" , have_scalapack=1 
	        try_dflags="$try_dflags -D__SCALAPACK")
             if test "$have_scalapack" -eq 0; 
             then 
# scalapack + blacs
               unset ac_cv_search_pdgemr2d # clear cached value
               scalapack_libs="-L$scalapack_dir -lscalapack"
  	       scalapack_libs="$scalapack_libs -L$blacs_dir -lblacs -lblacsF77init -lblacs"
	       LIBS="$mpi_libs $scalapack_libs"
               LDFLAGS=""
	       AC_SEARCH_LIBS(pdgemr2d, "" , have_scalapack=1 
	        try_dflags="$try_dflags -D__SCALAPACK")
             fi
             if test "$have_scalapack" -eq 0; 
             then 
# sci libraries (e.g. cray xt)
               unset ac_cv_search_pdgemr2d # clear cached value
	       scalapack_libs="-lsci"
	       LIBS="$mpi_libs $scalapack_libs"
	       AC_SEARCH_LIBS(pdgemr2d, "" , have_scalapack=1 
	         try_dflags="$try_dflags -D__SCALAPACK")
             fi
           fi
        fi
fi


# AF
# etsf-io and NetCDF
# 
if test "$etsf_io_libs" != "" ; then
    use_etsf_io=1
    etsf_io_switch="external"
fi
#
if test "$use_etsf_io" = "1" -a "$etsf_io_libs" = "" ; then
    etsf_io_libs="-L$topdir/extlibs/etsf_io/lib -letsf_io"
    etsf_io_include="${imod}$topdir/extlibs/etsf_io/include"
    etsf_io_switch="internal"
fi
if test "$use_etsf_io" = "1" ; then try_dflags="$try_dflags -D__ETSF_IO" ; fi
#
if test "$netcdf_libs" != "" ; then
    use_netcdf=1
    netcdf_switch="external"
fi
#
if test "$use_netcdf" = "1" -a "$netcdf_libs" = "" ; then
    netcdf_libs="-L$topdir/extlibs/netcdf/lib -lnetcdf"
    netcdf_include="${imod}$topdir/extlibs/netcdf/include"
    netcdf_switch="internal"
fi
if test "$use_netcdf" = "1" ; then try_dflags="$try_dflags -D__NETCDF" ; fi



# preprocessing flag for openmp (experimental)
if test "$use_openmp" -eq 1 ; then try_dflags="$try_dflags -D__OPENMP" ; fi

# preprocessing flag for signal trapping (experimental)
if test "$use_signals" -eq 1 ; then try_dflags="$try_dflags -D__TRAP_SIGUSR1" ; fi

if test "$dflags" = "" ; then dflags="$try_dflags" ; fi
echo setting DFLAGS... $dflags


if test "$iflags" = "" ; then iflags="$try_iflags" ; fi
echo setting IFLAGS... $iflags

# xlf compilers (AIX and powerpc) want comma-separated -D directives
if test "$xlf_flags" -ne 0
then
	fdflags="`echo $dflags | sed 's/  */,/g'`"
else
	fdflags="\$(DFLAGS)"
fi
echo setting FDFLAGS... $fdflags

if test "$ranlib" != "echo"
then
   AC_CHECK_PROG(ranlib,ranlib,ranlib,echo)
fi
echo setting RANLIB... $ranlib



# configure output messages
blas_line="BLAS_LIBS=$blas_libs"
if test "$have_scalapack" -eq 1; then
   scalapack_line="SCALAPACK_LIBS=$scalapack_libs"
else
   scalapack_libs=""
   scalapack_line="@delete@"
fi
lapack_line="LAPACK_LIBS=$lapack_libs"
# AF
if test "$fft_libs" != "" ; then 
   fft_line="FFT_LIBS=$fft_libs"
else
   fft_line="@delete@" 
fi
#
if test "$mpi_libs" != "" ; then 
   mpi_line="MPI_LIBS=$mpi_libs"
else 
   mpi_line="@delete@" 
fi
if test "$arch" = "aix" ; then 
   mass_line="MASS_LIBS=$mass_libs"
else
   mass_line="@delete@"
fi

# AF
if test "$iotk_libs" != "" ; then
   iotk_line="IOTK_LIBS=$iotk_libs"
else
   iotk_line="@delete@"
fi
if test "$etsf_io_libs" != "" ; then
   etsf_io_line="ETSF_IO_LIBS=$etsf_io_libs"
else
   etsf_io_line="@delete@"
fi
if test "$netcdf_libs" != "" ; then
   netcdf_line="NETCDF_LIBS=$netcdf_libs"
else
   netcdf_line="@delete@"
fi
# AF
if test "$have_cuda" -eq 1 ; then 
   cuda_line="CUDA_LIBS=$cuda_libs"
else
   cuda_line="@delete@"
fi
if test "$phigemm_libs" != "" ; then 
   phigemm_line="PHIGEMM_LIBS=$phigemm_libs"
else
   phigemm_line="@delete@"
fi
if test "$magma_libs" != "" ; then 
   magma_line="MAGMA_LIBS=$magma_libs"
else
   magma_line="@delete@"
fi

#


if test "$use_parallel" -ne 0
then
	if test "$parallel" -ne 0
	then
		parallel_report="Parallel environment detected successfully.\\
Configured for compilation of parallel executables."
	else
		parallel_report="Parallel environment not detected \
\(is this a parallel machine?\).\\
Configured for compilation of serial executables."
	fi
else
	parallel_report="Configured for compilation of serial executables."
fi





# export settings to generated files
AC_SUBST(topdir)         # <AF>
AC_SUBST(host)           # <AF>
AC_SUBST(arch)           # <AF>
#
AC_SUBST(cc)
AC_SUBST(cflags)
AC_SUBST(dflags)
AC_SUBST(fdflags)
AC_SUBST(cpp)
AC_SUBST(cppflags)
AC_SUBST(f90)
AC_SUBST(mpif90)
AC_SUBST(f90flags)
AC_SUBST(f77)
AC_SUBST(fflags)
AC_SUBST(fflags_noopt)
AC_SUBST(pre_fdflags)
AC_SUBST(imod)
AC_SUBST(iflags)
AC_SUBST(ld)
AC_SUBST(ldflags)
AC_SUBST(ld_libs)
AC_SUBST(blas_libs)
AC_SUBST(blas_switch)
AC_SUBST(lapack_libs)
AC_SUBST(lapack_switch)
AC_SUBST(fft_libs)
AC_SUBST(mpi_libs)
AC_SUBST(mass_libs)
AC_SUBST(pgplot_libs)
AC_SUBST(scalapack_libs)
AC_SUBST(ar)
AC_SUBST(arflags)
AC_SUBST(ranlib)
AC_SUBST(f90rule)
AC_SUBST(blas_line)
AC_SUBST(lapack_line)
AC_SUBST(scalapack_line)
AC_SUBST(fft_line)
AC_SUBST(mpi_line)
AC_SUBST(mass_line)
AC_SUBST(parallel_report)
#
AC_SUBST(etsf_io_libs)         # <AF>
AC_SUBST(etsf_io_switch)       # <AF>
AC_SUBST(etsf_io_include)      # <AF>
AC_SUBST(etsf_io_line)         # <AF>
#
AC_SUBST(netcdf_libs)          # <AF>
AC_SUBST(netcdf_switch)        # <AF>
AC_SUBST(netcdf_include)       # <AF>
AC_SUBST(netcdf_line)          # <AF>
#
AC_SUBST(iotk_libs)            # <AF>
AC_SUBST(iotk_switch)          # <AF>
AC_SUBST(iotk_include)         # <AF>
AC_SUBST(iotk_line)            # <AF>
#
# AF, CUDA stuff 
AC_SUBST(nvcc)
AC_SUBST(nvccflags)
AC_SUBST(cudamajor)
AC_SUBST(cuda_path)
AC_SUBST(cuda_include)
AC_SUBST(cuda_libs)
AC_SUBST(cuda_switch)
AC_SUBST(cuda_line)
#
AC_SUBST(phigemm_cc)
AC_SUBST(phigemm_fc)
AC_SUBST(phigemm_opts)
AC_SUBST(phigemm_include)
AC_SUBST(phigemm_libs)
AC_SUBST(phigemm_switch)
AC_SUBST(phigemm_line)
#
AC_SUBST(magma_include)
AC_SUBST(magma_libs)
AC_SUBST(magma_switch)
AC_SUBST(magma_line)
AC_SUBST(magma_gpu_target)

AC_CONFIG_HEADERS([include/c_defs.h:include/c_defs.h.in])
AC_CONFIG_FILES([include/configure.h:include/configure.h.in]) # AF
AC_CONFIG_FILES([make.sys:config/make.sys.in])
AC_CONFIG_FILES([config/configure.msg:config/configure.msg.in])

AC_CONFIG_FILES(extlibs/install/make_blas.inc)
AC_CONFIG_FILES(extlibs/install/make_lapack.inc)
AC_CONFIG_FILES(extlibs/install/make_magma.inc) # AF
AC_CONFIG_FILES(extlibs/install/make_iotk.inc)  # AF


AC_OUTPUT

# AF
# greate also f_defs.h with a (not so) dirty trick
#
if test -e include/c_defs.h
then
   test -e include/f_defs.h && rm include/f_defs.h
   grep "#define " include/c_defs.h > include/f_defs.h
fi
# AF

# AF
# setup for configuring extlibs
#
if test "$iotk_switch" = "internal" ; then
   test -e ./extlibs/iotk/configure-stamp && rm ./extlibs/iotk/configure-stamp
fi
if test "$use_phigemm" = "1" ; then
   test -e ./extlibs/phiGEMM/fconfigure-stamp && rm ./extlibs/phiGEMM/configure-stamp
fi
if test "$use_etsf_io" = "1" ; then
   test -e ./extlibs/etsf_io/configure-stamp && rm ./extlibs/etsf_io/configure-stamp
fi
if test "$use_netcdf" = "1" ; then
   test -e ./extlibs/netcdf/configure-stamp && rm ./extlibs/netcdf/configure-stamp
fi



# final messages
sed '/@delete@/d' ./config/configure.msg
echo configure: success
