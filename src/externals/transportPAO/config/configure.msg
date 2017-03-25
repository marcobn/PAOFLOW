--------------------------------------------------------------------
WanT can take advantage of several optimized numerical libraries
(essl, fftw, mkl...).  This configure script attempts to find them,
but may fail if they have been installed in non-standard locations.

The following libraries have been found:

  BLAS_LIBS= -lblas 
  LAPACK_LIBS= -llapack  -lblas 
  FFT_LIBS= -lfftw3 
  IOTK_LIBS=-L/home/marco/Programs/transportPAO/extlibs/iotk/lib -liotk
  @delete@
  @delete@
  @delete@
  @delete@
  @delete@
  @delete@
  @delete@

Please check if that is correct.

If any libraries are missing, you may specify a list of directories
to search and retry, as follows:
  ./configure LIBDIRS="list of directories, separated by spaces"

For more info, read the WanT Manual (docs/want_manual.pdf).
--------------------------------------------------------------------
