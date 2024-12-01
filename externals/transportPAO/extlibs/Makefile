#
#===============================
# WanT package
#===============================
#
include ../make.sys


#
# MAIN target
#
all: libblas liblapack libiotk libphiGEMM libmagma libnetcdf libetsf_io

#
# blas
#
libblas:  libblas_$(BLAS_LIBS_SWITCH)

libblas_internal:
	 if test -d blas ;    then ( cd blas;    $(MAKE) ) ; fi

libblas_external:
libblas_:


#
# lapack
#
liblapack:  liblapack_$(LAPACK_LIBS_SWITCH)

liblapack_internal:
	 if test -d lapack ;  then ( cd lapack;  $(MAKE) ) ; fi

liblapack_external:
liblapack_:


#
# iotk
#
libiotk:  libiotk_$(IOTK_LIBS_SWITCH)

libiotk_internal:
	 if test -d iotk ;  then ( cd iotk;  $(MAKE) ) ; fi

libiotk_external:
libiotk_:


#
# phiGEMM
#
libphiGEMM:  libphiGEMM_$(PHIGEMM_LIBS_SWITCH)

libphiGEMM_internal:
	 if test -d phiGEMM ;  then ( cd phiGEMM;  $(MAKE) ) ; fi

libphiGEMM_external:
libphiGEMM_:


#
# magma
#
libmagma:  libmagma_$(MAGMA_LIBS_SWITCH)

libmagma_internal:
	 if test -d magma ;  then ( cd magma;  $(MAKE) ) ; fi

libmagma_external:
libmagma_:


#
# lib netcdf
#
libnetcdf: libnetcdf_$(NETCDF_LIBS_SWITCH)

libnetcdf_internal:
	if test -d netcdf ;    then ( cd netcdf;    $(MAKE) ) ; fi

libnetcdf_external:
libnetcdf_:
        

#
# lib etsf_io
#
libetsf_io: libnetcdf libetsf_io_$(ETSF_IO_LIBS_SWITCH)

libetsf_io_internal:
	if test -d etsf_io ;   then ( cd etsf_io;   $(MAKE) ) ; fi

libetsf_io_external:
libetsf_io_:


#
# cleaning
#
clean:
	if test -d archive ; then ( cd archive; $(MAKE) $@ ) ; fi
	if test -d blas ;    then ( cd blas;    $(MAKE) $@ ) ; fi
	if test -d lapack ;  then ( cd lapack;  $(MAKE) $@ ) ; fi
	if test -d iotk ;    then ( cd iotk;    $(MAKE) $@ ) ; fi
	if test -d phiGEMM ; then ( cd phiGEMM; $(MAKE) $@ ) ; fi
	if test -d magma ;   then ( cd magma;   $(MAKE) $@ ) ; fi
	if test -d netcdf ;  then ( cd netcdf;  $(MAKE) $@ ) ; fi
	if test -d etsf_io ; then ( cd etsf_io; $(MAKE) $@ ) ; fi

clean_all:
	if test -d archive ; then ( cd archive; $(MAKE) $@ ) ; fi
	if test -d blas ;    then ( cd blas;    $(MAKE) $@ ) ; fi
	if test -d lapack ;  then ( cd lapack;  $(MAKE) $@ ) ; fi
	if test -d iotk ;    then ( cd iotk;    $(MAKE) $@ ) ; fi
	if test -d phiGEMM ; then ( cd phiGEMM; $(MAKE) $@ ) ; fi
	if test -d magma ;   then ( cd magma;   $(MAKE) $@ ) ; fi
	if test -d netcdf ;  then ( cd netcdf;  $(MAKE) $@ ) ; fi
	if test -d etsf_io ; then ( cd etsf_io; $(MAKE) $@ ) ; fi
	- rm ./install/make_blas.inc
	- rm ./install/make_lapack.inc
	- rm ./install/make_iotk.inc
	- rm ./install/make_magma.inc
	- touch ./install/make_blas.inc
	- touch ./install/make_lapack.inc
	- touch ./install/make_iotk.inc
	- touch ./install/make_magma.inc

