#
#===============================
# WanT package
#===============================
#

#
# Package List
#
include ./package.list

all: $(tarball_blas) $(tarball_lapack) $(tarball_iotk) $(tarball_netcdf) $(tarball_etsf_io) \
     $(tarball_magma) $(tarball_phigemm)

$(tarball_blas):
	wget -O $(tarball_blas) $(url_blas) > /dev/null 2>&1
	@if test "`echo $$?`" -ne "0" ; then \
		curl -o $(tarball_blas) $(url_blas) > /dev/null 2>&1 ; \
	fi
	@if test ! -s $(tarball_blas) ; then \
		echo "*** Unable to download BLAS. Test whether curl or wget is installed and working," ; \
		echo "*** if you have direct access to the internet." ; \
		echo "*** If not, copy $(tarball_blas) into extlibs/archive/" ; \
	exit 1 ; fi

$(tarball_lapack):
	wget -O $(tarball_lapack) $(url_lapack) > /dev/null 2>&1
	@if test "`echo $$?`" -ne "0" ; then \
		curl -o $(tarball_lapack) $(url_lapack) > /dev/null 2>&1 ; \
	fi
	@if test ! -s $(tarball_lapack) ; then \
		echo "*** Unable to download LAPACK. Test whether curl or wget is installed and working," ; \
		echo "*** if you have direct access to the internet." ; \
		echo "*** If not, copy $(tarball_lapack) into extlibs/archive/" ; \
	exit 1 ; fi

$(tarball_iotk):
	wget -O $(tarball_iotk) $(url_iotk) > /dev/null 2>&1
	@if test "`echo $$?`" -ne "0" ; then \
		curl -o $(tarball_iotk) $(url_iotk) > /dev/null 2>&1 ; \
	fi
	@if test ! -s $(tarball_iotk) ; then \
		echo "*** Unable to download IOTK. Test whether curl or wget is installed and working," ; \
		echo "*** if you have direct access to the internet." ; \
		echo "*** If not, copy $(tarball_iotk) into extlibs/archive/" ; \
	exit 1 ; fi

$(tarball_netcdf):
	wget -O $(tarball_netcdf) $(url_netcdf) > /dev/null 2>&1
	@if test "`echo $$?`" -ne "0" ; then \
		curl -o $(tarball_netcdf) $(url_netcdf) > /dev/null 2>&1 ; \
	fi
	@if test ! -s $(tarball_netcdf) ; then \
		echo "*** Unable to download NETCDF. Test whether curl or wget is installed and working," ; \
		echo "*** if you have direct access to the internet." ; \
		echo "*** If not, copy $(tarball_netcdf) into extlibs/archive/" ; \
	exit 1 ; fi

$(tarball_etsf_io):
	wget -O $(tarball_etsf_io) $(url_etsf_io) > /dev/null 2>&1
	@if test "`echo $$?`" -ne "0" ; then \
		curl -o $(tarball_etsf_io) $(url_etsf_io) > /dev/null 2>&1 ; \
	fi
	@if test ! -s $(tarball_etsf_io) ; then \
		echo "*** Unable to download ETSF-IO. Test whether curl or wget is installed and working," ; \
		echo "*** if you have direct access to the internet." ; \
		echo "*** If not, copy $(tarball_etsf_io) into extlibs/archive/" ; \
	exit 1 ; fi

$(tarball_magma):
	wget -O $(tarball_magma) $(url_magma) > /dev/null 2>&1
	@if test "`echo $$?`" -ne "0" ; then \
		curl -o $(tarball_magma) $(url_magma) > /dev/null 2>&1 ; \
	fi
	@if test ! -s $(tarball_magma) ; then \
		echo "*** Unable to download MAGMA. Test whether curl or wget is installed and working," ; \
		echo "*** if you have direct access to the internet." ; \
		echo "*** If not, copy $(tarball_magma) into extlibs/archive/" ; \
	exit 1 ; fi

$(tarball_phigemm):
	wget -O $(tarball_phigemm) $(url_phigemm) > /dev/null 2>&1
	@if test "`echo $$?`" -ne "0" ; then \
		curl -o $(tarball_phigemm) $(url_phigemm) > /dev/null 2>&1 ; \
	fi
	@if test ! -s $(tarball_phigemm) ; then \
		echo "*** Unable to download PhiGEMM. Test whether curl or wget is installed and working," ; \
		echo "*** if you have direct access to the internet." ; \
		echo "*** If not, copy $(tarball_phigemm) into extlibs/archive/" ; \
	exit 1 ; fi

#
# cleaning
#
clean:

clean_all:
	if test -s $(tarball_blas)    && test "$(keep_blas)"    != "yes" ;  then rm $(tarball_blas)    ; fi
	if test -s $(tarball_lapack)  && test "$(keep_lapack)"  != "yes" ;  then rm $(tarball_lapack)  ; fi
	if test -s $(tarball_iotk)    && test "$(keep_iotk)"    != "yes" ;  then rm $(tarball_iotk)    ; fi
	if test -s $(tarball_netcdf)  && test "$(keep_netcdf)"  != "yes" ;  then rm $(tarball_netcdf)  ; fi
	if test -s $(tarball_etsf_io) && test "$(keep_etsf_io)" != "yes" ;  then rm $(tarball_etsf_io) ; fi
	if test -s $(tarball_phigemm) && test "$(keep_phigemm)" != "yes" ;  then rm $(tarball_phigemm) ; fi
	if test -s $(tarball_magma)   && test "$(keep_magma)"   != "yes" ;  then rm $(tarball_magma)   ; fi

