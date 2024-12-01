#
#===============================
# WanT package
#===============================
#
include ../../make.sys
include ../archive/package.list

LIBNAME=libblas.a
PACKAGE=$(pkgname_blas)
TARBALL=$(tarball_blas)

#
# redefinitions
#
INSTALL_DIR=$(TOPDIR)/extlibs/blas/tmp
#AUXFLAGS= --prefix=$(INSTALL_DIR)

#
# MAIN target
#
all: package-ready-stamp

uncompress-stamp:
	( cd ../archive ; if ! test -e $(TARBALL) ; then make $(TARBALL) ; fi )
	gunzip < ../archive/$(TARBALL) | ../../config/missing --run tar xf -
	test -d $(INSTALL_DIR) || mkdir $(INSTALL_DIR)
	touch uncompress-stamp

configure-stamp: uncompress-stamp
	if test -d $(PACKAGE) ; then ( cd $(PACKAGE);  \
		if test -e ./make.inc ; then rm ./make.inc ; fi ; \
	        if test -e ../../install/make_blas.inc ; then \
	           cp ../../install/make_blas.inc ./make.inc ; \
	        fi ) ; \
	fi
	touch configure-stamp
#	if test ./install ; then ( cd ./install ;  \
#	        ./configure_blas $(AUXFLAGS)  \
#	        FC=$(F90) \
#	        F77=$(F77) \
#	        FFLAGS="$(FFLAGS)" \
#	        FCFLAGS="$(F90FLAGS)" \
#	        INSTALL_DIR="$(INSTALL_DIR)" ); \
#	fi
	
package-ready-stamp: uncompress-stamp configure-stamp 
	if test -d $(PACKAGE) ; then \
	( cd $(PACKAGE);  $(MAKE) ) ; fi
	#
	if test -d ./lib      ; then rm -rf ./lib      ; fi
	mkdir ./lib ;
	mv $(PACKAGE)/*.a ./lib ;
	#
	#rm configure-stamp
	touch package-ready-stamp

#
# cleaning
#
clean:
	if test -d $(PACKAGE) ; then ( cd $(PACKAGE);  $(MAKE) clean ) ; fi
	- rm -rf ./lib package-ready-stamp configure-stamp

clean_all: clean
	if test -d $(PACKAGE) ; then ( rm -rf $(PACKAGE) ) ; fi 
	- rm uncompress-stamp package-ready-stamp
	- rm -rf $(INSTALL_DIR)
	

