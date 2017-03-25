#
#===============================
# WanT package
#===============================
#
include ../../make.sys
include ../archive/package.list

LIBNAME=libphiGEMM.a
PACKAGE=$(pkgname_phigemm)
TARBALL=$(tarball_phigemm)

#
# redefinitions
#
INSTALL_DIR=$(TOPDIR)/extlibs/phiGEMM/tmp
PARA_FLAGS=$(findstring __PARA,$(DFLAGS) )

#test: set_flags
#	@echo $(PARA_FLAGS)
#	@echo $(AUX_FLAGS)

#
# MAIN target
#
all: package-ready-stamp

#
# manage parallel configure
#
ifdef PARA_FLAGS
set_flags:
	AUX_FLAGS=--enable-parallel
else
set_flags:
	AUX_FLAGS=
endif


uncompress-stamp:
	( cd ../archive ; if ! test -e $(TARBALL) ; then make $(TARBALL) ; fi )
	gunzip < ../archive/$(TARBALL) | ../../config/missing --run tar xf -
	test -d $(INSTALL_DIR) || mkdir $(INSTALL_DIR)
	touch uncompress-stamp

configure-stamp: uncompress-stamp set_flags
	if test -d $(PACKAGE) ; then ( cd $(PACKAGE);  \
		if test -e ./make.inc ; then rm ./make.inc ; fi; \
		if test -x ./configure ; then ./configure  \
		--prefix=$(INSTALL_DIR) \
		$(AUX_FLAGS) \
	        FC="$(F90)" \
	        CC="mpicc" \
	        LD="mpicc" \
	        F77=$(F77) \
	        FFLAGS="$(FFLAGS)" \
	        FCFLAGS="$(F90FLAGS)" ; fi; )\
	fi
	touch configure-stamp

	        #CFLAGS="-O3 --compiler-options '-fPIC'" \
	        #CC="$(NCC)" \
	        #CFLAGS="$(NVCCFLAGS)" \
	
package-ready-stamp: uncompress-stamp configure-stamp
	if test -d $(PACKAGE) ; then \
	( cd $(PACKAGE);  $(MAKE) all ) ; fi
	#
	if test -d ./lib      ; then rm -rf ./lib      ; fi    ; mkdir ./lib ;
	if test -d ./include  ; then rm -rf ./include  ; fi    ; mkdir ./include ;
	#mv $(PACKAGE)/lib/lib*  ./lib ;
	cp $(PACKAGE)/lib/lib*  ./lib ;
	cp $(PACKAGE)/include/* ./include ;
	#if test -d ./bin      ; then rm -rf ./bin      ; fi    ; mkdir ./bin ;
	#mv $(PACKAGE)/src/*.x ./bin ;
	#
	touch package-ready-stamp

#
# cleaning
#
clean:
	#if test -d $(PACKAGE) ; then ( rm -rf $(PACKAGE) ) ; fi
	if test -d $(PACKAGE) ; then ( cd $(PACKAGE); \
		touch include/fake.mod; \
		touch testing/fake.o ; \
		$(MAKE) clean ) ; fi
	- rm -rf ./lib ./include package-ready-stamp configure-stamp 

clean_all: clean
	if test -d $(PACKAGE) ; then ( rm -rf $(PACKAGE) ) ; fi 
	- rm uncompress-stamp
	- rm -rf $(INSTALL_DIR)
	

