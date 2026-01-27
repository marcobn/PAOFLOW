#
#===============================
# WanT package
#===============================
#
include ../../make.sys
include ../archive/package.list

LIBNAME=libmagma.a
PACKAGE=$(pkgname_magma)
TARBALL=$(tarball_magma)

#
# redefinitions
#
INSTALL_DIR=$(TOPDIR)/extlibs/magma/tmp


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
		if test -e ./make.inc ; then rm ./make.inc ; fi; \
		if test -e ../../install/make_magma.inc ; then \
			cp ../../install/make_magma.inc ./make.inc ; fi; ) \
	 	fi
	touch configure-stamp

	        #CFLAGS="-O3 --compiler-options '-fPIC'" \
	        #CC="$(NCC)" \
	        #CFLAGS="$(NVCCFLAGS)" \
	
package-ready-stamp: uncompress-stamp configure-stamp
	if test -d $(PACKAGE) ; then \
	( cd $(PACKAGE);  $(MAKE) lib ) ; fi
	#
	if test -d ./lib      ; then rm -rf ./lib      ; fi    ; mkdir ./lib ;
	if test -d ./include  ; then rm -rf ./include  ; fi    ; mkdir ./include ;
	#mv $(PACKAGE)/lib/lib*  ./lib ;
	cp $(PACKAGE)/lib*a   ./lib ;
	cp $(PACKAGE)/include/*.h ./include ;
	cp $(PACKAGE)/include/*.mod ./include ;
	#if test -d ./bin      ; then rm -rf ./bin      ; fi    ; mkdir ./bin ;
	#mv $(PACKAGE)/src/*.x ./bin ;
	#
	#rm configure-stamp
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
	

