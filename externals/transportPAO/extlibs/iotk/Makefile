#
#===============================
# WanT package
#===============================
#
include ../../make.sys
include ../archive/package.list

LIBNAME=libiotk.a
PACKAGE=$(pkgname_iotk)
TARBALL=$(tarball_iotk)

#
# redefinitions
#
INSTALL_DIR=$(TOPDIR)/extlibs/iotk/tmp
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
		if test -e ../make.sys ; then rm ../make.sys ; fi ; \
	        if test -e ../../install/make_iotk.inc ; then \
	           cp ../../install/make_iotk.inc ../make.sys ; \
	        fi ; \
	        if test -e ../../install/iotk_specials.h ; then \
	           cp ../../install/iotk_specials.h ../iotk/include ; \
	        fi ) ; \
	fi
	if test -d $(PACKAGE) ; then ( cd $(PACKAGE);  \
	        if test -x ./configure ; then ./configure  \
	           FC=$(F90) \
	           IOTK_SAFEST="yes" \
	           IOTK_MAX_RANK=4; \
	        fi ) ; \
	fi
	touch configure-stamp
	           #IOTK_REAL_KINDS="ALL" \
	
package-ready-stamp: uncompress-stamp configure-stamp 
	if test -d $(PACKAGE) ; then \
	( cd $(PACKAGE);  $(MAKE) loclib_only ; $(MAKE) iotk.x ) ; fi
	#
	if test -d ./lib      ; then rm -rf ./lib      ; fi    ; mkdir ./lib ;
	if test -d ./bin      ; then rm -rf ./bin      ; fi    ; mkdir ./bin ;
	if test -d ./include  ; then rm -rf ./include  ; fi    ; mkdir ./include ;
	cp $(PACKAGE)/src/*.a ./lib ;
	cp $(PACKAGE)/src/*.x ./bin ;
	cp $(PACKAGE)/tools/iotk ./bin ;
	cp $(PACKAGE)/src/*.mod ./include ;
	#
	- ( cd ../../bin ; ln -sf ../extlibs/iotk/bin/* . )
	#
	touch package-ready-stamp

#
# cleaning
#
clean:
	if test -d $(PACKAGE) ; then ( cd $(PACKAGE);  $(MAKE) clean ) ; fi
	- rm -rf ./lib ./bin ./include  package-ready-stamp configure-stamp

clean_all: clean
	if test -d $(PACKAGE) ; then ( rm -rf $(PACKAGE) ) ; fi 
	- rm make.sys uncompress-stamp package-ready-stamp
	- rm -rf $(INSTALL_DIR)
	

