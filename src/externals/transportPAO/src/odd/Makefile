#
#===============================
# WanT package
#===============================
#
include ../../make.sys

OBJS= \
fft_desc_init.o 


#
# main targets
#
all : libfftqe.a


libfftqe.a : $(OBJS) 
	$(AR) $(ARFLAGS) $@ $?
	$(RANLIB) $@

#
# cleaning
# 
clean: 
	- /bin/rm -f *.o *.a *core *.F90 *.mod work.pc work.pcl
	        

# dependencies
include make.depend

