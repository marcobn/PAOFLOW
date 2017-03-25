#
#===============================
# WanT package
#===============================
#
include ../../make.sys

OBJS= \
c_mkdir.o \
cuda_env.o \
cptimer.o \
fft_stick.o \
gcube2plt.o



#
# main targets
#
all : libclib.a


libclib.a : $(OBJS) 
	$(AR) $(ARFLAGS) $@ $?
	$(RANLIB) $@

source :
	co -l $(OBJS:.o=.c)

#
# cleaning
# 
clean: 
	- /bin/rm -f *.o *.a *core *.F90 *.mod work.pc work.pcl

