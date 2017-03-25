#
#===============================
# WanT package
#===============================
#
include ../../make.sys

OBJS= \
fft_errore.o \
fft_hpsort.o \
fft_stick.o \
fftw2.o \
mod_fft_aux.o \
mod_fft_base.o \
mod_fft_interfaces.o \
mod_fft_parallel.o \
mod_fft_scalar.o \
mod_fft_types.o \
mod_stick_base.o \
mod_stick_set.o 


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

