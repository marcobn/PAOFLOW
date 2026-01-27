#
#===============================
# WanT package
#===============================
#
include ../../make.sys


OBJS= \
atmproj_tools.o \
compute_kham.o \
compute_rham.o \
control.o \
t_control.o \
converters.o \
crystal_tools.o \
datafiles.o \
radial_grids.o \
pseudo_types.o \
cp2k_tools.o \
etsf_io_data.o \
get_monkpack.o \
get_rgrid.o \
io_module.o \
internal_tools.o \
monkpack.o \
operator.o \
startup.o \
smearing_base.o \
shutdown.o \
version.o \
wannier90_tools.o \
cleanup.o \
correlation.o \
egrid.o \
fourier_par.o \
green.o \
gzero_maker.o \
hamiltonian.o \
hamiltonian_init.o \
hamiltonian_setup.o \
input.o \
input_parameters.o \
kpoints.o \
memusage.o \
operator_blc.o \
read_matrix.o \
smearing_T.o \
summary.o \
transfer.o \
transmittance.o \
workspace.o \
write_data.o 

BIN= \
conductor.x \
current.x 

EXTERNAL_LIBS= ../clib/libclib.a ../baselib/libbase.a
BIN_DEPS=      $(OBJS) $(EXTERNAL_LIBS)

#
# main targets
#
all: $(BIN)

conductor.x: conductor.o $(BIN_DEPS)
	$(LD) $(LDFLAGS) -o $@ conductor.o \
	     $(OBJS) $(LIBOBJS) $(LIBS)
	- ( cd ../../bin ; ln -sf ../src/transport/$@ . )

current.x: current.o $(BIN_DEPS)
	$(LD) $(LDFLAGS) -o $@ current.o \
	     $(OBJS) $(LIBOBJS) $(LIBS)
	- ( cd ../../bin ; ln -sf ../src/transport/$@ . )

clean:
	rm -rf *.o *.a *core *.F90 work.pc work.pcl *.mod *.x

# dependencies
include make.depend

