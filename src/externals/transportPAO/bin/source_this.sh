module purge
module load Intel/15.0
module load IMPI/5.0.1
source /opt/software/ClusterStudio/2015.0/impi_latest/bin64/mpivars.sh
source /opt/software/ClusterStudio/2015.0/bin/compilervars.sh intel64
source /opt/software/ClusterStudio/2015.0/mkl/bin/intel64/mklvars_intel64.sh
OMP_NUM_THREADS=1
