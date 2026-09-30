"""
Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO).
"""

__version__ = '3.0.0'

from PAOFLOW.utils.parallel_resources import configure_joblib_for_mpi as _configure_joblib_for_mpi

# Cap joblib/loky pools at this rank's share of the node under mpirun.
_configure_joblib_for_mpi()
