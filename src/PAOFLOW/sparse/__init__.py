"""Sparse Hamiltonian utilities used by PAOFLOW.

The modules in this package keep Hamiltonian-related quantities in sparse form
for as long as possible so large calculations do not fail because of dense
memory growth. They provide sparse builders, sparse Fourier transforms, and
streamed post-processing helpers that match the existing PAOFLOW workflow.
"""
