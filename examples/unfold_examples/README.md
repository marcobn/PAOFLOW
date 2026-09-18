Examples for the band unfolding algorithm:

1. siliconSC: unfolding of the 8 atom conventional cubic cell into the primitive FCC -- run with "python silicon.py"

2. TBG: twisted bilayer graphene -- includes the function to generate commensurate moire' cells -- can be run in parallel as "mpirun -np N python TBG.py n m" where n and m are the TBG supercell dimensions

3. TBG: twisted bilayer graphene -- sparse EDTB Hamiltonian construction and Lanczos diagonalization in SciPy eigsh -- parallelization with joblib and loky -- run with "python TBG_sparse.py"