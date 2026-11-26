import json
from pathlib import Path

import numpy as np

from PAOFLOW.models.band_unfold import unfold_bands
from PAOFLOW.models.edtb_params import EDTBModel

si_params = json.loads(Path('Si_SK_params.json').read_text())  # energies eV, distances bohr

# The primitive cell and the conventional cube of silicon
alat = 10.599478  # bohr
fcc = {
    'alat': alat,
    'a_vectors': [[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.5, 0.0]],
    'atoms': [
        {'species': 'Si', 'tau': [0.0, 0.0, 0.0]},
        {'species': 'Si', 'tau': [0.25, 0.25, 0.25]},
    ],
}
fcc_sites = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
cube = {
    'alat': alat,
    'a_vectors': np.eye(3).tolist(),
    'atoms': [
        {'species': 'Si', 'tau': list(np.add(s, b))}
        for b in ([0, 0, 0], [0.25, 0.25, 0.25])
        for s in fcc_sites
    ],
}

# Unfolding the perfect supercell
pc = EDTBModel(si_params, fcc).to_model_dict()
sc = EDTBModel(si_params, cube).to_model_dict()
points = {'G': [0, 0, 0], 'X': [0.5, 0.0, 0.5], 'L': [0.5, 0.5, 0.5]}
result = unfold_bands(pc, sc, sym_points=points, path_str='L-G-X', nk_per_seg=80)

# A substitutional impurity
impurity = EDTBModel(si_params, cube).to_model_dict()
site = impurity['model']['atoms']['0']
for orbital in site['orbitals']:
    site[orbital] += 1.0  # eV, on all nine orbitals of atom 0
result_imp = unfold_bands(pc, impurity, sym_points=points, path_str='L-G-X', nk_per_seg=80)

# Saving the bands and weights
np.savez(
    'si_unfolded.npz', kdist=result.kdist, kpath=result.kpath_cart,
    ticks=[t[0] for t in result.sym_ticks], E_pc=result.E_pc, E_sc=result.E_sc, W=result.W,
    E_imp=result_imp.E_sc, W_imp=result_imp.W,
)  # fmt: skip
