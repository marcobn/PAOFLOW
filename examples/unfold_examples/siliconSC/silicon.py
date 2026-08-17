from PAOFLOW import GPAO
from PAOFLOW.models.band_unfold import (
    FCC_PATH,
    FCC_SYM_POINTS,
    plot_unfolded,
    unfold_bands,
)
from PAOFLOW.models.edtb_params import EDTBModel

pplt = GPAO.GPAO()

Ry2eV = 13.60569193

_SK_PARAMS = "Si_SK_params.json"
_SK_GEOM = "Si_geometry.json"

print(f"Loading bulk EDTB model from {_SK_PARAMS} ...")
model = EDTBModel.from_files(_SK_PARAMS, _SK_GEOM)
print(model)


# ═══════════════════════════════════════════════════════════
# Band unfolding: 8-atom SC → 2-atom FCC Brillouin zone
# (uses the general-purpose band_unfold module)
# ═══════════════════════════════════════════════════════════

# Load PC and SC models
model_fcc_uf = EDTBModel.from_files("Si_SK_params.json", "Si_geometry.json")
model_sc_uf = EDTBModel.from_files("Si_SK_params.json", "Si_geometry_sc8.json")

md_fcc = model_fcc_uf.to_model_dict()
md_sc = model_sc_uf.to_model_dict()

# Run unfolding (fully automatic: finds M, translations, atom mapping)
result = unfold_bands(
    md_fcc,
    md_sc,
    sym_points=FCC_SYM_POINTS,
    path_str=FCC_PATH,
    nk_per_seg=80,
    verbose=True,
)

# Plot
plot_unfolded(result, y_lim=(-10, 4), title="Band unfolding: SC (8 atoms) → FCC BZ")
