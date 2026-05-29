"""Plot QE bands.x output for GaAs: overlay bare DFT and eACBN0 (U+V).

Reads
  * ``GaAs.bands.dat.gnu``      — from run_bands.py       (U+V)
  * ``GaAs_bare.bands.dat.gnu`` — from run_bands_bare.py  (bare, optional)

Both are aligned independently to their own VBM = 0 eV.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent

# Path segments used in GaAs.bands.in (must match the K_POINTS card).
LABELS = ['L', r'$\Gamma$', 'X', 'W', 'K', r'$\Gamma$']
SEG_NK = [40, 40, 20, 20, 40, 1]   # weights from K_POINTS tpiba_b
NOCC = 9                            # 18 valence electrons / 2


def _read_gnu(path: Path):
    """Return (k, E_unsorted, E_sorted).  E shape = (Nk, Nbands)."""
    bands, current = [], []
    for line in path.read_text().strip().split('\n'):
        s = line.strip()
        if not s:
            if current:
                bands.append(np.array(current, dtype=float))
                current = []
        else:
            current.append(s.split())
    if current:
        bands.append(np.array(current, dtype=float))
    k = bands[0][:, 0]
    E = np.column_stack([b[:, 1] for b in bands])
    return k, E, np.sort(E, axis=1)


def _gap_summary(E_sorted, tick_idx):
    vbm = E_sorted[:, NOCC - 1].max()
    cbm = E_sorted[:, NOCC].min()
    def cbm_at(label):
        return E_sorted[tick_idx[LABELS.index(label)], NOCC]
    return {
        'vbm': vbm,
        'cbm': cbm,
        'fundamental': cbm - vbm,
        'direct_G': cbm_at(r'$\Gamma$') - vbm,
        'L': cbm_at('L') - vbm,
        'X': cbm_at('X') - vbm,
    }


# --- datasets ---------------------------------------------------------------
datasets = []
for path, label, color in [
    (HERE / 'GaAs_bare.bands.dat.gnu', 'bare PBE',  'tab:gray'),
    (HERE / 'GaAs.bands.dat.gnu',      'eACBN0 U+V', 'tab:blue'),
]:
    if path.exists():
        k, E, E_s = _read_gnu(path)
        datasets.append((label, color, k, E, E_s))
    else:
        print(f'[skip] {path.name} not found')

if not datasets:
    raise SystemExit('No bands files found.')

# Use the first dataset's k-path for tick positions (paths should match).
k_ref = datasets[0][2]
tick_idx = np.clip(np.cumsum([0] + SEG_NK)[:-1], 0, len(k_ref) - 1)
tick_pos = k_ref[tick_idx]

# Print gap summary for each dataset.
for label, _, k, _, E_s in datasets:
    g = _gap_summary(E_s, tick_idx)
    print(f'\n[{label}]')
    print(f'  VBM = {g["vbm"]:.4f} eV   CBM = {g["cbm"]:.4f} eV')
    print(f'  fundamental gap  = {g["fundamental"]:.4f} eV')
    print(f'  Γ-Γ direct gap   = {g["direct_G"]:.4f} eV')
    print(f'  Γ→L gap          = {g["L"]:.4f} eV')
    print(f'  Γ→X gap          = {g["X"]:.4f} eV')

# --- plot -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
for label, color, k, E, E_s in datasets:
    vbm = E_s[:, NOCC - 1].max()
    Es = E - vbm
    for i in range(Es.shape[1]):
        ax.plot(k, Es[:, i], lw=1.0, color=color,
                label=label if i == 0 else None)

ax.axhline(0.0, color='k', lw=0.5, ls='--')
for x in tick_pos[1:-1]:
    ax.axvline(x, color='k', lw=0.4, alpha=0.5)

ax.set_xticks(tick_pos)
ax.set_xticklabels(LABELS)
ax.set_xlim(k_ref.min(), k_ref.max())
ax.set_ylim(-8, 5)
ax.set_ylabel('E - E$_{VBM}$ (eV)')

# Title: show fundamental + Γ-Γ for the U+V (primary) dataset.
primary = next((d for d in datasets if d[0] != 'bare PBE'), datasets[0])
_, _, _, _, E_s = primary
g = _gap_summary(E_s, tick_idx)
ax.set_title(
    f'GaAs bands: bare PBE vs eACBN0 (U+V)   '
    f'$\\Gamma$-$\\Gamma$ = {g["direct_G"]:.3f} eV,  '
    f'$\\Gamma$$\\to$L = {g["L"]:.3f} eV,  '
    f'$\\Gamma$$\\to$X = {g["X"]:.3f} eV'
)
ax.legend(loc='upper right', frameon=False)
fig.tight_layout()

out = HERE / 'bands_compare_qe.png'
fig.savefig(out, dpi=150)
print(f'\nwrote {out}')
plt.show()
