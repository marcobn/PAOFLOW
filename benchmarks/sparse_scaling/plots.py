"""Figures of the per-cell curves, shared by every system's ``observables.ipynb``.

``Benchmark(name)`` loads the finished cases of ``systems/<name>/runs``: the
dense reference and every size of every sparse pipeline.  ``show(names)``
draws one figure per pipeline with one panel per curve of :mod:`observables`;
each size is a coloured line and the reference a dashed black one.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np

import observables
from config import load_system, output_root

REFERENCE = 'reference_d00'

# palette: deep blue, vermillion, amber, then Okabe-Ito
PALETTE = ['#004488', '#D55E00', '#FFC20A', '#009E73', '#56B4E9', '#CC79A7', '#000000']
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3})

_MU = 'μ (eV)'
_E = 'E - E$_F$ (eV)'
_W = 'ħω (eV)'

# name: (title, x label, y label)
LABELS = {
    'dos': ('DoS per primitive cell', _E, 'states/eV'),
    'doping_mu': ('μ at the doping of the system', 'T (K)', 'μ (eV)'),
    'sigma_xx': ('σ$_{xx}$/τ', _MU, '1/(Ω m s)'),
    'sigma_zz': ('σ$_{zz}$/τ', _MU, '1/(Ω m s)'),
    'seebeck_xx': ('Seebeck S$_{xx}$', _MU, 'V/K'),
    'kappa_xx': ('κ$^{el}_{xx}$/τ', _MU, 'W/(m K s)'),
    'pf_xx': ('power factor/τ, xx', _MU, 'W/(m K$^2$ s)'),
    'hall_trace': ('Hall tensor trace', _MU, 'm$^3$/C'),
    'hall_xyz': ('Hall R$_{xy,z}$', _MU, 'm$^3$/C'),
    'hall_xzy': ('Hall R$_{xz,y}$', _MU, 'm$^3$/C'),
    'hall_yzx': ('Hall R$_{yz,x}$', _MU, 'm$^3$/C'),
    'nernst_xyz': ('Nernst N$_{xyz}$', _MU, 'V/(K T)'),
    'nernst_yxz': ('Nernst N$_{yxz}$', _MU, 'V/(K T)'),
    'sigma_xy': ('σ$_{xy}$/τ ÷ max |σ$_{xx}$|', _MU, ''),
    'sigma_xz': ('σ$_{xz}$/τ ÷ max |σ$_{xx}$|', _MU, ''),
    'sigma_yz': ('σ$_{yz}$/τ ÷ max |σ$_{xx}$|', _MU, ''),
    'seebeck_xy': ('S$_{xy}$ ÷ max |S$_{xx}$|', _MU, ''),
    'kappa_xy': ('κ$^{el}_{xy}$ ÷ max |κ$_{xx}$|', _MU, ''),
    'pf_xy': ('PF$_{xy}$ ÷ max |PF$_{xx}$|', _MU, ''),
    'hall_zero': ('RMS of the 6 vanishing R$_{ij,k}$ ÷ max |R$_{xy,z}$|', _MU, ''),
    'nernst_zero': ('RMS of the 21 vanishing N$_{ijk}$ ÷ max |N$_{xyz}$|', _MU, ''),
    'lr_cond_xx': ('Kubo σ$_{xx}$(μ)', _MU, 'S/cm'),
    'lr_cond_xy': ('Kubo σ$_{xy}$(μ)', _MU, 'S/cm'),
    'lr_cond_zz': ('Kubo σ$_{zz}$(μ)', _MU, 'S/cm'),
    'lr_ahc_xy': ('anomalous Hall σ$_{xy}$, Kubo', _MU, 'S/cm'),
    'lr_ahc_yz': ('anomalous Hall σ$_{yz}$, Kubo', _MU, 'S/cm'),
    'berry_ahc_xy': ('anomalous Hall σ$_{xy}$, Berry curvature', _MU, 'S/cm'),
    'berry_ahc_yz': ('anomalous Hall σ$_{yz}$, Berry curvature', _MU, 'S/cm'),
    'mcd_re_xy': ('magnetic circular dichroism, Re σ$_{xy}$(ω)', _W, ''),
    'mcd_im_xy': ('magnetic circular dichroism, Im σ$_{xy}$(ω)', _W, ''),
    'berry_shc_z_xy': ('spin Hall σ$^z_{xy}$, Berry curvature', _MU, '(ħ/e) S/cm'),
    'berry_shc_x_yz': ('spin Hall σ$^x_{yz}$, Berry curvature', _MU, '(ħ/e) S/cm'),
    'berry_shc_z_xz': ('spin Hall σ$^z_{xz}$, Berry curvature', _MU, '(ħ/e) S/cm'),
    'scd_re_z_xy': ('spin circular dichroism, Re σ$^z_{xy}$(ω)', _W, ''),
    'scd_im_z_xy': ('spin circular dichroism, Im σ$^z_{xy}$(ω)', _W, ''),
    'berry_ohc_z_xy': ('orbital Hall σ$^z_{xy}$', _MU, '(ħ/e) S/cm'),
    'berry_ohc_x_yz': ('orbital Hall σ$^x_{yz}$', _MU, '(ħ/e) S/cm'),
    'berry_ohc_z_xz': ('orbital Hall σ$^z_{xz}$', _MU, '(ħ/e) S/cm'),
    'lr_shc_even_z_xy': ('spin Hall σ$^z_{xy}$, Kubo', _MU, '(ħ/e) S/cm'),
    'lr_shc_even_x_yz': ('spin Hall σ$^x_{yz}$, Kubo', _MU, '(ħ/e) S/cm'),
    'lr_shc_even_z_xz': ('spin Hall σ$^z_{xz}$, Kubo', _MU, '(ħ/e) S/cm'),
    'ree_spin_xx': ('Rashba-Edelstein, spin, xx', _E, ''),
    'ree_spin_zz': ('Rashba-Edelstein, spin, zz', _E, ''),
    'ree_spin_xy': ('Rashba-Edelstein, spin, xy', _E, ''),
    'ree_orbital_xx': ('Rashba-Edelstein, orbital, xx', _E, ''),
    'ree_orbital_zz': ('Rashba-Edelstein, orbital, zz', _E, ''),
    'ree_orbital_xy': ('Rashba-Edelstein, orbital, xy', _E, ''),
    'lr_ree_even_xx': ('Rashba-Edelstein χ$_{xx}$, Kubo', _MU, ''),
    'lr_ree_even_zz': ('Rashba-Edelstein χ$_{zz}$, Kubo', _MU, ''),
    'lr_ree_even_xy': ('Rashba-Edelstein χ$_{xy}$, Kubo', _MU, ''),
    'bands': (
        'band path, the bands one after another (d = 0 only)',
        'band × k-point',
        'E - E$_F$ (eV)',
    ),
    'spin_texture': ('RMS |⟨S⟩| over the mesh, per band (d = 0 only)', 'band in the window', ''),
    'cond_xx': ('σ$_{xx}$(E), band-diagonal', _E, ''),
    'cond_xy': ('σ$_{xy}$(E), band-diagonal', _E, ''),
    'cond_zz': ('σ$_{zz}$(E), band-diagonal', _E, ''),
    'epsi_xx': ('ε″$_{xx}$', _W, ''),
    'epsi_zz': ('ε″$_{zz}$', _W, ''),
    'epsr_xx': ('ε′$_{xx}$', _W, ''),
    'eels_xx': ('EELS, -Im 1/ε$_{xx}$', _W, ''),
    'refl_xx': ('reflectivity R$_{xx}$', _W, ''),
    'nref_xx': ('refractive index n$_{xx}$', _W, ''),
    'kref_xx': ('extinction κ$_{xx}$', _W, ''),
    'alpha_xx': ('absorption α$_{xx}$', _W, '1/m'),
    'sigmar_xx': ('Re σ$_{xx}$(ω)', _W, ''),
    'sigmai_xx': ('Im σ$_{xx}$(ω)', _W, ''),
    'inv_mass_dos': ('Σ g(E - E$_{nk}$) / m$^{dos}_{nk}$', _E, '1/(m$_e$ eV)'),
    'inv_mass_trace': ('Σ g(E - E$_{nk}$) tr m$^{-1}_{nk}$ / 3', _E, '1/(m$_e$ eV)'),
}

# curves that span many decades (the Hall and Nernst tensors diverge in a
# gap, where the carrier density vanishes): symlog y axis with this linear
# threshold.  A metal has no gap; the axis is then linear (see Benchmark)
SYMLOG = {
    'hall_trace': 1e-11,
    'hall_xyz': 1e-11,
    'hall_xzy': 1e-11,
    'hall_yzx': 1e-11,
    'nernst_xyz': 1e5,
    'nernst_yxz': 1e5,
}


def symlog(ax, linthresh):
    # symlog y axis with at most about five decades labelled on each side
    ax.set_yscale('symlog', linthresh=linthresh)
    top = np.log10(max(abs(y) for y in ax.get_ylim()))
    exps = np.arange(np.log10(linthresh) + 1, np.floor(top) + 1)
    exps = exps[::-1][:: max(1, -(-len(exps) // 5))]
    ax.set_yticks([0.0] + [s * 10.0**e for e in exps for s in (-1, 1)])


def style(d):
    return dict(color=PALETTE[d % len(PALETTE)], ls=('-', '--', ':')[d // len(PALETTE)], lw=1.1)


class Benchmark:
    """The finished cases of one system, and its figures.

    Parameters
    ----------
    name : str
        The system's folder name (``'si'``, ``'pt'``, ...).
    root : str, optional
        Its runs directory; ``BENCH_OUTPUT_ROOT`` or ``systems/<name>/runs``
        by default.
    symlog : bool
        Draw the Hall and Nernst tensors on a symlog axis (for a semiconductor).
    """

    def __init__(self, name, root=None, symlog=True):
        self.name = name
        self.system = load_system(name)
        self.root = root or os.environ.get('BENCH_OUTPUT_ROOT', output_root(name))
        self.symlog = SYMLOG if symlog else {}
        self.pipelines = [
            p
            for p, pipe in self.system['pipelines'].items()
            if pipe.get('enabled', True) and pipe['sparse']
        ]
        ref_case = self._load(os.path.join(self.root, REFERENCE))
        self.ref = ref_case['curves'] if ref_case else {}
        if not self.ref:
            print('no finished %s yet: the figures show the sparse sizes only' % REFERENCE)
        self.runs = {}
        for pipeline in self.pipelines:
            paths = sorted(glob.glob(os.path.join(self.root, pipeline + '_d[0-9][0-9]')))
            cases = [c for c in map(self._load, paths) if c is not None]
            if cases:
                self.runs[pipeline] = cases
                print('%-20s d = %s' % (pipeline, ', '.join(str(c['d']) for c in cases)))

    @staticmethod
    def _load(path):
        if not os.path.exists(os.path.join(path, 'record.json')):
            return None
        rec = observables.load_record(path)
        if rec.get('status') != 'ok':
            print('skipping %s: status %s' % (os.path.basename(path), rec.get('status')))
            return None
        return {
            'd': rec['case']['d'],
            'nawf': rec['hamiltonian']['nawf'],
            'curves': observables.load(path, pdos=False, optics=True),
        }

    def describe(self, pipeline):
        sc = self.system['pipelines'][pipeline].get('sparse_config', {})
        parts = ['bond_order %d' % sc['bond_order']] if 'bond_order' in sc else []
        if sc.get('hopping_threshold') == 0:
            parts.append('every H(R) element')
        w = sc.get('interior_window')
        if w is None:
            parts.append('window from the property ranges')
        elif w:
            parts.append('interior window [%.1f, %.1f] eV' % (w['elo'], w['ehi']))
        else:
            parts.append('interior window from the property ranges')
        return '%s %s: %s' % (self.system['label'].split(' (')[0], pipeline, ', '.join(parts))

    def show(self, names, ncol=3, normalized=False):
        """One figure per pipeline, one panel per curve it has.

        ``normalized`` divides every curve by its ``scale`` (the norm of its
        deviation), which shows a component that must vanish relative to its
        symmetry-allowed partner.
        """
        for pipeline, cases in self.runs.items():
            present = [n for n in names if any(n in c['curves'] for c in cases)]
            if not present:
                continue
            nc = min(ncol, len(present))
            nr = -(-len(present) // nc)
            fig, axes = plt.subplots(nr, nc, figsize=(4.4 * nc, 3.2 * nr), squeeze=False)
            legend = {}
            for ax, name in zip(axes.flat, present):
                curves = [(c, c['curves'][name]) for c in cases if name in c['curves']]
                norm = lambda cv: cv['scale'] if normalized and cv['scale'] else 1.0
                for c, cv in curves:
                    ax.plot(
                        cv['x'],
                        cv['y'] / norm(cv),
                        label='d=%d (nawf %d)' % (c['d'], c['nawf']),
                        **style(c['d']),
                    )
                if name in self.ref:
                    r = self.ref[name]
                    ax.plot(r['x'], r['y'] / norm(r), 'k--', lw=1.0, label='dense reference')
                if name in self.symlog:
                    symlog(ax, self.symlog[name])
                ax.set_xlim(
                    min(cv['x'].min() for _, cv in curves), max(cv['x'].max() for _, cv in curves)
                )
                cuts = [cv['xmax'] for _, cv in curves if cv['xmax'] is not None]
                if cuts:
                    ax.axvline(min(cuts), color='#888888', lw=0.8)
                title, xlabel, ylabel = LABELS.get(name, (name, '', ''))
                if normalized and not title.count('÷'):
                    title += ' ÷ scale'
                ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
                legend.update(zip(*reversed(ax.get_legend_handles_labels())))
            for ax in axes.flat[len(present) :]:
                ax.axis('off')
            fig.suptitle(self.describe(pipeline))
            fig.tight_layout()
            fig.legend(
                legend.values(),
                legend.keys(),
                loc='center left',
                bbox_to_anchor=(1.0, 0.5),
                fontsize=8,
            )
            plt.show()

    def show_bands(self, emin=-3.0, emax=3.0):
        """The band path of the reference against d = 0 of every sparse pipeline
        that wrote one: the bands (left, sparse in colour, reference dashed) and
        sparse minus reference per band in meV (right).  The sparse file holds
        the well-projected bands only, and NaN above the usable top."""
        path = os.path.join(self.root, REFERENCE, 'output', 'bands_0.dat')
        if not os.path.exists(path):
            print('no reference band path yet')
            return
        ref = np.loadtxt(path)[:, 1:]
        for pipeline in self.pipelines:
            path = os.path.join(self.root, pipeline + '_d00', 'output', 'bands_0.dat')
            if not os.path.exists(path):
                continue
            E = np.loadtxt(path)[:, 1:]
            nb = E.shape[1]
            diff = 1e3 * (E - ref[:, :nb])
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
            k = np.arange(len(ref))
            axes[0].plot(k, ref, 'k--', lw=0.8)
            for b in range(nb):
                axes[0].plot(k, E[:, b], color=PALETTE[0], lw=1.0)
            axes[0].set(ylim=(emin, emax), xlabel='k-point on the path', ylabel='E - E$_F$ (eV)')
            axes[0].set_title('bands: sparse d = 0 (blue), dense reference (dashed)')
            near = np.abs(ref[:, :nb]) <= max(abs(emin), abs(emax))
            for b in range(nb):
                if near[:, b].any():
                    axes[1].plot(k, diff[:, b], lw=0.9, color=PALETTE[(b % (len(PALETTE) - 1)) + 1])
            axes[1].set(xlabel='k-point on the path', ylabel='sparse - dense (meV)')
            ok = np.isfinite(diff) & near
            axes[1].set_title(
                'bands within the window: RMS %.1f meV, max %.1f meV'
                % (np.sqrt(np.mean(diff[ok] ** 2)), np.abs(diff[ok]).max())
            )
            fig.suptitle(self.describe(pipeline))
            fig.tight_layout()
            plt.show()
