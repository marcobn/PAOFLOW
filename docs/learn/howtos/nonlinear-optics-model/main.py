from PAOFLOW import PAOFLOW

model = {'label': 'graphene', 't': 2.7, 'delta': 1.0}  # sublattice potential: no inversion
paoflow = PAOFLOW.PAOFLOW(model=model, outputdir='nlo')
paoflow.nonlinear_optics(
    nk=(300, 300, 1),
    emin=0.0,
    emax=6.0,
    ne=601,
    eta=0.05,
    components=['yyy', 'yxx', 'xxy', 'xxx'],
    nelec=2,  # a model has no electron count of its own: fill the lower band
    twoD=True,
)
paoflow.finish_execution()

# The same model with inversion symmetry
centrosymmetric = PAOFLOW.PAOFLOW(
    model={'label': 'graphene', 't': 2.7, 'delta': 0.0}, outputdir='nlo_inv'
)
centrosymmetric.nonlinear_optics(
    nk=(120, 120, 1), emin=0.0, emax=6.0, ne=601, eta=0.05, components=['yyy'], nelec=2, twoD=True
)
centrosymmetric.finish_execution()

# Symmetry checks
import numpy as np


def spectrum(name: str, path: str = 'nlo') -> tuple[np.ndarray, np.ndarray]:
    """Photon energy (eV) and value of one output file."""
    return np.loadtxt(f'{path}/{name}.dat', unpack=True)


energy, yyy = spectrum('shift_yyy')
yxx, xxy, xxx = (spectrum(f'shift_{c}')[1] for c in ('yxx', 'xxy', 'xxx'))
symmetry = max(np.abs(yyy + yxx).max(), np.abs(yyy + xxy).max())
_, chi = spectrum('shga_yyy')
_, chi_re = spectrum('shgr_yyy')
gap = (energy > 0.8) & (energy < 1.5)  # the one-photon resonance at the 1 eV gap
e_gap, re_gap = energy[gap][chi_re[gap].argmin()], chi_re[gap].min()
print(f'shift current: peak {yyy.max():.2f} nm muA/V^2 at {energy[yyy.argmax()]:.2f} eV')
print(f'  |yyy + yxx|, |yyy + xxy| <= {symmetry:.1e}, |xxx| <= {np.abs(xxx).max():.1e}')
print(f'|chi_yyy|: {chi[0]:.3e} pm^2/V at 0, peak {chi.max():.3e} at {energy[chi.argmax()]:.2f} eV')
print(f'Re chi_yyy: minimum {re_gap:.3e} pm^2/V at {e_gap:.2f} eV')
for kind in ('shiftm', 'injectm', 'injectc'):  # odd under time reversal; circular injection
    maxima = [
        f'{c} {np.abs(spectrum(f"{kind}_{c}")[1]).max():.1e}' for c in ('yyy', 'yxx', 'xxy', 'xxx')
    ]
    print(f'max |{kind}|: ' + ', '.join(maxima))
inv_shift, inv_chi = spectrum('shift_yyy', 'nlo_inv')[1], spectrum('shga_yyy', 'nlo_inv')[1]
print(f'delta = 0: max |shift| {np.abs(inv_shift).max():.1e}, max |chi| {inv_chi.max():.1e}')
