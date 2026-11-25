from pathlib import Path

from PAOFLOW import PAOFLOW
from PAOFLOW.basis_gen import generate_basis_for_pseudo

savedir = 'silicon.save'


def hamiltonian(route, **projection):
    """Projections, projectability, PAO Hamiltonian and bands of one route, in output_<route>/."""
    paoflow = PAOFLOW.PAOFLOW(savedir=savedir, outputdir=f'output_{route}', verbose=True)
    if route == 'projwfc':
        paoflow.read_atomic_proj_QE()
    else:
        paoflow.projections(**projection)
    paoflow.projectability(pthr=0.95)
    paoflow.pao_hamiltonian()
    paoflow.bands(ibrav=2, nk=500, fname='bands')
    paoflow.finish_execution()


# The projections of projwfc.x
hamiltonian('projwfc')

# The pseudo-orbitals of the pseudopotential
hamiltonian('minimal', configuration='minimal')

# Radial tables of the larger bases
if not Path('BASIS_PS/Si').is_dir():
    generate_basis_for_pseudo(f'{savedir}/Si.pbe-tm-new-gipaw-v2.1.UPF', 'BASIS_PS', preset='extended')

# The standard and extended presets
hamiltonian('standard', configuration='standard', basispath='BASIS_PS/')
hamiltonian('extended', configuration='extended', basispath='BASIS_PS/')

# Shells generated at run time
hamiltonian('generated', configuration='standard')

# Shells chosen by hand
hamiltonian('custom', configuration={'Si': ['3S', '3P', '3D', '4S']}, basispath='BASIS_PS/')
