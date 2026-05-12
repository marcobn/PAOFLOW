#
# PAOFLOW
#
# Copyright 2016-2024 - Marco BUONGIORNO NARDELLI (mbn@unt.edu)
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .

import numpy as np


def read_poscar(fname, verbose=False):
    """
    Read a VASP POSCAR file and extract structural information.

    Arguments:
        fname (str): Path to the POSCAR file
        verbose (bool): Print debug information

    Returns:
        dict: Dictionary containing:
            - 'comment': First line comment
            - 'scaling_factor': Lattice scaling factor
            - 'lattice_vectors': 3x3 array of lattice vectors (in Angstrom)
            - 'species': List of atomic species
            - 'num_atoms': List of number of atoms for each species
            - 'positions': Nx3 array of atomic positions
            - 'dynamics': Nx3 array of selective dynamics (True/False for each coordinate)
            - 'is_selective_dynamics': Boolean indicating if selective dynamics are present
            - 'is_cartesian': Boolean indicating if coordinates are Cartesian (True) or Direct (False)
    """

    with open(fname, 'r') as f:
        lines = f.readlines()

    if len(lines) < 7:
        raise ValueError('POSCAR file appears to be truncated or invalid')

    # Line 1: Comment
    comment = lines[0].strip()

    # Line 2: Scaling factor
    scaling_factor = float(lines[1].split()[0])

    # Lines 3-5: Lattice vectors
    lattice_vectors = np.zeros((3, 3))
    for i in range(3):
        lattice_vectors[i, :] = np.array([float(x) for x in lines[2 + i].split()[:3]])
    lattice_vectors *= scaling_factor

    # Line 6: Species (or number of atoms if species not listed)
    line6 = lines[5].split()
    species = []
    num_atoms_per_species = []

    # Check if line 6 contains species names or numbers
    try:
        # Try to parse as integers (old POSCAR format without species names)
        num_atoms_per_species = [int(x) for x in line6]
        species = ['Atom' + str(i) for i in range(len(num_atoms_per_species))]
        species_line_idx = 5
    except ValueError:
        # Line 6 contains species names
        species = line6
        num_atoms_per_species = [int(x) for x in lines[6].split()]
        species_line_idx = 6

    total_atoms = sum(num_atoms_per_species)

    # Check for Selective Dynamics
    next_line_idx = species_line_idx + 1
    is_selective_dynamics = False
    selective_dyn_line = lines[next_line_idx].strip()[0].upper()

    if selective_dyn_line == 'S':
        is_selective_dynamics = True
        next_line_idx += 1

    # Check for Cartesian or Direct coordinates
    coord_line = lines[next_line_idx].strip()[0].upper()
    is_cartesian = coord_line == 'C' or coord_line == 'K'

    if verbose:
        print('Comment:', comment)
        print('Scaling factor:', scaling_factor)
        print('Species:', species)
        print('Num atoms per species:', num_atoms_per_species)
        print('Total atoms:', total_atoms)
        print('Is selective dynamics:', is_selective_dynamics)
        print('Is Cartesian:', is_cartesian)

    # Read atomic positions
    positions = np.zeros((total_atoms, 3))
    dynamics = np.ones((total_atoms, 3), dtype=bool)  # True = movable (default)

    atom_idx = 0
    for i in range(total_atoms):
        line_idx = next_line_idx + 1 + i
        parts = lines[line_idx].split()

        # Get coordinates
        positions[atom_idx, :] = np.array([float(x) for x in parts[:3]])

        # Get selective dynamics if present
        if is_selective_dynamics:
            dyn_parts = parts[3:6]
            for j in range(3):
                dynamics[atom_idx, j] = dyn_parts[j][0].upper() == 'T'

        atom_idx += 1

    # Convert to Cartesian if needed
    if not is_cartesian:
        positions = np.dot(positions, lattice_vectors)

    result = {
        'comment': comment,
        'scaling_factor': scaling_factor,
        'lattice_vectors': lattice_vectors,
        'species': species,
        'num_atoms': num_atoms_per_species,
        'positions': positions,
        'dynamics': dynamics,
        'is_selective_dynamics': is_selective_dynamics,
        'is_cartesian': is_cartesian,
    }

    if verbose:
        print('Lattice vectors shape:', lattice_vectors.shape)
        print('Positions shape:', positions.shape)
        print('Dynamics shape:', dynamics.shape)

    return result


def poscar_to_data_controller(data_controller, fname, verbose=False):
    """
    Read a POSCAR file and populate the DataController.

    Arguments:
        data_controller (DataController): DataController object to populate
        fname (str): Path to POSCAR file
        verbose (bool): Print debug information
    """
    import numpy as np

    poscar_data = read_poscar(fname, verbose=verbose)

    arry, attr = data_controller.data_dicts()

    # Set lattice vectors and other structural info
    arry['a_vectors'] = poscar_data['lattice_vectors']
    attr['alat'] = 1.0  # Already in Angstrom

    # Build atomic species and positions
    atoms = []
    atom_positions = []
    for spec, count in zip(poscar_data['species'], poscar_data['num_atoms']):
        for _ in range(count):
            atoms.append(spec)

    arry['atoms'] = atoms
    arry['tau'] = poscar_data['positions']

    # Store additional info
    arry['selective_dynamics'] = poscar_data['dynamics']
    attr['is_selective_dynamics'] = poscar_data['is_selective_dynamics']

    if verbose:
        print('Populated DataController with POSCAR data')
        print('  Atoms:', arry['atoms'])
        print('  Positions shape:', arry['tau'].shape)
        print('  Lattice vectors shape:', arry['a_vectors'].shape)
