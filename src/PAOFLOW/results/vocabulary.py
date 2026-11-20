"""PAOFLOW's controlled vocabulary: the meaning of each term of the results catalogue.

A term is the ``name`` of a :class:`~PAOFLOW.results.model.Kind`.  It identifies a file layout
(columns, units, energy reference, how spin and tensor components split across files); the links
identify the physical concept in a shared vocabulary (QUDT quantity kinds, PhySH concepts).  A
published term keeps its meaning: a changed definition gets a new term.  The run record
(``paoflow_outputs.json``) names the term of every file under ``kind``.

``python -m PAOFLOW.results.vocabulary`` prints the table of the documentation page
``docs/studio/run-records.md``.
"""

from __future__ import annotations

FERMI = 'Energies are relative to the Fermi level.'


def qudt(name: str) -> str:
    """Address of a QUDT quantity kind.

    Parameters
    ----------
    name : str
        Its name (``'SeebeckCoefficient'``).

    Returns
    -------
    str
    """
    return 'https://qudt.org/vocab/quantitykind/' + name


def physh(uuid: str) -> str:
    """Address of a PhySH concept.

    Parameters
    ----------
    uuid : str
        Its identifier.

    Returns
    -------
    str
    """
    return 'https://doi.org/10.29172/' + uuid


DOS = (qudt('EnergyDensityOfStates'), physh('aaff2519-9891-4d4e-a067-1ea36b6b08d4'))
FERMI_SURFACE = (physh('78d12677-c7dc-427e-acd7-cbff78a1713f'),)
BERRY = (physh('8abddf0b-ea8e-4c03-af96-883903427079'),)
ANOMALOUS_HALL = physh('8adc9a72-2460-4d3d-ad53-a44092c6a198')
SPIN_HALL = physh('bb4a3b78-a738-4e19-abd6-1feea6226ea1')
OPTICAL = physh('05ff9faf-4943-45b0-b627-49eb8a9404c6')
SHIFT_CURRENT = physh('9fffec69-8e96-42d0-b6bf-42d1234858eb')
ELPHON = (physh('2cb8570a-bc2f-41ab-a3f1-1467b288c3f0'),)
ELIASHBERG = (physh('c6f039f8-0580-4488-a6e0-b8bf5455a86d'),)
LANDAUER = physh('9c19d585-8eec-411b-af4c-0412fb91e670')
DHVA = (physh('2f031f28-5911-407e-8bce-149e70acf79b'),)
ARPES = (physh('a4074854-ac2f-4771-9451-0ce9bbad43ce'),)
DFT_U = (physh('ffcb7c9c-2fd2-4caa-ac0e-9832b20fc351'),)

#: term -> (definition, addresses of the shared terms it corresponds to).
TERMS: dict[str, tuple[str, tuple[str, ...]]] = {
    'manifest': ('The run record: what the runs of this folder did and wrote.', ()),
    'berry_bands': (
        'Berry curvature (or its spin or orbital counterpart) of one tensor component along the '
        'band path, one column per k-point index.',
        BERRY,
    ),
    'bands': (
        'Band energies along a k-path: the k-point index, then one column per band, one file '
        'per spin channel. ' + FERMI,
        (),
    ),
    'site_projected_bands': (
        'Band energies along a k-path with the weight of each band on the atomic sites. ' + FERMI,
        (),
    ),
    'velocity': (
        'Band velocity along one Cartesian direction on the k-path, one column per band.',
        (),
    ),
    'pdos': (
        'Density of states projected on one atomic orbital, per spin channel. ' + FERMI,
        DOS,
    ),
    'dos': (
        'Density of states of the PAO Hamiltonian, or the sum of its projections, per spin '
        'channel. ' + FERMI,
        DOS,
    ),
    'spin_texture': (
        'Expectation value of the spin (or orbital) operator on each band along a k-path.',
        (),
    ),
    'fermi_surface': (
        'Band energies on the full k-grid in XCrySDen BXSF format, all bands that cross the '
        'chosen window in one file.',
        FERMI_SURFACE,
    ),
    'fermi_sheet': (
        'Energies of a single band on the k-grid in BXSF format, the input of pyskeaf.',
        FERMI_SURFACE,
    ),
    'effective_mass': (
        'Effective-mass tensors of the band extrema, in units of the electron mass.',
        (qudt('EffectiveMass'),),
    ),
    'projectability': (
        'Projectability of each Kohn-Sham state on the atomic-orbital basis against its energy.',
        (),
    ),
    'boltzmann': (
        'A Boltzmann transport tensor in the constant relaxation-time form (conductivity, '
        'Seebeck coefficient, electronic thermal conductivity or power factor) against the '
        'chemical potential, for each temperature.',
        (
            qudt('ElectricConductivity'),
            qudt('SeebeckCoefficient'),
            qudt('ThermalConductivity'),
        ),
    ),
    'hall_coefficient': (
        'Hall coefficient from Boltzmann transport against the chemical potential, grouped by '
        'temperature.',
        (qudt('HallCoefficient'),),
    ),
    'nernst': (
        'Nernst coefficient per relaxation time from Boltzmann transport against the chemical '
        'potential, grouped by temperature.',
        (),
    ),
    'doping': (
        'Chemical potential against temperature at a fixed n- or p-type carrier concentration.',
        (),
    ),
    'doped_transport': (
        'Resistivity, conductivity, Seebeck coefficient and chemical potential of a doped '
        'semiconductor against temperature.',
        (qudt('ElectricConductivity'), qudt('SeebeckCoefficient')),
    ),
    'conductivity_component': (
        'One Cartesian component of the conductivity against energy, per spin channel.',
        (qudt('ElectricConductivity'),),
    ),
    'hall_dc': (
        'Static anomalous, spin or orbital Hall conductivity of one tensor component against '
        'the position of the Fermi level.',
        (ANOMALOUS_HALL, SPIN_HALL),
    ),
    'hall_ac': (
        'Frequency-dependent Hall response (magnetic, spin or orbital circular dichroism), real '
        'or imaginary part, against photon energy.',
        (),
    ),
    'berry_curvature': (
        'Berry curvature (or its spin or orbital counterpart) of one tensor component summed '
        'over the occupied bands along the k-path.',
        BERRY,
    ),
    'kubo_optical': (
        'Kubo-Greenwood optical response of one observable and tensor component against photon '
        'energy.',
        (),
    ),
    'kubo_response': (
        'Kubo-Bastin static response of one observable and tensor component against the '
        'position of the Fermi level.',
        (),
    ),
    'linear_response': (
        'Linear-response coefficient (conductivity, anomalous or spin Hall, Rashba-Edelstein), '
        'time-reversal even or odd part, against energy.',
        (),
    ),
    'rashba_edelstein': (
        'Rashba-Edelstein response: spin or orbital density induced by an electric field, '
        'against the position of the Fermi level.',
        (),
    ),
    'weyl_points': ('Positions, energies and chiralities of the Weyl points found.', ()),
    'z2_wcc': (
        'Wannier charge centres against k2 on one k3 plane, the data behind the Z2 invariant.',
        (),
    ),
    'mirror_chern': ('Mirror Chern number of each mirror sector.', ()),
    'spin_susceptibility': ('Static spin susceptibility of the self-consistent spin model.', ()),
    'z2': ('The Z2 topological invariants of the occupied bands.', ()),
    'berry_phase': ('Berry phase of the occupied bands along strings of k-points.', ()),
    'berry_bxsf': (
        'Berry curvature (or its spin or orbital counterpart) on the full k-grid in BXSF format.',
        BERRY,
    ),
    'emissivity_spectral': (
        'Hemispherical spectral emissivity of one tensor component against photon energy.',
        (),
    ),
    'emissivity_directional': (
        'Reflectivity or emissivity at one angle of incidence against photon energy.',
        (qudt('Reflectivity'),),
    ),
    'emissivity_total': (
        'Total hemispherical emissivity (Planck average) against temperature.',
        (),
    ),
    'optics': (
        'A linear optical function of one tensor component against photon energy: real or '
        'imaginary dielectric function, optical conductivity, energy-loss function, refractive '
        'index, extinction coefficient, absorption coefficient or reflectivity.',
        (
            qudt('PermittivityRatio'),
            qudt('RefractiveIndex'),
            qudt('LinearAbsorptionCoefficient'),
            qudt('Reflectivity'),
            OPTICAL,
        ),
    ),
    'jdos': ('Joint density of states against photon energy.', ()),
    'nlo': (
        'A second-order optical response of one tensor component against photon energy: shift '
        'current, injection current, second-harmonic susceptibility or photocurrent.',
        (SHIFT_CURRENT,),
    ),
    'shg_terms': (
        'One term of the second-harmonic susceptibility of one tensor component against photon '
        'energy.',
        (),
    ),
    'photocurrent_channels': (
        'One Fermi-sea or Fermi-surface channel of the photocurrent conductivity of one tensor '
        'component against photon energy.',
        (),
    ),
    'coupled_dielectric': (
        'Dielectric function at one level of the response kernel (independent particles, local '
        'fields, exchange-correlation kernels) against photon energy.',
        (qudt('PermittivityRatio'),),
    ),
    'coupled_shg': (
        'Second-harmonic susceptibility at one level of the response kernel against photon energy.',
        (),
    ),
    'molecular_beta': (
        'First hyperpolarizability of a molecule, one tensor component, against photon energy.',
        (),
    ),
    'nlo_summary': ('Text summary of a non-linear optics run.', ()),
    'phonon_bands': ('Phonon frequencies along a q-path, one column per branch.', ()),
    'phonon_dos': (
        'Phonon density of states against frequency.',
        (qudt('VibrationalDensityOfStates'),),
    ),
    'phonon_thermal': (
        'Harmonic free energy, entropy and heat capacity of the phonons against temperature.',
        (),
    ),
    'ir_spectrum': ('Infrared absorption spectrum of the zone-centre phonons.', ()),
    'raman_spectrum': ('Raman spectrum of the zone-centre phonons.', ()),
    'phonon_modes': (
        'Zone-centre phonon modes with their infrared or Raman activity.',
        (),
    ),
    'qha': (
        'A quasi-harmonic quantity (energy-volume curve, volume, thermal expansion, bulk '
        'modulus, Gibbs energy, heat capacity) against temperature or volume.',
        (),
    ),
    'alpha2f': (
        'Eliashberg spectral function (or its transport counterpart) against phonon energy.',
        ELPHON + ELIASHBERG,
    ),
    'gap_vs_t': (
        'Superconducting gap and mass renormalisation from the Migdal-Eliashberg equations '
        'against temperature.',
        (qudt('SuperconductorEnergyGap'),) + ELIASHBERG,
    ),
    'max_eigenvalue': (
        'Largest eigenvalue of the linearised Migdal-Eliashberg kernel against temperature; it '
        'reaches one at the critical temperature.',
        (qudt('SuperconductionTransitionTemperature'),) + ELIASHBERG,
    ),
    'mobility': (
        'Phonon-limited carrier mobilities, carrier densities, conductivity and Seebeck '
        'coefficient against temperature.',
        (qudt('Mobility'), qudt('ElectronDensity'), qudt('HoleDensity')) + ELPHON,
    ),
    'resistivity': (
        'Phonon-limited resistivity of a metal against temperature, total and per scattering '
        'channel.',
        ELPHON,
    ),
    'phonon_assisted_absorption': (
        'Phonon-assisted optical absorption coefficient against photon energy at one temperature.',
        (qudt('LinearAbsorptionCoefficient'),) + ELPHON,
    ),
    'indirect_eps2': (
        'Imaginary dielectric function from indirect (phonon-assisted) or direct transitions '
        'against photon energy at one temperature.',
        (qudt('PermittivityRatio'),),
    ),
    'thermal_emissivity': (
        'Spectral emissivity or dielectric function against photon energy at one temperature.',
        (),
    ),
    'thermal_emissivity_total': (
        'Total hemispherical emissivity of a slab and of an opaque half-space against temperature.',
        (),
    ),
    'scattering_rates': (
        'Electron-phonon scattering rates of the states against their energy at one temperature.',
        ELPHON,
    ),
    'eliashberg_self_energy': (
        'Real and imaginary parts of the electron self-energy from the Eliashberg function '
        'against energy at one temperature.',
        ELPHON,
    ),
    'coupling_path': (
        'Magnitude of the electron-phonon matrix element along a path of phonon wavevectors.',
        ELPHON,
    ),
    'thermal_summary': (
        'Summary table of the temperature renormalisation of the bands or of the coupling.',
        (),
    ),
    'eliashberg_iso': (
        'Isotropic Migdal-Eliashberg solution at one temperature in the file formats of EPW '
        '(imaginary axis, Pade, analytic continuation, quasiparticle density of states).',
        ELIASHBERG,
    ),
    'arpes_map': (
        'Simulated angle-resolved photoemission intensity on a momentum-energy map.',
        ARPES,
    ),
    'thermal_spectral': (
        'Electron spectral function on a momentum-energy map at a finite temperature.',
        (),
    ),
    'elphon_npz': ('Arrays of an electron-phonon workflow in NumPy format.', ELPHON),
    'conductance': (
        'Landauer transmission of the conductor against energy, in units of the conductance '
        'quantum.',
        (qudt('ElectricalConductance'), LANDAUER),
    ),
    'doscond': (
        'Density of states of the conductor region against energy.',
        DOS,
    ),
    'current': (
        'Current against bias voltage from the zero-bias transmission.',
        (qudt('ElectricCurrent'), qudt('Voltage'), LANDAUER),
    ),
    'finite_bias_current': (
        'Current against bias voltage with the potential profile under bias (self-consistent or '
        'linear).',
        (qudt('ElectricCurrent'), qudt('Voltage'), LANDAUER),
    ),
    'finite_bias_planar_average': (
        'Planar average of the potential energy under bias along the transport axis.',
        (),
    ),
    'finite_bias_potential': (
        'Potential energy under bias along the transport axis.',
        (),
    ),
    'finite_bias_charge': (
        'Charge induced by the bias along the transport axis.',
        (),
    ),
    'finite_bias_transmission': (
        'Transmission against energy for each bias voltage.',
        (LANDAUER,),
    ),
    'acbn0_u': ('Hubbard U and J of one orbital from ACBN0.', DFT_U),
    'acbn0_v': ('Intersite Hubbard V of one pair of orbitals from extended ACBN0.', DFT_U),
    'sscha': (
        'State of a stochastic self-consistent harmonic approximation run at one temperature.',
        (),
    ),
    'dhva': (
        'de Haas-van Alphen frequencies of the extremal Fermi-surface orbits against the field '
        'angle.',
        DHVA,
    ),
    'table': ('A column file that no other term describes.', ()),
    'grid_arrays': (
        'Band energies or spin or orbital texture of one band on the k-grid in NumPy format.',
        (),
    ),
    'npz': ('NumPy arrays that no other term describes.', ()),
    'json': ('A JSON file that no other term describes.', ()),
}


def optimade(name: str) -> str:
    """Address of an OPTIMADE property definition of a structure.

    Parameters
    ----------
    name : str
        Its name (``'lattice_vectors'``).

    Returns
    -------
    str
    """
    return 'https://schemas.optimade.org/defs/v1.2/properties/optimade/structures/' + name


#: key of ``inputs`` in the run record -> (definition, addresses of the shared terms).
STRUCTURE: dict[str, tuple[str, tuple[str, ...]]] = {
    'cell_bohr': (
        'The three lattice vectors of the cell, one per row, in bohr.',
        (qudt('LatticeVector'), optimade('lattice_vectors')),
    ),
    'positions_bohr': (
        'Cartesian position of each atom, in bohr, in the order of atoms.',
        (qudt('ParticlePositionVector'), optimade('cartesian_site_positions')),
    ),
    'atoms': ('Chemical symbol of each atom of the cell.', (optimade('species_at_sites'),)),
    'species': (
        'The distinct elements of the cell, each with the name of its pseudopotential file.',
        (optimade('species'),),
    ),
}


def vocabulary_markdown() -> str:
    """The vocabulary as a Markdown table, in the order of the catalogue.

    Returns
    -------
    str
        One row per term: term, category, definition, shared terms (linked).
    """
    from .catalogue import KINDS

    rows = ['| Term | Category | Definition | Shared terms |', '|---|---|---|---|']
    for kind in KINDS:
        definition, links = TERMS[kind.name]
        shared = ', '.join(
            '[%s](%s)' % ('PhySH' if 'doi.org' in link else link.rsplit('/', 1)[1], link)
            for link in links
        )
        rows.append('| `%s` | %s | %s | %s |' % (kind.name, kind.category, definition, shared))
    return '\n'.join(rows)


if __name__ == '__main__':
    print(vocabulary_markdown())
