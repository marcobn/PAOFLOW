# The run record

Every PAOFLOW run writes a run record, `paoflow_outputs.json`, into its output folder. You do
not switch it on and there is nothing to configure. It answers, months later, the questions a
data-availability statement or a referee asks: which calculation produced this file, with which
code, from which inputs, in which units.

## What it holds

One entry per run of the folder:

- an identifier, and when the run started and finished (UTC);
- the PAOFLOW version, and the commit when PAOFLOW runs from a git checkout;
- the DFT calculation it started from: code and version, functional, cell, atoms,
  pseudopotentials, and a checksum (sha256) of the input files;
- your script, with its checksum, and every PAOFLOW call it made, in order, with the arguments
  (defaults included), the time and the memory each took;
- the versions of Python, numpy, scipy and mpi4py;
- the machine as far as it matters: MPI ranks, threads, CPU model, memory, BLAS library;
- every file the run wrote: what it is (its *term*, below), the quantity and unit of its axes,
  its size and its checksum.

The record is rewritten after every call. If the job is killed, the record is there, marked
`unfinished`, and shows the last call that completed. `finish_execution()` marks it `finished`.

It never holds the name of the machine, your user name, a job number or an absolute path.

The output files themselves do not change: units and meaning travel in the record, not in file
headers.

In PAOFLOW Studio the record is the *Run information* entry of the Results tab.

## Format

The record is JSON. Its schema ships with the package
(`PAOFLOW/writers/run_record.schema.json`); later versions only add keys.

```{literalinclude} ../../../src/PAOFLOW/writers/run_record.schema.json
:language: json
```

## Terms

Each output file is labelled with a term: the name of its file layout (columns, units, energy
reference, how spin and tensor components split across files). A term keeps its meaning across
versions. Where a shared vocabulary has the physical concept, the term links to it (QUDT for
quantities, PhySH for subjects).

<!-- vocabulary: python -m PAOFLOW.results.vocabulary -->
| Term | Category | Definition | Shared terms |
|---|---|---|---|
| `manifest` | Run information | The run record: what the runs of this folder did and wrote. |  |
| `berry_bands` | Topology and Hall | Berry curvature (or its spin or orbital counterpart) of one tensor component along the band path, one column per k-point index. | [PhySH](https://doi.org/10.29172/8abddf0b-ea8e-4c03-af96-883903427079) |
| `bands` | Electronic structure | Band energies along a k-path: the k-point index, then one column per band, one file per spin channel. Energies are relative to the Fermi level. |  |
| `site_projected_bands` | Electronic structure | Band energies along a k-path with the weight of each band on the atomic sites. Energies are relative to the Fermi level. |  |
| `velocity` | Electronic structure | Band velocity along one Cartesian direction on the k-path, one column per band. |  |
| `pdos` | Electronic structure | Density of states projected on one atomic orbital, per spin channel. Energies are relative to the Fermi level. | [EnergyDensityOfStates](https://qudt.org/vocab/quantitykind/EnergyDensityOfStates), [PhySH](https://doi.org/10.29172/aaff2519-9891-4d4e-a067-1ea36b6b08d4) |
| `dos` | Electronic structure | Density of states of the PAO Hamiltonian, or the sum of its projections, per spin channel. Energies are relative to the Fermi level. | [EnergyDensityOfStates](https://qudt.org/vocab/quantitykind/EnergyDensityOfStates), [PhySH](https://doi.org/10.29172/aaff2519-9891-4d4e-a067-1ea36b6b08d4) |
| `spin_texture` | Electronic structure | Expectation value of the spin (or orbital) operator on each band along a k-path. |  |
| `fermi_surface` | Electronic structure | Band energies on the full k-grid in XCrySDen BXSF format, all bands that cross the chosen window in one file. | [PhySH](https://doi.org/10.29172/78d12677-c7dc-427e-acd7-cbff78a1713f) |
| `fermi_sheet` | Quantum oscillations | Energies of a single band on the k-grid in BXSF format, the input of pyskeaf. | [PhySH](https://doi.org/10.29172/78d12677-c7dc-427e-acd7-cbff78a1713f) |
| `effective_mass` | Electronic structure | Effective-mass tensors of the band extrema, in units of the electron mass. | [EffectiveMass](https://qudt.org/vocab/quantitykind/EffectiveMass) |
| `projectability` | Electronic structure | Projectability of each Kohn-Sham state on the atomic-orbital basis against its energy. |  |
| `boltzmann` | Boltzmann transport | A Boltzmann transport tensor in the constant relaxation-time form (conductivity, Seebeck coefficient, electronic thermal conductivity or power factor) against the chemical potential, for each temperature. | [ElectricConductivity](https://qudt.org/vocab/quantitykind/ElectricConductivity), [SeebeckCoefficient](https://qudt.org/vocab/quantitykind/SeebeckCoefficient), [ThermalConductivity](https://qudt.org/vocab/quantitykind/ThermalConductivity) |
| `hall_coefficient` | Boltzmann transport | Hall coefficient from Boltzmann transport against the chemical potential, grouped by temperature. | [HallCoefficient](https://qudt.org/vocab/quantitykind/HallCoefficient) |
| `nernst` | Boltzmann transport | Nernst coefficient per relaxation time from Boltzmann transport against the chemical potential, grouped by temperature. |  |
| `doping` | Boltzmann transport | Chemical potential against temperature at a fixed n- or p-type carrier concentration. |  |
| `doped_transport` | Boltzmann transport | Resistivity, conductivity, Seebeck coefficient and chemical potential of a doped semiconductor against temperature. | [ElectricConductivity](https://qudt.org/vocab/quantitykind/ElectricConductivity), [SeebeckCoefficient](https://qudt.org/vocab/quantitykind/SeebeckCoefficient) |
| `conductivity_component` | Boltzmann transport | One Cartesian component of the conductivity against energy, per spin channel. | [ElectricConductivity](https://qudt.org/vocab/quantitykind/ElectricConductivity) |
| `hall_dc` | Topology and Hall | Static anomalous, spin or orbital Hall conductivity of one tensor component against the position of the Fermi level. | [PhySH](https://doi.org/10.29172/8adc9a72-2460-4d3d-ad53-a44092c6a198), [PhySH](https://doi.org/10.29172/bb4a3b78-a738-4e19-abd6-1feea6226ea1) |
| `hall_ac` | Topology and Hall | Frequency-dependent Hall response (magnetic, spin or orbital circular dichroism), real or imaginary part, against photon energy. |  |
| `berry_curvature` | Topology and Hall | Berry curvature (or its spin or orbital counterpart) of one tensor component summed over the occupied bands along the k-path. | [PhySH](https://doi.org/10.29172/8abddf0b-ea8e-4c03-af96-883903427079) |
| `kubo_optical` | Topology and Hall | Kubo-Greenwood optical response of one observable and tensor component against photon energy. |  |
| `kubo_response` | Topology and Hall | Kubo-Bastin static response of one observable and tensor component against the position of the Fermi level. |  |
| `linear_response` | Topology and Hall | Linear-response coefficient (conductivity, anomalous or spin Hall, Rashba-Edelstein), time-reversal even or odd part, against energy. |  |
| `rashba_edelstein` | Topology and Hall | Rashba-Edelstein response: spin or orbital density induced by an electric field, against the position of the Fermi level. |  |
| `weyl_points` | Topology and Hall | Positions, energies and chiralities of the Weyl points found. |  |
| `z2_wcc` | Topology and Hall | Wannier charge centres against k2 on one k3 plane, the data behind the Z2 invariant. |  |
| `mirror_chern` | Topology and Hall | Mirror Chern number of each mirror sector. |  |
| `spin_susceptibility` | Electronic structure | Static spin susceptibility of the self-consistent spin model. |  |
| `z2` | Topology and Hall | The Z2 topological invariants of the occupied bands. |  |
| `berry_phase` | Topology and Hall | Berry phase of the occupied bands along strings of k-points. |  |
| `berry_bxsf` | Topology and Hall | Berry curvature (or its spin or orbital counterpart) on the full k-grid in BXSF format. | [PhySH](https://doi.org/10.29172/8abddf0b-ea8e-4c03-af96-883903427079) |
| `emissivity_spectral` | Optics | Hemispherical spectral emissivity of one tensor component against photon energy. |  |
| `emissivity_directional` | Optics | Reflectivity or emissivity at one angle of incidence against photon energy. | [Reflectivity](https://qudt.org/vocab/quantitykind/Reflectivity) |
| `emissivity_total` | Optics | Total hemispherical emissivity (Planck average) against temperature. |  |
| `optics` | Optics | A linear optical function of one tensor component against photon energy: real or imaginary dielectric function, optical conductivity, energy-loss function, refractive index, extinction coefficient, absorption coefficient or reflectivity. | [PermittivityRatio](https://qudt.org/vocab/quantitykind/PermittivityRatio), [RefractiveIndex](https://qudt.org/vocab/quantitykind/RefractiveIndex), [LinearAbsorptionCoefficient](https://qudt.org/vocab/quantitykind/LinearAbsorptionCoefficient), [Reflectivity](https://qudt.org/vocab/quantitykind/Reflectivity), [PhySH](https://doi.org/10.29172/05ff9faf-4943-45b0-b627-49eb8a9404c6) |
| `jdos` | Optics | Joint density of states against photon energy. |  |
| `nlo` | Non-linear optics | A second-order optical response of one tensor component against photon energy: shift current, injection current, second-harmonic susceptibility or photocurrent. | [PhySH](https://doi.org/10.29172/9fffec69-8e96-42d0-b6bf-42d1234858eb) |
| `shg_terms` | Non-linear optics | One term of the second-harmonic susceptibility of one tensor component against photon energy. |  |
| `photocurrent_channels` | Non-linear optics | One Fermi-sea or Fermi-surface channel of the photocurrent conductivity of one tensor component against photon energy. |  |
| `coupled_dielectric` | Non-linear optics | Dielectric function at one level of the response kernel (independent particles, local fields, exchange-correlation kernels) against photon energy. | [PermittivityRatio](https://qudt.org/vocab/quantitykind/PermittivityRatio) |
| `coupled_shg` | Non-linear optics | Second-harmonic susceptibility at one level of the response kernel against photon energy. |  |
| `molecular_beta` | Non-linear optics | First hyperpolarizability of a molecule, one tensor component, against photon energy. |  |
| `nlo_summary` | Non-linear optics | Text summary of a non-linear optics run. |  |
| `phonon_bands` | Phonons | Phonon frequencies along a q-path, one column per branch. |  |
| `phonon_dos` | Phonons | Phonon density of states against frequency. | [VibrationalDensityOfStates](https://qudt.org/vocab/quantitykind/VibrationalDensityOfStates) |
| `phonon_thermal` | Phonons | Harmonic free energy, entropy and heat capacity of the phonons against temperature. |  |
| `ir_spectrum` | Phonons | Infrared absorption spectrum of the zone-centre phonons. |  |
| `raman_spectrum` | Phonons | Raman spectrum of the zone-centre phonons. |  |
| `phonon_modes` | Phonons | Zone-centre phonon modes with their infrared or Raman activity. |  |
| `qha` | Phonons | A quasi-harmonic quantity (energy-volume curve, volume, thermal expansion, bulk modulus, Gibbs energy, heat capacity) against temperature or volume. |  |
| `alpha2f` | Electron–phonon | Eliashberg spectral function (or its transport counterpart) against phonon energy. | [PhySH](https://doi.org/10.29172/2cb8570a-bc2f-41ab-a3f1-1467b288c3f0), [PhySH](https://doi.org/10.29172/c6f039f8-0580-4488-a6e0-b8bf5455a86d) |
| `gap_vs_t` | Electron–phonon | Superconducting gap and mass renormalisation from the Migdal-Eliashberg equations against temperature. | [SuperconductorEnergyGap](https://qudt.org/vocab/quantitykind/SuperconductorEnergyGap), [PhySH](https://doi.org/10.29172/c6f039f8-0580-4488-a6e0-b8bf5455a86d) |
| `max_eigenvalue` | Electron–phonon | Largest eigenvalue of the linearised Migdal-Eliashberg kernel against temperature; it reaches one at the critical temperature. | [SuperconductionTransitionTemperature](https://qudt.org/vocab/quantitykind/SuperconductionTransitionTemperature), [PhySH](https://doi.org/10.29172/c6f039f8-0580-4488-a6e0-b8bf5455a86d) |
| `mobility` | Electron–phonon | Phonon-limited carrier mobilities, carrier densities, conductivity and Seebeck coefficient against temperature. | [Mobility](https://qudt.org/vocab/quantitykind/Mobility), [ElectronDensity](https://qudt.org/vocab/quantitykind/ElectronDensity), [HoleDensity](https://qudt.org/vocab/quantitykind/HoleDensity), [PhySH](https://doi.org/10.29172/2cb8570a-bc2f-41ab-a3f1-1467b288c3f0) |
| `resistivity` | Electron–phonon | Phonon-limited resistivity of a metal against temperature, total and per scattering channel. | [PhySH](https://doi.org/10.29172/2cb8570a-bc2f-41ab-a3f1-1467b288c3f0) |
| `phonon_assisted_absorption` | Electron–phonon | Phonon-assisted optical absorption coefficient against photon energy at one temperature. | [LinearAbsorptionCoefficient](https://qudt.org/vocab/quantitykind/LinearAbsorptionCoefficient), [PhySH](https://doi.org/10.29172/2cb8570a-bc2f-41ab-a3f1-1467b288c3f0) |
| `indirect_eps2` | Electron–phonon | Imaginary dielectric function from indirect (phonon-assisted) or direct transitions against photon energy at one temperature. | [PermittivityRatio](https://qudt.org/vocab/quantitykind/PermittivityRatio) |
| `thermal_emissivity` | Electron–phonon | Spectral emissivity or dielectric function against photon energy at one temperature. |  |
| `thermal_emissivity_total` | Electron–phonon | Total hemispherical emissivity of a slab and of an opaque half-space against temperature. |  |
| `scattering_rates` | Electron–phonon | Electron-phonon scattering rates of the states against their energy at one temperature. | [PhySH](https://doi.org/10.29172/2cb8570a-bc2f-41ab-a3f1-1467b288c3f0) |
| `eliashberg_self_energy` | Electron–phonon | Real and imaginary parts of the electron self-energy from the Eliashberg function against energy at one temperature. | [PhySH](https://doi.org/10.29172/2cb8570a-bc2f-41ab-a3f1-1467b288c3f0) |
| `coupling_path` | Electron–phonon | Magnitude of the electron-phonon matrix element along a path of phonon wavevectors. | [PhySH](https://doi.org/10.29172/2cb8570a-bc2f-41ab-a3f1-1467b288c3f0) |
| `thermal_summary` | Electron–phonon | Summary table of the temperature renormalisation of the bands or of the coupling. |  |
| `eliashberg_iso` | Electron–phonon | Isotropic Migdal-Eliashberg solution at one temperature in the file formats of EPW (imaginary axis, Pade, analytic continuation, quasiparticle density of states). | [PhySH](https://doi.org/10.29172/c6f039f8-0580-4488-a6e0-b8bf5455a86d) |
| `arpes_map` | Electron–phonon | Simulated angle-resolved photoemission intensity on a momentum-energy map. | [PhySH](https://doi.org/10.29172/a4074854-ac2f-4771-9451-0ce9bbad43ce) |
| `thermal_spectral` | Electron–phonon | Electron spectral function on a momentum-energy map at a finite temperature. |  |
| `elphon_npz` | Electron–phonon | Arrays of an electron-phonon workflow in NumPy format. | [PhySH](https://doi.org/10.29172/2cb8570a-bc2f-41ab-a3f1-1467b288c3f0) |
| `conductance` | Green's-function transport | Landauer transmission of the conductor against energy, in units of the conductance quantum. | [ElectricalConductance](https://qudt.org/vocab/quantitykind/ElectricalConductance), [PhySH](https://doi.org/10.29172/9c19d585-8eec-411b-af4c-0412fb91e670) |
| `doscond` | Green's-function transport | Density of states of the conductor region against energy. | [EnergyDensityOfStates](https://qudt.org/vocab/quantitykind/EnergyDensityOfStates), [PhySH](https://doi.org/10.29172/aaff2519-9891-4d4e-a067-1ea36b6b08d4) |
| `current` | Green's-function transport | Current against bias voltage from the zero-bias transmission. | [ElectricCurrent](https://qudt.org/vocab/quantitykind/ElectricCurrent), [Voltage](https://qudt.org/vocab/quantitykind/Voltage), [PhySH](https://doi.org/10.29172/9c19d585-8eec-411b-af4c-0412fb91e670) |
| `finite_bias_current` | Green's-function transport | Current against bias voltage with the potential profile under bias (self-consistent or linear). | [ElectricCurrent](https://qudt.org/vocab/quantitykind/ElectricCurrent), [Voltage](https://qudt.org/vocab/quantitykind/Voltage), [PhySH](https://doi.org/10.29172/9c19d585-8eec-411b-af4c-0412fb91e670) |
| `finite_bias_planar_average` | Green's-function transport | Planar average of the potential energy under bias along the transport axis. |  |
| `finite_bias_potential` | Green's-function transport | Potential energy under bias along the transport axis. |  |
| `finite_bias_charge` | Green's-function transport | Charge induced by the bias along the transport axis. |  |
| `finite_bias_transmission` | Green's-function transport | Transmission against energy for each bias voltage. | [PhySH](https://doi.org/10.29172/9c19d585-8eec-411b-af4c-0412fb91e670) |
| `acbn0_u` | ACBN0 | Hubbard U and J of one orbital from ACBN0. | [PhySH](https://doi.org/10.29172/ffcb7c9c-2fd2-4caa-ac0e-9832b20fc351) |
| `acbn0_v` | ACBN0 | Intersite Hubbard V of one pair of orbitals from extended ACBN0. | [PhySH](https://doi.org/10.29172/ffcb7c9c-2fd2-4caa-ac0e-9832b20fc351) |
| `sscha` | Anharmonic phonons (SSCHA) | State of a stochastic self-consistent harmonic approximation run at one temperature. |  |
| `dhva` | Quantum oscillations | de Haas-van Alphen frequencies of the extremal Fermi-surface orbits against the field angle. | [PhySH](https://doi.org/10.29172/2f031f28-5911-407e-8bce-149e70acf79b) |
| `table` | Other files | A column file that no other term describes. |  |
| `grid_arrays` | Electronic structure | Band energies or spin or orbital texture of one band on the k-grid in NumPy format. |  |
| `npz` | Other files | NumPy arrays that no other term describes. |  |
| `json` | Other files | A JSON file that no other term describes. |  |
<!-- end vocabulary -->
