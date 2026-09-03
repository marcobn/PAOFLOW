from types import SimpleNamespace

import numpy as np

from PAOFLOW.pyskeaf.config import (
    RYDBERG_IN_EV,
    SkeafConfig,
    read_config_in,
    write_config_in,
)
from PAOFLOW.pyskeaf.io_bxsf import BXSFData, read_bxsf
from PAOFLOW.pyskeaf.results import (
    Orbit,
    SKEAFResult,
    write_orbit_outlines,
    write_results_freqvsangle,
    write_results_long,
    write_results_short,
)
from PAOFLOW.writers.write2bxsf import write2bxsf


def test_read_bxsf_accepts_bandgrid_marker_variants(tmp_path):
    for marker in ('BEGIN_BANDGRID_3D_BANDS', 'BANDGRID_3D_BANDS'):
        bxsf = tmp_path / f'{marker}.bxsf'
        bxsf.write_text(
            f"""
BEGIN_INFO
  Fermi Energy:     0.000000000
END_INFO

BEGIN_BLOCK_BANDGRID_3D
band_energies
{marker}
           1
           2           2           2
    0.000000  0.000000  0.000000
    0.100000  0.000000  0.000000
    0.000000  0.100000  0.000000
    0.000000  0.000000  0.100000
  BAND:     1
    0.000000000    1.000000000    2.000000000    3.000000000
    4.000000000    5.000000000    6.000000000    7.000000000
END_BANDGRID_3D
END_BLOCK_BANDGRID_3d
""",
            encoding='utf-8',
        )

        data = read_bxsf(bxsf)

        assert (data.nx, data.ny, data.nz) == (2, 2, 2)
        assert data.num_kpoints == 8
        assert data.energies[1, 1, 1] == 7.0


def test_multiband_fermi_surface_bxsf_uses_shifted_zero_energy(tmp_path):
    controller = SimpleNamespace(
        data_dicts=lambda: (
            {'b_vectors': np.eye(3)},
            {
                'Efermi': 16.262988097,
                'alat': 1.0,
                'nk1': 1,
                'nk2': 1,
                'nk3': 1,
                'workpath': str(tmp_path),
                'outputdir': '.',
            },
        )
    )

    write2bxsf(
        controller,
        'FermiSurf_0.bxsf',
        np.zeros((1, 1, 1, 1)),
        nbnd=1,
        indices=[0],
        fermi_up=0.2,
        fermi_dw=-0.2,
    )

    text = (tmp_path / 'FermiSurf_0.bxsf').read_text(encoding='utf-8')
    fermi_line = next(line for line in text.splitlines() if 'Fermi Energy:' in line)
    assert float(fermi_line.split(':', 1)[1]) == 0.0
    assert 'Shift Range: -0.200000000eV to 0.200000000eV' in text


def test_read_config_ignores_bang_comments_before_fixed_width(tmp_path):
    config = tmp_path / 'config.in'
    config.write_text(
        """cylinder.bxsf                    ! Filename (50 chars. max)
   13.764336                     ! Fermi energy (eV)
 60                             ! Interpolated number of points per single side
  0.000000                       ! Theta (degrees)
  0.000000                       ! Phi (degrees)
r                                ! H-vector direction
  0.0000                         ! Minimum extremal FS freq. (kT)
  0.010                          ! Maximum fractional diff. between orbit freqs. for averaging
  0.150                          ! Maximum distance between orbit avg. coords. for averaging
y                                ! Allow extremal orbits near super-cell walls?
  0.000000                       ! Starting theta (degrees)
  0.000000                       ! Ending theta (degrees)
  0.000000                       ! Starting phi (degrees)
 90.000000                       ! Ending phi (degrees)
   19                            ! Number of rotation angles
""",
        encoding='utf-8',
    )

    cfg = read_config_in(config)

    assert cfg.filename == 'cylinder.bxsf'
    assert abs(cfg.fermi_energy - 1.011660) < 1.0e-6
    assert cfg.numint == 60
    assert cfg.num_rots == 19


def test_write_config_uses_bang_comments(tmp_path):
    config = tmp_path / 'config.in'

    write_config_in(
        SkeafConfig(filename='tiny-pocket.bxsf', fermi_energy=1.0, numint=120),
        config,
    )

    text = config.read_text(encoding='utf-8')
    assert '! Filename' in text
    assert '13.605693' in text
    assert 'Fermi energy (eV)' in text
    assert f'{chr(124)} Filename' not in text


def test_short_and_long_results_report_fermi_energy_in_ev(tmp_path):
    bxsf = BXSFData(
        filename='band.bxsf',
        fermi_energy=0.0,
        nx=2,
        ny=2,
        nz=2,
        recip_au=np.eye(3),
        recip_ang=np.eye(3),
        energies=np.zeros((2, 2, 2)),
    )
    result = SKEAFResult(
        config_filename='',
        bxsf_filename=bxsf.filename,
        fermi_energy=1.0,
        config=SkeafConfig(),
        bxsf=bxsf,
    )
    short_path = tmp_path / 'qo_results_short.out'
    long_path = tmp_path / 'qo_results_long.out'

    write_results_short(result, short_path)
    write_results_long(result, long_path)

    for path in (short_path, long_path):
        text = path.read_text(encoding='utf-8')
        assert 'Fermi energy:    13.605693 eV' in text
        assert ' Ryd ' not in text


def test_result_outputs_use_public_angle_names_and_aligned_columns(tmp_path):
    bxsf = BXSFData(
        filename='band.bxsf',
        fermi_energy=0.0,
        nx=2,
        ny=2,
        nz=2,
        recip_au=np.eye(3),
        recip_ang=np.eye(3),
        energies=np.zeros((2, 2, 2)),
    )
    config = SkeafConfig(
        hvd='r',
        numint=60,
        num_rots=19,
        theta_start=0.0,
        theta_end=np.pi / 4.0,
        phi_start=0.0,
        phi_end=np.pi / 2.0,
        allow_ext_near_walls=True,
    )
    orbit = Orbit(
        theta=np.pi / 4.0,
        phi=np.pi / 2.0,
        frequency_kT=1.25,
        freq_uncertainty_kT=0.01,
        effective_mass=0.5,
        curvature=-2.0,
        orbit_type=-1,
        num_copies=4,
    )
    result = SKEAFResult(
        config_filename='',
        bxsf_filename=bxsf.filename,
        fermi_energy=0.0,
        orbits=[orbit],
        config=config,
        bxsf=bxsf,
    )
    short_path = tmp_path / 'qo_results_short.out'
    long_path = tmp_path / 'qo_results_long.out'
    frequency_path = tmp_path / 'qo_results_freqvsangle.out'
    outline_ang_path = tmp_path / 'qo_results_orbitoutlines_invAng.out'
    outline_au_path = tmp_path / 'qo_results_orbitoutlines_invau.out'

    write_results_short(result, short_path)
    write_results_long(result, long_path)
    write_results_freqvsangle(result, frequency_path)
    write_orbit_outlines(result, outline_ang_path, outline_au_path)

    for path in (short_path, long_path):
        text = path.read_text(encoding='utf-8')
        assert 'num_interpolation =   60' in text
        assert 'b_field = rotation' in text
        assert 'num_angles =    19' in text
        assert 'Azimuthal =' in text
        assert 'Polar =' in text
        assert 'azimuthal =' in text
        assert 'polar =' in text
        assert 'Theta' not in text and 'Phi' not in text
        assert 'theta =' not in text and 'phi =' not in text
        assert 'Minimum extremal FS freq.:' in text
        assert 'Maximum fractional diff. between orbit freqs. for averaging:' in text
        assert (
            'Maximum distance (fraction of RUC side length) between orbit avg. '
            'coords. for averaging:'
        ) in text
        assert (
            'Extremal orbits near super-cell walls are ALLOWED to be included ' 'in the output.'
        ) in text

    for path in (outline_ang_path, outline_au_path):
        text = path.read_text(encoding='utf-8')
        assert 'Azimuthal(deg)' in text
        assert 'Polar(deg)' in text
        assert 'Theta(deg)' not in text and 'Phi(deg)' not in text

    lines = frequency_path.read_text(encoding='utf-8').splitlines()
    assert [field.strip() for field in lines[0].split(',')] == [
        'Azimuthal(deg)',
        'Polar(deg)',
        'Freq(kT)',
        'mstar(me)',
        'Curv(kTA2)',
        'Type(+e-h)',
        'NumOrbCopy',
    ]
    assert [index for index, char in enumerate(lines[0]) if char == ','] == [
        index for index, char in enumerate(lines[1]) if char == ','
    ]
    assert ',  ' in lines[0] and ',  ' in lines[1]
    data = np.loadtxt(frequency_path, delimiter=',', skiprows=1)
    assert np.allclose(data[:3], [45.0, 90.0, 1.25])


def test_default_result_filenames_include_fermi_energy_ev(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    bxsf = BXSFData(
        filename='band.bxsf',
        fermi_energy=0.0,
        nx=2,
        ny=2,
        nz=2,
        recip_au=np.eye(3),
        recip_ang=np.eye(3),
        energies=np.zeros((2, 2, 2)),
    )
    result = SKEAFResult(
        config_filename='',
        bxsf_filename=bxsf.filename,
        fermi_energy=0.2 / RYDBERG_IN_EV,
        config=SkeafConfig(),
        bxsf=bxsf,
    )

    write_results_freqvsangle(result)
    write_results_short(result)
    write_results_long(result)
    write_orbit_outlines(result)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        'qo_EF_0.2_freqvsangle.out',
        'qo_EF_0.2_long.out',
        'qo_EF_0.2_orbitoutlines_invAng.out',
        'qo_EF_0.2_orbitoutlines_invau.out',
        'qo_EF_0.2_short.out',
    ]
