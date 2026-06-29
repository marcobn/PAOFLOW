from PAOFLOW.pyskeaf.config import read_config_in, SkeafConfig, write_config_in
from PAOFLOW.pyskeaf.io_bxsf import read_bxsf


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


def test_read_config_ignores_bang_comments_before_fixed_width(tmp_path):
    config = tmp_path / 'config.in'
    config.write_text(
        """cylinder.bxsf                    ! Filename (50 chars. max)
    1.011660                     ! Fermi energy (Rydbergy)
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
    assert cfg.numint == 60
    assert cfg.num_rots == 19


def test_write_config_uses_bang_comments(tmp_path):
    config = tmp_path / 'config.in'

    write_config_in(SkeafConfig(filename='tiny-pocket.bxsf', numint=120), config)

    text = config.read_text(encoding='utf-8')
    assert '! Filename' in text
    assert f'{chr(124)} Filename' not in text
