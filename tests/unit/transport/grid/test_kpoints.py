"""Unit tests for k-point and R-point grid utilities."""

import numpy as np
import pytest

from PAOFLOW.transport.grid.kpoints import (
    KpointsData,
    compute_fourier_phase_table,
    compute_ivr_par,
    initialize_kpoints,
    initialize_meshsize,
    initialize_r_vectors,
    kpoints_equivalent,
    kpoints_mask,
)


@pytest.mark.unit
def test_kpoints_mask_inserts_transport_direction():
    """The transport direction should receive the inserted init value."""
    vect = (2, 3)

    np.testing.assert_allclose(kpoints_mask(vect, 1, 'x'), [1, 2, 3])
    np.testing.assert_allclose(kpoints_mask(vect, 1, 'y'), [2, 1, 3])
    np.testing.assert_allclose(kpoints_mask(vect, 1, 'z'), [2, 3, 1])


@pytest.mark.unit
def test_kpoints_mask_invalid_inputs():
    """Invalid vector shapes or directions should raise errors."""
    with pytest.raises(ValueError):
        kpoints_mask((1, 2, 3), 0, 'x')

    with pytest.raises(ValueError):
        kpoints_mask((1, 2), 0, 'a')


@pytest.mark.unit
def test_kpoints_equivalent_time_reversal():
    """Time-reversal partners are equivalent modulo 1."""
    v1 = np.array([0.25, 0.5])
    v2 = np.array([-0.25, -0.5])

    assert kpoints_equivalent(v1, v2)
    assert not kpoints_equivalent(v1, v1)


@pytest.mark.unit
def test_initialize_meshsize_defaults_to_nr_par():
    """When nk_par is missing, it should mirror the R-mesh size."""
    nk_par, nr_par = initialize_meshsize(np.array([2, 3, 4]), transport_direction='x')

    np.testing.assert_allclose(nr_par, [3, 4])
    np.testing.assert_allclose(nk_par, [3, 4])


@pytest.mark.unit
def test_initialize_meshsize_safe_mesh_enforced():
    """Safe mesh option rejects nk_par smaller than nr_par."""
    with pytest.raises(ValueError):
        initialize_meshsize(
            np.array([2, 2, 2]),
            transport_direction='z',
            nk_par=np.array([1, 1]),
            use_safe_kmesh=True,
        )


@pytest.mark.unit
def test_initialize_kpoints_symmetry_weights():
    """Symmetrized k-mesh should return normalized weights."""
    vkpts, weights = initialize_kpoints(
        nk_par=np.array([2, 2]),
        s_par=np.array([0, 0]),
        transport_direction='z',
        use_sym=True,
    )

    assert weights.sum() == pytest.approx(1.0)
    assert vkpts.shape[1] == 3


@pytest.mark.unit
def test_initialize_kpoints_no_symmetry_full_mesh():
    """Disabling symmetry keeps full mesh size."""
    vkpts, weights = initialize_kpoints(
        nk_par=np.array([2, 2]),
        s_par=np.array([0, 0]),
        transport_direction='z',
        use_sym=False,
    )

    assert len(vkpts) == 4
    assert len(weights) == 4


@pytest.mark.unit
def test_compute_fourier_phase_table_basic():
    """Phase table should match exp(i 2pi k.R) for trivial inputs."""
    vkpts = np.array([[0.0, 0.0, 0.0]])
    ivr = np.array([[0, 0, 0], [1, 0, 0]])

    table = compute_fourier_phase_table(vkpts, ivr)

    np.testing.assert_allclose(table[:, 0], [1.0 + 0.0j, 1.0 + 0.0j])


@pytest.mark.unit
def test_initialize_r_vectors_include_negatives():
    """Hermitian symmetry ensures -R is present (except for self-inverse R=0)."""
    ivr_par3d, wr_par = initialize_r_vectors((1, 1), transport_direction='z')

    assert ivr_par3d.shape[1] == 3
    assert wr_par.sum() == pytest.approx(1.0)

    for vec in ivr_par3d:
        if np.all(vec == 0):
            continue
        assert any(np.all(other == -vec) for other in ivr_par3d)


@pytest.mark.unit
def test_compute_ivr_par_returns_transposed_grid():
    """compute_ivr_par returns a (2, nR) array of integer vectors."""
    ivr_par, wr_par = compute_ivr_par((2, 1))

    assert ivr_par.shape[0] == 2
    assert ivr_par.shape[1] == len(wr_par)


@pytest.mark.unit
def test_kpointsdata_memory_usage_counts_arrays():
    """Memory usage should scale with present arrays."""
    data = KpointsData()
    data.vkpt_par = np.zeros((2, 2))
    data.wk_par = np.zeros(2)

    mem = data.memory_usage()

    assert mem > 0.0
