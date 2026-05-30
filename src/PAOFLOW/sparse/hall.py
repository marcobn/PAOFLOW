from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mpi4py import MPI

from ..perturb_split import perturb_split
from .operators import _iter_streamed_derivative_batches

if TYPE_CHECKING:
    from ..DataController import DataController


comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def _hall_energy_grid(attributes: dict[str, object]) -> np.ndarray:
    """Build the energy mesh used for Hall spectra.

    Parameters
    ----------
    attributes : dict[str, object]
        Runtime attributes containing the Hall energy-window configuration.

    Returns
    -------
    numpy.ndarray
        Energy grid with shape ``(esizeH,)``.

    Notes
    -----
    The Hall outputs are reported on a one-dimensional energy mesh between the
    user-specified bounds. The sparse implementation keeps the same energy-grid
    convention as the dense code so that only the operator handling changes, not
    the final observable being plotted or written.

    Parallelization strategy:
        No MPI communication is needed because every rank can rebuild the same
        small energy mesh from the replicated scalar attributes.
    """
    shift = float(attributes['shift'])
    if shift != 0.0:
        attributes['emaxH'] = np.amin(np.array([shift, float(attributes['emaxH'])]))
    return np.linspace(
        float(attributes['eminH']), float(attributes['emaxH']), int(attributes['esizeH'])
    )


def _accumulate_berry_curvature_local(
    omega_bands: np.ndarray,
    ik_local: int,
    jksp: np.ndarray,
    pksp: np.ndarray,
    eigvals: np.ndarray,
    deltap: float,
) -> None:
    """Accumulate the Berry-curvature weight for one local k-point.

    Parameters
    ----------
    omega_bands : numpy.ndarray
        Local accumulator ``Omega_n(k)`` with shape ``(nkp_local, nbands)``.
    ik_local : int
        Local k-point index to update.
    jksp : numpy.ndarray
        First projected operator in the band basis, shape ``(nbands, nbands)``.
    pksp : numpy.ndarray
        Second projected operator in the band basis, shape ``(nbands, nbands)``.
    eigvals : numpy.ndarray
        Band energies for the same local k-point and spin channel.
    deltap : float
        Broadening entering the Berry-curvature denominator.

    Returns
    -------
    None
        Updates ``omega_bands`` in place.

    Notes
    -----
    This routine evaluates the band-resolved Berry-curvature weight from the
    standard interband expression

    ``Omega_n(k) ~ -2 Im[j_nm p_mn] / ((E_n - E_m)^2 + delta^2)``.

    Because the operators are already projected into the local band basis, the
    sparse workflow can accumulate the physical quantity of interest directly
    without storing dense global derivative tensors.

    Parallelization strategy:
        The contraction is local to one rank-owned k-point. Only the compact
        accumulated spectra are communicated later.
    """
    energy_differences = (eigvals - eigvals[:, None]) ** 2 + deltap**2
    energy_differences[np.where(energy_differences < 1.0e-4)] = np.inf
    omega_bands[ik_local, : eigvals.size] = -2.0 * np.sum(
        np.imag(jksp * pksp.T) / energy_differences,
        axis=1,
    )


def _project_operator_pair(
    operator_left: np.ndarray,
    operator_right: np.ndarray,
    vecs: np.ndarray,
    degen: list[np.ndarray] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project two operators into the degenerate-aware band basis.

    Parameters
    ----------
    operator_left, operator_right : numpy.ndarray
        Orbital-basis operators with shape ``(nbands, nbands)``.
    vecs : numpy.ndarray
        Eigenvectors defining the current band basis.
    degen : list[numpy.ndarray] | numpy.ndarray
        Degeneracy information used by ``perturb_split``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The two projected operators in the band basis.

    Notes
    -----
    Hall formulas require operator matrix elements in a basis that treats
    degenerate subspaces consistently. This helper fixes the sparse Hall call
    site to the two projected operators that are actually needed.
    """
    projected = perturb_split(operator_left, operator_right, vecs, degen)
    return projected[0], projected[1]


def _finalize_berry_curvature(
    data_controller: DataController,
    omega_bands: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Convert local Berry-curvature weights into final Hall outputs.

    Parameters
    ----------
    data_controller : DataController
        Runtime container with local eigendata, smearing widths, and FFT-grid
        metadata.
    omega_bands : numpy.ndarray
        Local band-resolved Berry-curvature weights, shape
        ``(nkp_local, nbands)``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray | None, numpy.ndarray | None]
        Energy grid, root-only integrated Hall curve, and root-only FFT-grid
        Berry-curvature difference used for ``.bxsf`` output.

    Notes
    -----
    Once the band-resolved weights ``Omega_n(k)`` are known, the remaining step
    is to fold them with the occupations as a function of energy and then sum
    over k-points. The sparse path stores only these compact weights and applies
    the same occupation logic as the dense code, which preserves the observable
    while avoiding dense operator storage.

    Parallelization strategy:
        Each rank converts only its local ``Omega_n(k)`` slice into
        energy-dependent quantities. MPI communication is limited to gathering or
        reducing the final compact outputs.
    """
    from ..communication import gather_full
    from ..smearing import intgaussian, intmetpax

    arrays, attributes = data_controller.data_dicts()
    assert arrays is not None and attributes is not None

    ene = _hall_energy_grid(attributes)
    band_count = int(omega_bands.shape[1])
    omega_energy_local = np.zeros((omega_bands.shape[0], ene.size), dtype=float)
    eigvals_local = arrays['E_k'][:, :band_count, 0]
    deltakp_local = arrays['deltakp'][:, :band_count, 0]

    for iene, energy in enumerate(ene):
        if attributes['smearing'] == 'gauss':
            weights = intgaussian(eigvals_local, energy, deltakp_local)
        elif attributes['smearing'] == 'm-p':
            weights = intmetpax(eigvals_local, energy, deltakp_local)
        else:
            weights = 0.5 * (-np.sign(eigvals_local - energy) + 1.0)
        omega_energy_local[:, iene] = np.sum(omega_bands[:, :band_count] * weights, axis=1)

    omega_energy = gather_full(omega_energy_local, int(attributes['npool']))

    hall_curve = None
    omega_grid = None
    if rank == 0:
        assert omega_energy is not None
        hall_curve = np.sum(omega_energy, axis=0) / float(attributes['nkpnts'])

        lower_index = 0
        upper_index = ene.size - 1
        for iene in range(ene.size - 1):
            if ene[iene] <= float(attributes['fermi_dw']) and ene[iene + 1] >= float(
                attributes['fermi_dw']
            ):
                lower_index = iene
            if ene[iene] <= float(attributes['fermi_up']) and ene[iene + 1] >= float(
                attributes['fermi_up']
            ):
                upper_index = iene

        omega_energy = np.reshape(
            omega_energy,
            (
                int(attributes['nk1']),
                int(attributes['nk2']),
                int(attributes['nk3']),
                ene.size,
            ),
            order='C',
        )
        omega_grid = omega_energy[:, :, :, upper_index] - omega_energy[:, :, :, lower_index]

    return ene, hall_curve, omega_grid


def _accumulate_ac_conductivity_local(
    sigma_local: np.ndarray,
    ene: np.ndarray,
    jksp: np.ndarray,
    pksp: np.ndarray,
    eigvals: np.ndarray,
    smearing: str | None,
    temp: float,
    deltakp: np.ndarray,
    deltakp2: np.ndarray | None,
    delta: float | None,
) -> None:
    """Accumulate the AC Hall contribution of one local k-point.

    Parameters
    ----------
    sigma_local : numpy.ndarray
        Local complex accumulator with shape ``(nenergy,)``.
    ene : numpy.ndarray
        Energy mesh for the AC Hall spectrum.
    jksp : numpy.ndarray
        First projected operator in the band basis with shape ``(nbands, nbands)``.
    pksp : numpy.ndarray
        Second projected operator in the band basis with shape ``(nbands, nbands)``.
    eigvals : numpy.ndarray
        Band energies for the same local k-point and spin channel.
    smearing : str | None
        Smearing mode used by the dense Hall implementation.
    temp : float
        Fermi-Dirac temperature used when ``smearing is None``.
    deltakp : numpy.ndarray
        Local adaptive widths with shape ``(nbands,)``.
    deltakp2 : numpy.ndarray | None
        Local pairwise widths with shape ``(nbands, nbands)`` when smearing is active.
    delta : float | None
        Constant Lorentzian width used by the dense no-smearing branch. This
        is ignored when a smearing model is active.

    Returns
    -------
    None
        Updates ``sigma_local`` in place.

    Notes
    -----
    This helper evaluates the Kubo-like AC Hall sum from one pair of projected
    operators at one k-point. The sparse implementation keeps only those local
    band-space matrices and computes the transition denominators in vectorized
    chunks, so it avoids caching a large dense tensor over all k-points.

    Parallelization strategy:
        Each contribution is computed independently for the local k-point. Only
        the final compact complex spectrum is reduced across ranks.
    """
    from ..smearing import intgaussian, intmetpax

    ef = 0.0
    eps = 1.0e-16

    if smearing is None:
        occupations = 1.0 / (np.exp(eigvals / temp) + 1.0)
    elif smearing == 'gauss':
        occupations = intgaussian(eigvals, ef, deltakp)
    else:
        occupations = intmetpax(eigvals, ef, deltakp)

    band_count = int(eigvals.size)
    if band_count <= 1:
        return

    off_diagonal_mask = ~np.eye(band_count, dtype=bool)
    energy_differences = ((eigvals[:, None] - eigvals[None, :]) ** 2)[off_diagonal_mask]
    transition_weights = ((occupations[:, None] - occupations[None, :]) * np.imag(pksp * jksp.T))[
        off_diagonal_mask
    ]

    if transition_weights.size == 0:
        return

    transition_chunk_size = 4096

    if smearing is not None:
        assert deltakp2 is not None
        pair_widths = deltakp2[off_diagonal_mask]
        for chunk_start in range(0, transition_weights.size, transition_chunk_size):
            chunk_stop = min(chunk_start + transition_chunk_size, transition_weights.size)
            chunk_weights = transition_weights[chunk_start:chunk_stop]
            chunk_differences = energy_differences[chunk_start:chunk_stop]
            chunk_widths = pair_widths[chunk_start:chunk_stop]
            denominator = (
                chunk_differences[None, :]
                - (ene[:, None] + 1.0j * chunk_widths[None, :]) ** 2
                + eps
            )
            sigma_local += np.sum(chunk_weights[None, :] / denominator, axis=1)
    else:
        assert delta is not None
        energy_shift = ene[:, None] + 1.0j * delta
        for chunk_start in range(0, transition_weights.size, transition_chunk_size):
            chunk_stop = min(chunk_start + transition_chunk_size, transition_weights.size)
            chunk_weights = transition_weights[chunk_start:chunk_stop]
            chunk_differences = energy_differences[chunk_start:chunk_stop]
            denominator = chunk_differences[None, :] - energy_shift**2 + eps
            sigma_local += np.sum(chunk_weights[None, :] / denominator, axis=1)


def _reduce_ac_conductivity(
    data_controller: DataController,
    ene: np.ndarray,
    sigma_local: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Reduce the distributed AC Hall spectrum to the final observable.

    Parameters
    ----------
    data_controller : DataController
        Runtime container with the global k-point count.
    ene : numpy.ndarray
        AC conductivity energy grid.
    sigma_local : numpy.ndarray
        Local complex AC Hall accumulator with shape ``(nenergy,)``.

    Returns
    -------
    tuple[numpy.ndarray | None, numpy.ndarray | None]
        Energy grid and root-only reduced conductivity spectrum.

    Notes
    -----
    After each rank has accumulated its own local spectral contribution, only
    the final one-dimensional spectrum needs to be reduced. The sparse path
    keeps exactly that communication pattern, so it does not gather any dense
    operator data.

    Parallelization strategy:
        A pair of MPI reductions sums the local real and imaginary parts of the
        spectrum. No additional communication is introduced.
    """
    arrays, attributes = data_controller.data_dicts()
    assert arrays is not None and attributes is not None

    sigma_real = np.zeros(ene.size, dtype=float) if rank == 0 else None
    sigma_imag = np.zeros(ene.size, dtype=float) if rank == 0 else None

    sigma_local_real = np.ascontiguousarray(np.real(sigma_local))
    sigma_local_imag = np.ascontiguousarray(np.imag(sigma_local))

    comm.Reduce(sigma_local_real, sigma_real, op=MPI.SUM)
    comm.Reduce(sigma_local_imag, sigma_imag, op=MPI.SUM)

    if rank == 0:
        assert sigma_real is not None and sigma_imag is not None
        sigma = (sigma_real + 1.0j * sigma_imag) / float(attributes['nkpnts'])
        return ene, np.nan_to_num(sigma)
    return None, None


def do_anomalous_Hall(data_controller: DataController, do_ac: bool) -> None:
    """Compute anomalous Hall observables from sparse derivative streams.

    Parameters
    ----------
    data_controller : DataController
        Runtime container with sparse no-bridge eigendata, adaptive widths, and
        the sparse ``H(R)`` representation.
    do_ac : bool
        If ``True``, also compute the AC Hall or magnetic circular dichroism
        spectrum.

    Returns
    -------
    None
        Writes the same Hall files as the dense implementation.

    Notes
    -----
    The anomalous Hall conductivity is built from interband matrix elements of
    Hamiltonian derivatives. In the dense workflow this often means storing a
    local dense derivative tensor before doing the band-basis contractions. The
    sparse version instead streams only the derivative directions needed for the
    chosen tensor component, projects them one k-point at a time, and
    immediately accumulates the Berry and AC contributions. This preserves the
    Hall formula while avoiding the large dense derivative storage.

    Parallelization strategy:
        Each rank consumes only its local derivative stream and reduces only the
        final compact Hall outputs. Berry and AC accumulators are updated from
        the same streamed data, so the sparse path avoids both dense caching and
        duplicate derivative passes.
    """
    from ..constants import ANGSTROM_AU, ELECTRONVOLT_SI, H_OVER_TPI, LL

    arrays, attributes = data_controller.data_dicts()
    assert arrays is not None and attributes is not None

    if attributes['dftSO'] is False:
        if rank == 0:
            print('Relativistic calculation with SO required')
            comm.Abort()
        comm.Barrier()

    if rank == 0 and attributes['verbose']:
        print('Writing bxsf files for Berry Curvature')

    a_tensor = arrays['a_tensor']
    band_count = int(arrays['E_k'].shape[1])
    deltap = float(attributes['deltaH'])

    for itensor in range(a_tensor.shape[0]):
        ipol = int(a_tensor[itensor][0])
        jpol = int(a_tensor[itensor][1])

        omega_bands = np.zeros((arrays['E_k'].shape[0], band_count), dtype=float)
        if do_ac:
            ene_ac = np.linspace(0.0, float(attributes['shift']), int(attributes['esizeH']))
            sigma_local = np.zeros(ene_ac.size, dtype=complex)
            delta_value = (
                None
                if attributes['smearing'] is not None
                else float(attributes.get('delta', attributes['deltaH']))
            )
        else:
            ene_ac = None
            sigma_local = None
            delta_value = None

        for batch_start, batch_stop, ispin, dh_batch in _iter_streamed_derivative_batches(
            data_controller,
            directions=(ipol, jpol),
        ):
            if ispin != 0:
                continue
            for batch_offset, ik_local in enumerate(range(batch_start, batch_stop)):
                vecs = arrays['v_k'][ik_local, :, :, ispin]
                degen = arrays['degen'][ispin][ik_local]
                pksp_i, pksp_j = _project_operator_pair(
                    dh_batch[batch_offset, ipol, :, :],
                    dh_batch[batch_offset, jpol, :, :],
                    vecs,
                    degen,
                )
                _accumulate_berry_curvature_local(
                    omega_bands,
                    ik_local,
                    pksp_i[:band_count, :band_count],
                    pksp_j[:band_count, :band_count],
                    arrays['E_k'][ik_local, :band_count, ispin],
                    deltap,
                )
                if do_ac:
                    assert sigma_local is not None and ene_ac is not None
                    _accumulate_ac_conductivity_local(
                        sigma_local,
                        ene_ac,
                        pksp_i[:band_count, :band_count],
                        pksp_j[:band_count, :band_count],
                        arrays['E_k'][ik_local, :band_count, ispin],
                        attributes['smearing'],
                        float(attributes['temp']),
                        arrays['deltakp'][ik_local, :band_count, ispin],
                        (
                            arrays['deltakp2'][ik_local, :band_count, :band_count, ispin]
                            if attributes['smearing'] is not None
                            else None
                        ),
                        delta_value,
                    )

        ene, ahc, omega_grid = _finalize_berry_curvature(data_controller, omega_bands)

        cgs_conv = None
        if rank == 0:
            cgs_conv = (
                1.0e8 * ANGSTROM_AU * ELECTRONVOLT_SI**2 / (H_OVER_TPI * float(attributes['omega']))
            )
            assert ahc is not None
            ahc *= cgs_conv

        cart_indices = (str(LL[ipol]), str(LL[jpol]))

        omega_grid_spin = (
            np.empty(
                (int(attributes['nk1']), int(attributes['nk2']), int(attributes['nk3']), 2),
                dtype=float,
            )
            if rank == 0
            else None
        )
        if rank == 0:
            assert omega_grid is not None and omega_grid_spin is not None
            omega_grid_spin[:, :, :, 0] = omega_grid_spin[:, :, :, 1] = omega_grid
        data_controller.write_bxsf('Berry_%s%s.bxsf' % cart_indices, omega_grid_spin, 2)

        data_controller.write_file_row_col('ahcEf_%s%s.dat' % cart_indices, ene, ahc)

        if do_ac:
            assert sigma_local is not None and ene_ac is not None
            ene_ac, sigma = _reduce_ac_conductivity(data_controller, ene_ac, sigma_local)
            if rank == 0:
                assert sigma is not None and cgs_conv is not None and ene_ac is not None
                sigma *= cgs_conv
                sigma_imag = np.imag(ene_ac * sigma / 105.4571)
                sigma_real = np.real(sigma)
            else:
                sigma_imag = None
                sigma_real = None

            data_controller.write_file_row_col('MCDi_%s%s.dat' % cart_indices, ene_ac, sigma_imag)
            data_controller.write_file_row_col('MCDr_%s%s.dat' % cart_indices, ene_ac, sigma_real)


def do_spin_Hall(
    data_controller: DataController,
    twoD: bool,
    do_ac: bool,
    projection: np.ndarray,
) -> None:
    """Compute spin Hall observables from sparse derivative streams.

    Parameters
    ----------
    data_controller : DataController
        Runtime container with sparse no-bridge eigendata, adaptive widths, and
        the sparse ``H(R)`` representation.
    twoD : bool
        If ``True``, use the two-dimensional conversion factor from the dense
        implementation.
    do_ac : bool
        If ``True``, also compute the AC spin Hall spectrum.
    projection : numpy.ndarray
        Orbital projection matrix used in the projected spin-current operator.

    Returns
    -------
    None
        Writes the same Hall files as the dense implementation.

    Notes
    -----
    The spin Hall response uses a spin-current operator together with the usual
    Hamiltonian-derivative operator. The resulting Berry and AC contractions are
    still dense in band space, but they do not require a dense derivative tensor
    for all k-points at once. This routine therefore constructs the needed
    projected operators on demand from streamed sparse derivatives and reduces
    the observables immediately.

    Parallelization strategy:
        Each rank processes only its local k-point window. Berry and AC outputs
        share the same sparse derivative stream; when ``do_ac=True`` the code
        performs only the extra local projection needed for the AC spin current,
        without replaying the sparse derivative stream.
    """
    from ..constants import ANGSTROM_AU, ELECTRONVOLT_SI, H_OVER_TPI, LL

    arrays, attributes = data_controller.data_dicts()
    assert arrays is not None and attributes is not None

    if attributes['dftSO'] is False:
        if rank == 0:
            print('Relativistic calculation with SO required')
            comm.Abort()
        comm.Barrier()

    if rank == 0 and attributes['verbose']:
        print('Writing bxsf files for Spin Berry Curvature')

    s_tensor = arrays['s_tensor']
    band_count = int(arrays['E_k'].shape[1])
    deltap = float(attributes['deltaH'])

    for itensor in range(s_tensor.shape[0]):
        ipol = int(s_tensor[itensor][0])
        jpol = int(s_tensor[itensor][1])
        spol = int(s_tensor[itensor][2])
        spin_operator = arrays['Sj'][spol]

        omega_bands = np.zeros((arrays['E_k'].shape[0], band_count), dtype=float)
        if do_ac:
            ene_ac = np.linspace(0.0, float(attributes['shift']), int(attributes['esizeH']))
            sigma_local = np.zeros(ene_ac.size, dtype=complex)
            delta_value = (
                None
                if attributes['smearing'] is not None
                else float(attributes.get('delta', attributes['deltaH']))
            )
        else:
            ene_ac = None
            sigma_local = None
            delta_value = None

        for batch_start, batch_stop, ispin, dh_batch in _iter_streamed_derivative_batches(
            data_controller,
            directions=(ipol, jpol),
        ):
            if ispin != 0:
                continue
            for batch_offset, ik_local in enumerate(range(batch_start, batch_stop)):
                vecs = arrays['v_k'][ik_local, :, :, ispin]
                degen = arrays['degen'][ispin][ik_local]
                direction_i = dh_batch[batch_offset, ipol, :, :]
                direction_j = dh_batch[batch_offset, jpol, :, :]
                spin_current = 0.5 * (spin_operator @ direction_i + direction_i @ spin_operator)
                projected_spin_current = 0.5 * (
                    projection @ spin_current + spin_current @ projection
                )
                jksp, pksp = _project_operator_pair(
                    projected_spin_current,
                    direction_j,
                    vecs,
                    degen,
                )
                _accumulate_berry_curvature_local(
                    omega_bands,
                    ik_local,
                    jksp[:band_count, :band_count],
                    pksp[:band_count, :band_count],
                    arrays['E_k'][ik_local, :band_count, ispin],
                    deltap,
                )
                if do_ac:
                    assert sigma_local is not None and ene_ac is not None
                    jksp_ac, pksp_ac = _project_operator_pair(
                        spin_current,
                        direction_j,
                        vecs,
                        degen,
                    )
                    _accumulate_ac_conductivity_local(
                        sigma_local,
                        ene_ac,
                        jksp_ac[:band_count, :band_count],
                        pksp_ac[:band_count, :band_count],
                        arrays['E_k'][ik_local, :band_count, ispin],
                        attributes['smearing'],
                        float(attributes['temp']),
                        arrays['deltakp'][ik_local, :band_count, ispin],
                        (
                            arrays['deltakp2'][ik_local, :band_count, :band_count, ispin]
                            if attributes['smearing'] is not None
                            else None
                        ),
                        delta_value,
                    )

        ene, shc, omega_grid = _finalize_berry_curvature(data_controller, omega_bands)

        cgs_conv = None
        if rank == 0:
            if twoD:
                av0 = arrays['a_vectors'][0, :]
                av1 = arrays['a_vectors'][1, :]
                cgs_conv = 1.0 / (
                    np.linalg.norm(np.cross(av0, av1)) * float(attributes['alat']) ** 2
                )
            else:
                cgs_conv = (
                    1.0e8
                    * ANGSTROM_AU
                    * ELECTRONVOLT_SI**2
                    / (H_OVER_TPI * float(attributes['omega']))
                )
            assert shc is not None
            shc *= cgs_conv

        cart_indices = (str(LL[spol]), str(LL[ipol]), str(LL[jpol]))

        omega_grid_spin = (
            np.empty(
                (int(attributes['nk1']), int(attributes['nk2']), int(attributes['nk3']), 2),
                dtype=float,
            )
            if rank == 0
            else None
        )
        if rank == 0:
            assert omega_grid is not None and omega_grid_spin is not None
            omega_grid_spin[:, :, :, 0] = omega_grid_spin[:, :, :, 1] = omega_grid
        data_controller.write_bxsf('Spin_Berry_%s_%s%s.bxsf' % cart_indices, omega_grid_spin, 2)

        data_controller.write_file_row_col('shcEf_%s_%s%s.dat' % cart_indices, ene, shc)

        if do_ac:
            assert sigma_local is not None and ene_ac is not None
            ene_ac, sigma = _reduce_ac_conductivity(data_controller, ene_ac, sigma_local)
            if rank == 0:
                assert sigma is not None and cgs_conv is not None and ene_ac is not None
                sigma *= cgs_conv
                sigma_imag = np.imag(ene_ac * sigma / 105.4571)
                sigma_real = np.real(sigma)
            else:
                sigma_imag = None
                sigma_real = None

            data_controller.write_file_row_col(
                'SCDi_%s_%s%s.dat' % cart_indices,
                ene_ac,
                sigma_imag,
            )
            data_controller.write_file_row_col(
                'SCDr_%s_%s%s.dat' % cart_indices,
                ene_ac,
                sigma_real,
            )
