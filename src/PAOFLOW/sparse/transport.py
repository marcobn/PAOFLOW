"""Sparse transport glue.

Boltzmann transport in PAOFLOW is already a purely band-diagonal computation:
:func:`PAOFLOW.boltzmann.do_transport.do_transport` consumes only ``E_k``, the
band velocities ``velkp``, and ``deltakp`` — never an ``(nawf, nawf)`` matrix.
The sparse path therefore feeds its selected-band ``E_k``/``velkp``/``deltakp``
straight into the existing kernel, after verifying the transport window lies
inside the eigensolver window.
"""

import numpy as np


def run_transport(
    data_controller,
    tmin,
    tmax,
    nt,
    emin,
    emax,
    scattering_channels,
    scattering_weights,
    tau_dict,
    do_hall,
    write_to_file,
    save_tensors,
):
    """Compute transport tensors from the sparse selected-band spectrum.

    Parameters mirror :meth:`PAOFLOW.PAOFLOW.PAOFLOW.transport`.

    Raises
    ------
    ValueError
        If the requested transport window ``[emin, emax]`` extends beyond the
        band window computed by ``pao_eigh`` (which would silently truncate the
        spectrum and give wrong tensors).
    """
    from ..boltzmann.do_transport import do_transport

    arry, attr = data_controller.data_dicts()

    # Guard: the transport window must be covered by the computed spectrum.
    e_top = float(np.min(arry['E_k'][:, -1, :]))
    if emax > e_top + 1e-6:
        raise ValueError(
            f'Sparse transport: requested emax={emax} eV exceeds the highest '
            f'computed band ({e_top:.3f} eV at the worst k-point). Recompute '
            'pao_eigh with a wider window so transport is not truncated.'
        )

    bnd = attr['bnd']
    ene = np.linspace(emin, emax, attr.get('transport_ne', 500))
    temps = np.linspace(tmin, tmax, nt)

    if 'tau_dict' not in attr:
        attr['tau_dict'] = tau_dict

    velkp = np.ascontiguousarray(arry['velkp'][:, :, :bnd, :])

    do_transport(
        data_controller,
        temps,
        ene,
        velkp,
        scattering_channels,
        scattering_weights,
        do_hall,
        write_to_file,
        save_tensors,
    )
