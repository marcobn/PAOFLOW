"""Debug-only serialization of transport intermediates.

These writers emit human-inspectable artifacts (the legacy ``.ham`` file,
projectabilities, k-space overlaps) that are not consumed by the transport
calculation itself. They run only when debug output is requested.

All inputs are read from the shared ``DataController`` store, which is the
single source of truth; the caller does not derive or pass them in.
"""

import os
from pathlib import Path

import numpy as np

from PAOFLOW.DataController import DataController
from PAOFLOW.transport.io.log_module import log_rank0
from PAOFLOW.transport.utils.converters import cartesian_to_crystal


def write_internal_format_files(
    output_dir: str,
    output_prefix: str,
    data_controller: DataController,
    do_overlap_transformation: bool,
) -> None:
    """
    Write Hamiltonian and optional overlap matrices in a format that matches the legacy IOTK-style .ham file structure.

    This is a debug-only serialization of the real-space blocks produced by
    :func:`populate_real_space_hamiltonian`, which must have been called first to
    populate ``HRs``/``SRs`` in the shared data store.

    The output includes:
    - Dimensional and symmetry metadata in a DATA tag
    - Real-space and reciprocal lattice vectors
    - K-point list and weights
    - R-vectors and their weights
    - Hamiltonian matrix blocks (VR.#)
    - Overlap matrix blocks (OVERLAP.#), if enabled

    Parameters
    ----------
    `output_dir` : str
        Directory to write the ``.ham`` file into (created if missing).
    `output_prefix` : str
        Prefix for the output file (e.g., 'al5_bulk' → 'al5_bulk.ham').
    `data_controller` : DataController
        Shared data store. The k-space Hamiltonian/overlap blocks, R-vectors,
        k-points, lattice vectors, and real-space ``HRs``/``SRs`` are all read
        from here.
    `do_overlap_transformation` : bool
        If True and overlap matrices are present, overlap blocks will be written to the output.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ham_file = output_prefix if output_prefix.endswith('.ham') else output_prefix + '.ham'
    arry, attr = data_controller.data_dicts()

    Hk = arry['Hk']
    Sk = arry.get('Sk')
    ivr = arry['ivr']
    wr = arry['wr']
    nk = arry['nk']
    nr = arry['nr']

    avec = arry['a_vectors'] * attr['alat']
    bvec = arry['b_vectors'] * (2.0 * np.pi / attr['alat'])
    kpts = arry['kpnts'].T
    vkpts_crystal = cartesian_to_crystal(arry['vkpts_cartesian'], bvec)
    wk = arry['wk']

    spin_component = 'all'
    shift = np.zeros(3, dtype=float)  # No shift in k-point grid for crystal coordinates
    nspin, _, dim, _ = Hk.shape
    nkpnts = kpts.shape[1]
    nrtot = ivr.shape[0]
    have_overlap = Sk is not None and do_overlap_transformation
    fermi_energy = 0.0

    Hr = arry['HRs']
    Sr = arry['SRs'] if have_overlap else None

    with open(ham_file, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<?iotk version="1.2.0"?>\n')
        f.write('<?iotk file_version="1.0"?>\n')
        f.write('<?iotk binary="F"?>\n')
        f.write('<Root>\n')
        f.write('  <HAMILTONIAN>\n')

        f.write(
            f'    <DATA dimwann="{dim}" nkpnts="{nkpnts}" nspin="{nspin}" spin_component="{spin_component}" '
        )
        f.write(
            f'nk="{nk[0]} {nk[1]} {nk[2]}" shift="{shift}" nrtot="{nrtot}" nr="{nr[0]} {nr[1]} {nr[2]}" '
        )
        f.write(f'have_overlap="{"T" if have_overlap else "F"}"\n')
        f.write(f'fermi_energy="{fermi_energy:.15E}"/>\n')

        f.write('    <DIRECT_LATTICE type="real" size="9" columns="3" units="bohr">\n')
        for row in avec.T:
            f.write(' ' + '  '.join(f'{x:.15E}' for x in row) + '\n')
        f.write('    </DIRECT_LATTICE>\n')

        f.write('    <RECIPROCAL_LATTICE type="real" size="9" columns="3" units="bohr^-1">\n')
        for row in bvec.T:
            f.write(' ' + '  '.join(f'{x:.15E}' for x in row) + '\n')
        f.write('    </RECIPROCAL_LATTICE>\n')

        f.write(f'    <VKPT type="real" size="{3 * nkpnts}" columns="3" units="crystal">\n')
        for i in range(vkpts_crystal.shape[1]):
            f.write(' ' + '  '.join(f'{vkpts_crystal[j, i]:.15E}' for j in range(3)) + '\n')
        f.write('    </VKPT>\n')

        f.write(f'    <WK type="real" size="{nkpnts}">\n')
        for w in wk:
            f.write(f' {w:.15E}\n')
        f.write('    </WK>\n')
        f.write(f'    <IVR type="integer" size="{3 * nrtot}" columns="3" units="crystal">\n')
        for row in ivr:
            f.write(' {:10d}{:10d}{:10d} \n'.format(*row))
        f.write('    </IVR>\n')
        f.write(f'    <WR type="real" size="{nrtot}">\n')
        for w in wr:
            f.write(f' {w:.15E}\n')
        f.write('    </WR>\n')
        f.write('    <RHAM>\n')
        for ir in range(nrtot):
            tag = f'VR.{ir + 1}'
            f.write(f'      <{tag} type="complex" size="{dim * dim}">\n')
            for z in Hr[ir].flatten():
                f.write(f' {z.real:> .15E},{z.imag:> .15E}\n')
            f.write(f'      </{tag}>\n')

            if have_overlap:
                tag = f'OVERLAP.{ir + 1}'
                f.write(f'      <{tag} type="complex" size="{dim * dim}">\n')
                for z in Sr[ir].flatten():
                    f.write(f' {z.real:> .15E},{z.imag:> .15E}\n')
                f.write(f'      </{tag}>\n')
        f.write('    </RHAM>\n')
        write_kham(Hk, f)

        f.write('  </HAMILTONIAN>\n')
        f.write('</Root>\n')


def write_kham(
    Hk: np.ndarray,
    f: object,
    spin_component: str = 'all',
    tag: str = 'KHAM',
    block_prefix: str = 'KH',
) -> None:
    """
    Write Hk to an IOTK-style XML file.

    Parameters
    ----------
    `Hk` : (nspin, nkpnts, dim, dim) complex ndarray
        Hamiltonian matrices in k-space.
    `output_file` : Path
        Destination XML file.
    `spin_component` : str
        One of: "all", "up", "down".
    `tag` : str
        Name of the XML block (default: "KHAM").
    `block_prefix` : str
        Prefix for matrix block tags (default: "KH" → <KH.1>, <KH.2>, ...)
    """
    f.write('  <HAMILTONIAN>\n')
    nspin, nkpnts, _, _ = Hk.shape

    for isp in range(nspin):
        if spin_component == 'up' and isp == 1:
            continue
        if spin_component == 'down' and isp == 0:
            continue

        if spin_component == 'all' and nspin == 2:
            f.write(f'    <SPIN.{isp + 1}>\n')

        f.write(f'      <{tag}>\n')
        for ik in range(nkpnts):
            tagname = f'{block_prefix}.{ik + 1}'
            mat = Hk[isp, ik]
            dim = mat.shape[0]
            f.write(f'        <{tagname} type="complex" size="{dim * dim}">\n')
            for i in range(dim):
                for j in range(dim):
                    z = mat[i, j]
                    f.write(f' {z.real: .15E},{z.imag: .15E}\n')
            f.write(f'        </{tagname}>\n')
        f.write(f'      </{tag}>\n')

        if spin_component == 'all' and nspin == 2:
            f.write(f'    </SPIN.{isp + 1}>\n')

    f.write('  </HAMILTONIAN>\n')


def write_projectability_files(output_dir: str, data_controller: DataController) -> None:
    arry, attr = data_controller.data_dicts()
    Hk = arry['Hk']
    proj = arry['U'].swapaxes(0, 1)
    eigvals = arry['my_eigsmat']
    nbnds = attr['nbnds']
    nspin, nkpnts, _, _ = Hk.shape

    for isp in range(nspin):
        proj_file = (
            os.path.join(output_dir, f'projectability_{["up", "dn"][isp]}.txt')
            if nspin == 2
            else os.path.join(output_dir, 'projectability.txt')
        )
        with open(proj_file, 'w') as f:
            f.write('# Energy (eV)        Projectability\n')
            for ik in range(nkpnts):
                for ib in range(nbnds):
                    proj_vec = proj[:, ib, ik, isp]
                    weight = np.vdot(proj_vec, proj_vec).real
                    energy = eigvals[ib, ik, isp]
                    f.write(f'{energy:20.13f}  {weight:20.13f}\n')
    log_rank0('Printed projectabilities to projectability.txt')


def write_overlap_files(
    output_dir: str, data_controller: DataController, do_overlap_transformation: bool
) -> None:
    if not do_overlap_transformation:
        return
    arry, _ = data_controller.data_dicts()
    Sk = arry.get('Sk')
    if Sk is None:
        return
    nR = Sk.shape[2]
    nawf = Sk.shape[0]
    kovp_file = os.path.join(output_dir, 'kovp.txt')
    with open(kovp_file, 'w') as f:
        f.write('# Overlap Real        Overlap Imag\n')
        for ik in range(nR):
            mat = Sk[:, :, ik]
            for i in range(nawf):
                for j in range(nawf):
                    f.write(f'{mat[i, j].real:20.13f}  {mat[i, j].imag:20.13f}\n')
    log_rank0('Printed overlap matrices to kovp.txt')
