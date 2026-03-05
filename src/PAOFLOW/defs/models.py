#
# PAOFLOW
#
# Copyright 2016-2024 - Marco BUONGIORNO NARDELLI (mbn@unt.edu)
#
# Reference:
#
# F.T. Cerasoli, A.R. Supka, A. Jayaraj, I. Siloi, M. Costa, J. Slawinska, S. Curtarolo, M. Fornari, D. Ceresoli, and M. Buongiorno Nardelli,
# Advanced modeling of materials with PAOFLOW 2.0: New features and software design, Comp. Mat. Sci. 200, 110828 (2021).
#
# M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo,R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang,
# PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on
# Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .


def Slater_Koster(data_controller, params):
    """Build a generalized Slater-Koster tight-binding model (two-center, up to 3NN).

    This routine populates the PAOFLOW data containers with a real-space
    tight-binding Hamiltonian constructed in the two-center approximation,
    up to third nearest neighbors. It uses a simple orbital basis per
    atom (s,p and d) and derives hopping matrix elements from the standard
    Slater-Koster direction-cosine expressions.

    High-level workflow:
    - Read lattice vectors and atomic positions from params and set up basic
        attributes (number of atoms, orbitals, hoppings).
    - Compute reciprocal lattice vectors and the cell volume.
    - Build a 3x3x3 supercell of atomic positions to identify neighbors and
        determine a first-neighbor cutoff.
    - Construct the Dnm matrix (orbital-position differences) for gradient
        calculations.
    - Allocate the real-space Hamiltonian array HRs and fill on-site energies
        and hopping terms using direction cosines and SK parameters.

    Data structure expectations (params):
    - params['model']['a_vectors']: 3x3 lattice vectors.
    - params['model']['atoms']: dict keyed by string indices ("0", "1", ...),
        each containing:
            - 'name': atomic species label
            - 'tau': fractional/cartesian position in lattice units
            - 'orbitals': list of orbital labels used to map on-site terms
            - on-site energies keyed by orbital label
    - params['model']['hoppings']: either a flat dict of SK parameters with keys
        'sss', 'sps', 'pps', 'ppp', etc., or a shell dict with 'nn' and optional
        'nnn'/'nnnn' blocks containing those keys.

    Output side-effects (data_controller):
    - arry['a_vectors'], arry['b_vectors'], arry['tau'], arry['atoms'],
        arry['shells'], arry['norbitals'], arry['sctau'], arry['Dnm'],
        arry['HRs']
    - attr['alat'], attr['omega'], attr['natoms'], attr['nawf'], attr['bnd'],
        attr['nbnds'], attr['nk1'], attr['nk2'], attr['nk3'], attr['nkpnts'],
        attr['nspin'], attr['dftSO'], attr['shift'], attr['cutoff']

    Notes:
    - The neighbor search uses a 3x3x3 supercell for nn only, 5x5x5 when nnn
        is enabled, and 7x7x7 when nnnn is enabled. Shells are determined from
        distinct neighbor distances and mid-point cutoffs.
    - Only the unpolarized case is supported; spin-orbit is disabled here.
    """

    import numpy as np

    arry, attr = data_controller.data_dicts()
    # Lattice Vectors
    arry["a_vectors"] = np.array(params["model"]["a_vectors"])
    attr["alat"] = 1.0

    # Atomic coordinates
    natoms = len(params["model"]["atoms"])
    tau = np.zeros((natoms, 3), dtype=float)
    for ia in range(natoms):
        tau[ia] = np.array(params["model"]["atoms"][str(ia)]["tau"])
    atoms = []
    shells = []
    for ia in range(natoms):
        atoms.append(params["model"]["atoms"][str(ia)]["name"])
        shells.append(params["model"]["atoms"][str(ia)])
    arry["tau"] = tau
    arry["atoms"] = atoms
    attr["natoms"] = natoms
    arry["shells"] = shells
    attr["nspin"] = 1  # only unpolarized case for now
    attr["dftSO"] = False  # no spin-orbit

    # Reciprocal Lattice
    arry["b_vectors"] = np.zeros((3, 3), dtype=float)
    volume = np.dot(
        np.cross(arry["a_vectors"][0, :], arry["a_vectors"][1, :]),
        arry["a_vectors"][2, :],
    )
    attr["omega"] = volume
    arry["b_vectors"][0, :] = (
        np.cross(arry["a_vectors"][1, :], arry["a_vectors"][2, :])
    ) / volume
    arry["b_vectors"][1, :] = (
        np.cross(arry["a_vectors"][2, :], arry["a_vectors"][0, :])
    ) / volume
    arry["b_vectors"][2, :] = (
        np.cross(arry["a_vectors"][0, :], arry["a_vectors"][1, :])
    ) / volume

    hoppings = params["model"]["hoppings"]
    use_third_neighbors = isinstance(hoppings, dict) and "nnnn" in hoppings
    use_second_neighbors = isinstance(hoppings, dict) and (
        "nnn" in hoppings or use_third_neighbors
    )
    if (use_second_neighbors or use_third_neighbors) and "nn" not in hoppings:
        raise ValueError(
            'Neighbor hoppings require a "nn" block in params["model"]["hoppings"].'
        )
    if use_third_neighbors and "nnn" not in hoppings:
        raise ValueError(
            'Third-neighbor hoppings require a "nnn" block in params["model"]["hoppings"].'
        )

    # dimensions of the supercell for two-center approximation
    if use_third_neighbors:
        cell_range = 3
    elif use_second_neighbors:
        cell_range = 2
    else:
        cell_range = 1
    nk1 = nk2 = nk3 = 2 * cell_range + 1
    nkpnts = nk1 * nk2 * nk3
    attr["nk1"] = nk1
    attr["nk2"] = nk2
    attr["nk3"] = nk3
    attr["nkpnts"] = nkpnts

    # on site and hopping parameters
    norbitals = np.zeros(natoms, dtype=int)
    for ia in range(natoms):
        norbitals[ia] = len(params["model"]["atoms"][str(ia)]["orbitals"])

    nawf = 0
    for ia in range(natoms):
        nawf += norbitals[ia]
    attr["nawf"] = nawf
    attr["bnd"] = nawf
    attr["nbnds"] = nawf
    attr["shift"] = 0

    # Define the Dnm matrix for correct calculation of gradient
    basis = []
    for i in range(attr["natoms"]):
        for k in range(len(arry["shells"][i]["orbitals"])):
            basis.append(arry["shells"][i]["tau"])
    arry["Dnm"] = np.empty((attr["nawf"], attr["nawf"], 3))
    for i in range(3):
        for n in range(attr["nawf"]):
            for m in range(attr["nawf"]):
                arry["Dnm"][n, m, i] = basis[n][i] - basis[m][i]

    # generate all the orbitals positions in the supercell
    sctau = np.zeros((natoms, nk1, nk2, nk3, 3), dtype=float)
    for i in range(-cell_range, cell_range + 1):
        for j in range(-cell_range, cell_range + 1):
            for k in range(-cell_range, cell_range + 1):
                for ia in range(natoms):
                    sctau[ia, i, k, j, :] = (
                        tau[ia]
                        + i * arry["a_vectors"][0]
                        + j * arry["a_vectors"][1]
                        + k * arry["a_vectors"][2]
                    )
    sctau = np.reshape(sctau, (natoms * nk1 * nk2 * nk3, 3), order="C")
    # make the list of neighbors and find cutoff for two-center approximation
    distance = lambda x, y: np.sqrt(np.sum((x - y) ** 2))
    cosines = lambda x, y: (y - x) / np.sqrt(np.sum((x - y) ** 2))
    dist = []
    for ia in range(natoms):
        for n in range(natoms * nk1 * nk2 * nk3):
            dist.append(distance(tau[ia], sctau[n]))
    unique_dist = np.sort(np.unique(dist))
    if unique_dist.size < 2:
        raise ValueError(
            "Unable to determine nearest-neighbor distances for Slater-Koster model."
        )

    dist_1 = unique_dist[1]
    if use_third_neighbors:
        if unique_dist.size < 4:
            raise ValueError(
                "Unable to determine third-neighbor distances for Slater-Koster model."
            )
        dist_2 = unique_dist[2]
        dist_3 = unique_dist[3]
        cutoff_1 = dist_1 + (dist_2 - dist_1) / 2.0
        cutoff_2 = dist_2 + (dist_3 - dist_2) / 2.0
        if unique_dist.size > 4:
            dist_4 = unique_dist[4]
            cutoff_3 = dist_3 + (dist_4 - dist_3) / 2.0
        else:
            cutoff_3 = dist_3 + (dist_3 - dist_2) / 2.0
    elif use_second_neighbors:
        if unique_dist.size < 3:
            raise ValueError(
                "Unable to determine second-neighbor distances for Slater-Koster model."
            )
        dist_2 = unique_dist[2]
        cutoff_1 = dist_1 + (dist_2 - dist_1) / 2.0
        if unique_dist.size > 3:
            dist_3 = unique_dist[3]
            cutoff_2 = dist_2 + (dist_3 - dist_2) / 2.0
        else:
            cutoff_2 = dist_2 + (dist_2 - dist_1) / 2.0
        cutoff_3 = None
    else:
        if unique_dist.size < 3:
            raise ValueError("Unable to determine cutoff for first-neighbor shell.")
        cutoff_1 = dist_1 + (unique_dist[2] - dist_1) / 2.0
        cutoff_2 = None
        cutoff_3 = None
    sctau = np.reshape(sctau, (natoms, nk1, nk2, nk3, 3), order="C")

    # debug
    arry["sctau"] = sctau
    attr["cutoff"] = cutoff_1
    attr["cutoff_1"] = cutoff_1
    if cutoff_2 is not None:
        attr["cutoff_2"] = cutoff_2
    if cutoff_3 is not None:
        attr["cutoff_3"] = cutoff_3
    arry["norbitals"] = norbitals

    HRs = np.zeros((nawf, nawf, nk1, nk2, nk3, 1), dtype=complex)

    # on-site matrix elements
    for ia in range(natoms):
        for no in range(norbitals[ia]):
            HRs[ia * norbitals[ia] + no, ia * norbitals[ia] + no, 0, 0, 0, 0] = params[
                "model"
            ]["atoms"][str(ia)][params["model"]["atoms"][str(ia)]["orbitals"][no]]

    # hopping matrix elements
    orbital_order = ("s", "px", "py", "pz", "dxy", "dyz", "dzx", "dx2-y2", "dz2")
    p_index_map = {"px": 0, "py": 1, "pz": 2}
    d_orbitals = set(orbital_order[4:])
    supported_orbitals = set(orbital_order)

    for shell in arry["shells"]:
        for orb in shell["orbitals"]:
            if orb not in supported_orbitals:
                raise ValueError(f"Unsupported orbital label: {orb}")

    if "nn" in hoppings:
        hoppings_shells = {"nn": hoppings["nn"]}
        if "nnn" in hoppings:
            hoppings_shells["nnn"] = hoppings["nnn"]
        if "nnnn" in hoppings:
            hoppings_shells["nnnn"] = hoppings["nnnn"]
    else:
        hoppings_shells = {"nn": hoppings}

    if use_second_neighbors and "nnn" not in hoppings_shells:
        raise KeyError(
            'Second-neighbor hoppings requested but no "nnn" block provided.'
        )
    if use_third_neighbors and "nnnn" not in hoppings_shells:
        raise KeyError(
            'Third-neighbor hoppings requested but no "nnnn" block provided.'
        )

    required_keys = ["sss", "sps", "pps", "ppp"]
    has_d = any(
        orb in d_orbitals for shell in arry["shells"] for orb in shell["orbitals"]
    )
    if has_d:
        required_keys.extend(["sds", "pds", "pdp", "dds", "ddp", "ddd"])
    for shell_name, shell_hoppings in hoppings_shells.items():
        missing_keys = [key for key in required_keys if key not in shell_hoppings]
        if missing_keys:
            raise KeyError(
                f"Missing Slater-Koster hopping keys for {shell_name}: {', '.join(missing_keys)}"
            )

    elements = sorted(set(arry["atoms"]))
    shell_names = ", ".join(sorted(hoppings_shells.keys()))
    print(
        f"Slater_Koster: elements={elements}, neighbor_shells={len(hoppings_shells)} ({shell_names})"
    )

    # Standard Slater-Koster normalization factors for cubic harmonics
    # (Slater & Koster, Phys. Rev. 94, 1498, 1954)
    sq3 = np.sqrt(3.0)  # √3 ≈ 1.7321
    hsq3 = sq3 / 2.0  # √3/2 ≈ 0.8660

    def _sd_value(d_orb, lx, ly, lz, shell_hoppings):
        l2 = lx * lx
        m2 = ly * ly
        n2 = lz * lz
        if d_orb == "dxy":
            return sq3 * lx * ly * shell_hoppings["sds"]
        if d_orb == "dyz":
            return sq3 * ly * lz * shell_hoppings["sds"]
        if d_orb == "dzx":
            return sq3 * lz * lx * shell_hoppings["sds"]
        if d_orb == "dx2-y2":
            return hsq3 * (l2 - m2) * shell_hoppings["sds"]
        if d_orb == "dz2":
            return (n2 - 0.5 * (l2 + m2)) * shell_hoppings["sds"]
        return None

    def _pd_value(p_orb, d_orb, lx, ly, lz, shell_hoppings):
        l2 = lx * lx
        m2 = ly * ly
        n2 = lz * lz
        if p_orb == "px":
            if d_orb == "dxy":
                return (
                    sq3 * l2 * ly * shell_hoppings["pds"]
                    + ly * (1.0 - 2.0 * l2) * shell_hoppings["pdp"]
                )
            if d_orb == "dyz":
                return (
                    sq3 * lx * ly * lz * shell_hoppings["pds"]
                    - 2.0 * lx * ly * lz * (shell_hoppings["pdp"])
                )
            if d_orb == "dzx":
                return (
                    sq3 * l2 * lz * shell_hoppings["pds"]
                    + lz * (1.0 - 2.0 * l2) * shell_hoppings["pdp"]
                )
            if d_orb == "dx2-y2":
                return (
                    hsq3 * lx * (l2 - m2) * shell_hoppings["pds"]
                    + lx * (1.0 - l2 + m2) * shell_hoppings["pdp"]
                )
            if d_orb == "dz2":
                return (
                    lx * (n2 - 0.5 * (l2 + m2)) * shell_hoppings["pds"]
                    - sq3 * lx * n2 * shell_hoppings["pdp"]
                )
        if p_orb == "py":
            if d_orb == "dxy":
                return (
                    sq3 * m2 * lx * shell_hoppings["pds"]
                    + lx * (1.0 - 2.0 * m2) * shell_hoppings["pdp"]
                )
            if d_orb == "dyz":
                return (
                    sq3 * m2 * lz * shell_hoppings["pds"]
                    + lz * (1.0 - 2.0 * m2) * shell_hoppings["pdp"]
                )
            if d_orb == "dzx":
                return (
                    sq3 * lx * ly * lz * shell_hoppings["pds"]
                    - 2.0 * lx * ly * lz * (shell_hoppings["pdp"])
                )
            if d_orb == "dx2-y2":
                return (
                    hsq3 * ly * (l2 - m2) * shell_hoppings["pds"]
                    - ly * (1.0 + l2 - m2) * shell_hoppings["pdp"]
                )
            if d_orb == "dz2":
                return (
                    ly * (n2 - 0.5 * (l2 + m2)) * shell_hoppings["pds"]
                    - sq3 * ly * n2 * shell_hoppings["pdp"]
                )
        if p_orb == "pz":
            if d_orb == "dxy":
                return (
                    sq3 * lx * ly * lz * shell_hoppings["pds"]
                    - 2.0 * lx * ly * lz * (shell_hoppings["pdp"])
                )
            if d_orb == "dyz":
                return (
                    sq3 * n2 * ly * shell_hoppings["pds"]
                    + ly * (1.0 - 2.0 * n2) * shell_hoppings["pdp"]
                )
            if d_orb == "dzx":
                return (
                    sq3 * n2 * lx * shell_hoppings["pds"]
                    + lx * (1.0 - 2.0 * n2) * shell_hoppings["pdp"]
                )
            if d_orb == "dx2-y2":
                return (
                    hsq3 * lz * (l2 - m2) * shell_hoppings["pds"]
                    - lz * (l2 - m2) * shell_hoppings["pdp"]
                )
            if d_orb == "dz2":
                return (
                    lz * (n2 - 0.5 * (l2 + m2)) * shell_hoppings["pds"]
                    + sq3 * lz * (l2 + m2) * shell_hoppings["pdp"]
                )
        return None

    def _dd_value(d_orb_a, d_orb_b, lx, ly, lz, shell_hoppings):
        l2 = lx * lx
        m2 = ly * ly
        n2 = lz * lz
        lm = lx * ly
        ln = lx * lz
        mn = ly * lz
        l2m2 = l2 * m2
        l2n2 = l2 * n2
        m2n2 = m2 * n2
        diff_lm = l2 - m2

        if d_orb_a == d_orb_b == "dxy":
            return (
                3.0 * l2m2 * shell_hoppings["dds"]
                + (l2 + m2 - 4.0 * l2m2) * shell_hoppings["ddp"]
                + (n2 + l2m2) * shell_hoppings["ddd"]
            )
        if d_orb_a == d_orb_b == "dyz":
            return (
                3.0 * m2n2 * shell_hoppings["dds"]
                + (m2 + n2 - 4.0 * m2n2) * shell_hoppings["ddp"]
                + (l2 + m2n2) * shell_hoppings["ddd"]
            )
        if d_orb_a == d_orb_b == "dzx":
            return (
                3.0 * l2n2 * shell_hoppings["dds"]
                + (l2 + n2 - 4.0 * l2n2) * shell_hoppings["ddp"]
                + (m2 + l2n2) * shell_hoppings["ddd"]
            )
        if d_orb_a == d_orb_b == "dx2-y2":
            return (
                0.75 * diff_lm**2 * shell_hoppings["dds"]
                + (l2 + m2 - diff_lm**2) * shell_hoppings["ddp"]
                + (n2 + 0.25 * diff_lm**2) * shell_hoppings["ddd"]
            )
        if d_orb_a == d_orb_b == "dz2":
            term = n2 - 0.5 * (l2 + m2)
            return (
                term**2 * shell_hoppings["dds"]
                + 3.0 * n2 * (l2 + m2) * shell_hoppings["ddp"]
                + 0.75 * (l2 + m2) ** 2 * shell_hoppings["ddd"]
            )

        if (d_orb_a, d_orb_b) in (("dxy", "dyz"), ("dyz", "dxy")):
            return (
                3.0 * lx * m2 * lz * shell_hoppings["dds"]
                + ln * (1.0 - 4.0 * m2) * shell_hoppings["ddp"]
                + ln * (m2 - 1.0) * shell_hoppings["ddd"]
            )
        if (d_orb_a, d_orb_b) in (("dxy", "dzx"), ("dzx", "dxy")):
            return (
                3.0 * l2 * ly * lz * shell_hoppings["dds"]
                + mn * (1.0 - 4.0 * l2) * shell_hoppings["ddp"]
                + mn * (l2 - 1.0) * shell_hoppings["ddd"]
            )
        if (d_orb_a, d_orb_b) in (("dyz", "dzx"), ("dzx", "dyz")):
            return (
                3.0 * ly * n2 * lx * shell_hoppings["dds"]
                + lm * (1.0 - 4.0 * n2) * shell_hoppings["ddp"]
                + lm * (n2 - 1.0) * shell_hoppings["ddd"]
            )

        if (d_orb_a, d_orb_b) in (("dxy", "dx2-y2"), ("dx2-y2", "dxy")):
            return (
                1.5 * lm * diff_lm * shell_hoppings["dds"]
                + 2.0 * lm * (m2 - l2) * shell_hoppings["ddp"]
                + 0.5 * lm * diff_lm * shell_hoppings["ddd"]
            )
        if (d_orb_a, d_orb_b) in (("dyz", "dx2-y2"), ("dx2-y2", "dyz")):
            return (
                1.5 * mn * diff_lm * shell_hoppings["dds"]
                - mn * (1.0 + 2.0 * diff_lm) * shell_hoppings["ddp"]
                + mn * (1.0 + 0.5 * diff_lm) * shell_hoppings["ddd"]
            )
        if (d_orb_a, d_orb_b) in (("dzx", "dx2-y2"), ("dx2-y2", "dzx")):
            return (
                1.5 * ln * diff_lm * shell_hoppings["dds"]
                + ln * (1.0 - 2.0 * diff_lm) * shell_hoppings["ddp"]
                - ln * (1.0 - 0.5 * diff_lm) * shell_hoppings["ddd"]
            )

        if (d_orb_a, d_orb_b) in (("dxy", "dz2"), ("dz2", "dxy")):
            return sq3 * (
                lm * (n2 - 0.5 * (l2 + m2)) * shell_hoppings["dds"]
                - 2.0 * lm * n2 * shell_hoppings["ddp"]
                + 0.5 * lm * (1.0 + n2) * shell_hoppings["ddd"]
            )
        if (d_orb_a, d_orb_b) in (("dyz", "dz2"), ("dz2", "dyz")):
            return sq3 * (
                mn * (n2 - 0.5 * (l2 + m2)) * shell_hoppings["dds"]
                + mn * (l2 + m2 - n2) * shell_hoppings["ddp"]
                - 0.5 * mn * (l2 + m2) * shell_hoppings["ddd"]
            )
        if (d_orb_a, d_orb_b) in (("dzx", "dz2"), ("dz2", "dzx")):
            return sq3 * (
                ln * (n2 - 0.5 * (l2 + m2)) * shell_hoppings["dds"]
                + ln * (l2 + m2 - n2) * shell_hoppings["ddp"]
                - 0.5 * ln * (l2 + m2) * shell_hoppings["ddd"]
            )
        if (d_orb_a, d_orb_b) in (("dx2-y2", "dz2"), ("dz2", "dx2-y2")):
            return sq3 * (
                0.5 * diff_lm * (n2 - 0.5 * (l2 + m2)) * shell_hoppings["dds"]
                + n2 * (m2 - l2) * shell_hoppings["ddp"]
                + 0.25 * (1.0 + n2) * diff_lm * shell_hoppings["ddd"]
            )

        return None

    def _sk_sp_value(orb_a, orb_b, lx, ly, lz, shell_hoppings):
        if orb_a == "s" and orb_b in d_orbitals:
            return _sd_value(orb_b, lx, ly, lz, shell_hoppings)
        if orb_b == "s" and orb_a in d_orbitals:
            return _sd_value(orb_a, lx, ly, lz, shell_hoppings)
        if orb_a in p_index_map and orb_b in d_orbitals:
            return _pd_value(orb_a, orb_b, lx, ly, lz, shell_hoppings)
        if orb_b in p_index_map and orb_a in d_orbitals:
            value = _pd_value(orb_b, orb_a, lx, ly, lz, shell_hoppings)
            return -value if value is not None else None
        if orb_a in d_orbitals and orb_b in d_orbitals:
            return _dd_value(orb_a, orb_b, lx, ly, lz, shell_hoppings)

        if orb_a == "s" and orb_b == "s":
            return shell_hoppings["sss"]

        if orb_a == "s" and orb_b in p_index_map:
            return (lx, ly, lz)[p_index_map[orb_b]] * shell_hoppings["sps"]

        if orb_b == "s" and orb_a in p_index_map:
            return -(lx, ly, lz)[p_index_map[orb_a]] * shell_hoppings["sps"]

        if orb_a == orb_b and orb_a in p_index_map:
            ll = (lx, ly, lz)[p_index_map[orb_a]]
            return ll**2 * shell_hoppings["pps"] + (1.0 - ll**2) * shell_hoppings["ppp"]

        if (orb_a, orb_b) in (("px", "py"), ("py", "px")):
            return lx * ly * (shell_hoppings["pps"] - shell_hoppings["ppp"])

        if (orb_a, orb_b) in (("py", "pz"), ("pz", "py")):
            return ly * lz * (shell_hoppings["pps"] - shell_hoppings["ppp"])

        if (orb_a, orb_b) in (("px", "pz"), ("pz", "px")):
            return lx * lz * (shell_hoppings["pps"] - shell_hoppings["ppp"])

        return None

    def _validate_sk_tables():
        if not has_d:
            return

        test_dirs = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
            np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0),
        ]

        for vec in test_dirs:
            lx, ly, lz = vec
            for a in d_orbitals:
                for b in d_orbitals:
                    v_ab = _dd_value(a, b, lx, ly, lz, hoppings_shells["nn"])
                    v_ba = _dd_value(b, a, lx, ly, lz, hoppings_shells["nn"])
                    if v_ab is None or v_ba is None:
                        continue
                    if not np.isclose(v_ab, v_ba, atol=1e-12, rtol=1e-12):
                        raise ValueError(
                            "Inconsistent d-d SK symmetry for "
                            f"{a},{b} at direction ({lx:.3f},{ly:.3f},{lz:.3f})"
                        )

    if params.get("model", {}).get("validate_sk", False):
        _validate_sk_tables()

    for i in range(-cell_range, cell_range + 1):
        for j in range(-cell_range, cell_range + 1):
            for k in range(-cell_range, cell_range + 1):
                for ia in range(natoms):
                    for ib in range(natoms):
                        dist_val = distance(tau[ia], sctau[ib, i, j, k, :])
                        if dist_val <= 0:
                            continue
                        if dist_val < cutoff_1:
                            shell_key = "nn"
                        elif cutoff_2 is not None and dist_val < cutoff_2:
                            shell_key = "nnn"
                        elif cutoff_3 is not None and dist_val < cutoff_3:
                            shell_key = "nnnn"
                        else:
                            continue

                        lx = cosines(tau[ia], sctau[ib, i, j, k, :])[0]
                        ly = cosines(tau[ia], sctau[ib, i, j, k, :])[1]
                        lz = cosines(tau[ia], sctau[ib, i, j, k, :])[2]

                        shell_hoppings = hoppings_shells[shell_key]
                        orbitals_a = arry["shells"][ia]["orbitals"]
                        orbitals_b = arry["shells"][ib]["orbitals"]
                        for noa, orb_a in enumerate(orbitals_a):
                            for nob, orb_b in enumerate(orbitals_b):
                                value = _sk_sp_value(
                                    orb_a, orb_b, lx, ly, lz, shell_hoppings
                                )
                                if value is not None:
                                    HRs[
                                        ia * norbitals[ia] + noa,
                                        ib * norbitals[ib] + nob,
                                        i,
                                        j,
                                        k,
                                        0,
                                    ] = value

        arry["HRs"] = HRs


def graphene(data_controller, params):
    import numpy as np

    from .constants import ANGSTROM_AU

    arry, attr = data_controller.data_dicts()

    attr["nk1"] = 3
    attr["nk2"] = 3
    attr["nk3"] = 1

    attr["bnd"] = 4
    attr["shift"] = 100.0
    attr["Efermi"] = 0.0
    attr["nspin"] = 1
    attr["nawf"] = 2
    attr["nspin"] = 1
    attr["natoms"] = 2

    attr["alat"] = 2.46 * ANGSTROM_AU

    arry["HRs"] = np.zeros(
        (
            attr["nawf"],
            attr["nawf"],
            attr["nk1"],
            attr["nk2"],
            attr["nk3"],
            attr["nspin"],
        ),
        dtype=complex,
    )

    # H00
    arry["HRs"][0, 1, 0, 0, 0, 0] = params["t"]
    arry["HRs"][1, 0, 0, 0, 0, 0] = params["t"]

    # H10
    arry["HRs"][1, 0, 1, 0, 0, 0] = params["t"]

    # H20
    arry["HRs"][:, :, 2, 0, 0, 0] = np.conj(arry["HRs"][:, :, 1, 0, 0, 0]).T

    # H01
    arry["HRs"][1, 0, 0, 1, 0, 0] = params["t"]

    # H02
    arry["HRs"][:, :, 0, 2, 0, 0] = np.conj(arry["HRs"][:, :, 0, 1, 0, 0]).T

    # Lattice Vectors
    arry["a_vectors"] = np.zeros((3, 3), dtype=float)
    arry["a_vectors"] = np.array([[1.0, 0, 0], [0.5, 3**0.5 / 2, 0], [0, 0, 10]])
    arry["a_vectors"] = arry["a_vectors"]

    # Atomic coordinates
    arry["tau"] = np.zeros((2, 3), dtype=float)

    arry["tau"][0, 0] = 0.50000
    arry["tau"][0, 1] = 0.28867
    arry["tau"][1, 0] = 1.00000
    arry["tau"][1, 1] = 0.57735

    # Reciprocal Lattice
    arry["b_vectors"] = np.zeros((3, 3), dtype=float)
    volume = np.dot(
        np.cross(arry["a_vectors"][0, :], arry["a_vectors"][1, :]),
        arry["a_vectors"][2, :],
    )
    arry["b_vectors"][0, :] = (
        np.cross(arry["a_vectors"][1, :], arry["a_vectors"][2, :])
    ) / volume
    arry["b_vectors"][1, :] = (
        np.cross(arry["a_vectors"][2, :], arry["a_vectors"][0, :])
    ) / volume
    arry["b_vectors"][2, :] = (
        np.cross(arry["a_vectors"][0, :], arry["a_vectors"][1, :])
    ) / volume

    arry["atoms"] = ["C", "C"]


def graphene2(data_controller, params):
    import numpy as np

    from .constants import ANGSTROM_AU

    arry, attr = data_controller.data_dicts()

    attr["nk1"] = 3
    attr["nk2"] = 3
    attr["nk3"] = 1

    attr["nawf"] = 2
    attr["nspin"] = 1
    attr["natoms"] = 2

    arry["naw"] = np.array([1, 1])

    attr["alat"] = 2.46 * ANGSTROM_AU

    arry["HRs"] = np.zeros(
        (
            attr["nawf"],
            attr["nawf"],
            attr["nk1"],
            attr["nk2"],
            attr["nk3"],
            attr["nspin"],
        ),
        dtype=complex,
    )

    # H00
    arry["HRs"][0, 0, 0, 0, 0, 0] = params["delta"] / 2
    arry["HRs"][1, 1, 0, 0, 0, 0] = -params["delta"] / 2

    # H00
    arry["HRs"][0, 1, 0, 0, 0, 0] = params["t"]
    arry["HRs"][1, 0, 0, 0, 0, 0] = params["t"]

    # H10
    arry["HRs"][1, 0, 1, 0, 0, 0] = params["t"]

    # H20
    arry["HRs"][:, :, 2, 0, 0, 0] = np.conj(arry["HRs"][:, :, 1, 0, 0, 0]).T

    # H01
    arry["HRs"][1, 0, 0, 1, 0, 0] = params["t"]

    # H02
    arry["HRs"][:, :, 0, 2, 0, 0] = np.conj(arry["HRs"][:, :, 0, 1, 0, 0]).T

    # Lattice Vectors
    arry["a_vectors"] = np.zeros((3, 3), dtype=float)
    arry["a_vectors"] = np.array([[1.0, 0, 0], [0.5, 3**0.5 / 2, 0], [0, 0, 10]])
    arry["a_vectors"] = arry["a_vectors"]

    # Atomic coordinates
    arry["tau"] = np.zeros((2, 3), dtype=float)

    arry["tau"][0, 0] = 0.50000
    arry["tau"][0, 1] = 0.28867
    arry["tau"][1, 0] = 1.00000
    arry["tau"][1, 1] = 0.57735

    # Reciprocal Lattice
    arry["b_vectors"] = np.zeros((3, 3), dtype=float)
    volume = np.dot(
        np.cross(arry["a_vectors"][0, :], arry["a_vectors"][1, :]),
        arry["a_vectors"][2, :],
    )
    arry["b_vectors"][0, :] = (
        np.cross(arry["a_vectors"][1, :], arry["a_vectors"][2, :])
    ) / volume
    arry["b_vectors"][1, :] = (
        np.cross(arry["a_vectors"][2, :], arry["a_vectors"][0, :])
    ) / volume
    arry["b_vectors"][2, :] = (
        np.cross(arry["a_vectors"][0, :], arry["a_vectors"][1, :])
    ) / volume

    arry["atoms"] = ["C", "C"]


def cubium(data_controller, params):
    import numpy as np

    from .constants import ANGSTROM_AU

    arry, attr = data_controller.data_dicts()

    attr["nk1"] = 3
    attr["nk2"] = 3
    attr["nk3"] = 3
    attr["Efermi"] = 6 * params["t"]
    attr["nawf"] = 1
    attr["nspin"] = 1
    attr["natoms"] = 1
    attr["bnd"] = 1
    attr["shift"] = 0
    attr["dftSO"] = False
    attr["nkpnts"] = attr["nk1"] * attr["nk2"] * attr["nk3"]
    attr["nbnds"] = 1
    attr["nelec"] = 2

    attr["alat"] = 1.0 * ANGSTROM_AU
    attr["omega"] = attr["alat"] ** 3

    arry["HRs"] = np.zeros(
        (
            attr["nawf"],
            attr["nawf"],
            attr["nk1"],
            attr["nk2"],
            attr["nk3"],
            attr["nspin"],
        ),
        dtype=complex,
    )

    # H000
    arry["HRs"][0, 0, 0, 0, 0, 0] = 0.0 - attr["Efermi"]

    # H100
    arry["HRs"][0, 0, 1, 0, 0, 0] = params["t"]

    # H200
    arry["HRs"][:, :, 2, 0, 0, 0] = np.conj(arry["HRs"][:, :, 1, 0, 0, 0]).T

    # H010
    arry["HRs"][0, 0, 0, 1, 0, 0] = params["t"]

    # H020
    arry["HRs"][:, :, 0, 2, 0, 0] = np.conj(arry["HRs"][:, :, 0, 1, 0, 0]).T

    # H001
    arry["HRs"][0, 0, 0, 0, 1, 0] = params["t"]

    # H002
    arry["HRs"][:, :, 0, 0, 2, 0] = np.conj(arry["HRs"][:, :, 0, 0, 1, 0]).T

    # Lattice Vectors
    arry["a_vectors"] = np.zeros((3, 3), dtype=float)
    arry["a_vectors"] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # Atomic coordinates
    arry["tau"] = np.zeros((1, 3), dtype=float)

    # Reciprocal Lattice
    arry["b_vectors"] = np.zeros((3, 3), dtype=float)
    volume = np.dot(
        np.cross(arry["a_vectors"][0, :], arry["a_vectors"][1, :]),
        arry["a_vectors"][2, :],
    )
    arry["b_vectors"][0, :] = (
        np.cross(arry["a_vectors"][1, :], arry["a_vectors"][2, :])
    ) / volume
    arry["b_vectors"][1, :] = (
        np.cross(arry["a_vectors"][2, :], arry["a_vectors"][0, :])
    ) / volume
    arry["b_vectors"][2, :] = (
        np.cross(arry["a_vectors"][0, :], arry["a_vectors"][1, :])
    ) / volume

    arry["atoms"] = ["Cu"]


def cubium2(data_controller, params):
    import numpy as np

    from .constants import ANGSTROM_AU

    arry, attr = data_controller.data_dicts()

    attr["nk1"] = 3
    attr["nk2"] = 3
    attr["nk3"] = 3

    attr["nawf"] = 2
    attr["nspin"] = 1
    attr["natoms"] = 1
    attr["bnd"] = 2
    attr["shift"] = 0
    attr["dftSO"] = False
    attr["nkpnts"] = attr["nk1"] * attr["nk2"] * attr["nk3"]
    attr["nbnds"] = 2
    attr["nelec"] = 2
    attr["alat"] = 1.0 * ANGSTROM_AU
    attr["omega"] = attr["alat"] ** 3

    arry["HRs"] = np.zeros(
        (
            attr["nawf"],
            attr["nawf"],
            attr["nk1"],
            attr["nk2"],
            attr["nk3"],
            attr["nspin"],
        ),
        dtype=complex,
    )

    # H000
    arry["HRs"][0, 0, 0, 0, 0, 0] = -params["Eg"] / 2 - 6.0 * params["t"]
    arry["HRs"][1, 1, 0, 0, 0, 0] = params["Eg"] / 2 + 6.0 * params["t"]

    # H100
    arry["HRs"][0, 0, 1, 0, 0, 0] = params["t"]
    arry["HRs"][1, 1, 1, 0, 0, 0] = -params["t"]

    # H200
    arry["HRs"][:, :, 2, 0, 0, 0] = np.conj(arry["HRs"][:, :, 1, 0, 0, 0]).T

    # H010
    arry["HRs"][0, 0, 0, 1, 0, 0] = params["t"]
    arry["HRs"][1, 1, 0, 1, 0, 0] = -params["t"]

    # H020
    arry["HRs"][:, :, 0, 2, 0, 0] = np.conj(arry["HRs"][:, :, 0, 1, 0, 0]).T

    # H001
    arry["HRs"][0, 0, 0, 0, 1, 0] = params["t"]
    arry["HRs"][1, 1, 0, 0, 1, 0] = -params["t"]

    # H002
    arry["HRs"][:, :, 0, 0, 2, 0] = np.conj(arry["HRs"][:, :, 0, 0, 1, 0]).T

    # Lattice Vectors
    arry["a_vectors"] = np.zeros((3, 3), dtype=float)
    arry["a_vectors"] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # Atomic coordinates
    arry["tau"] = np.zeros((1, 3), dtype=float)

    # Reciprocal Lattice
    arry["b_vectors"] = np.zeros((3, 3), dtype=float)
    volume = np.dot(
        np.cross(arry["a_vectors"][0, :], arry["a_vectors"][1, :]),
        arry["a_vectors"][2, :],
    )
    arry["b_vectors"][0, :] = (
        np.cross(arry["a_vectors"][1, :], arry["a_vectors"][2, :])
    ) / volume
    arry["b_vectors"][1, :] = (
        np.cross(arry["a_vectors"][2, :], arry["a_vectors"][0, :])
    ) / volume
    arry["b_vectors"][2, :] = (
        np.cross(arry["a_vectors"][0, :], arry["a_vectors"][1, :])
    ) / volume

    arry["atoms"] = ["Cu"]


def Kane_Mele(data_controller, params):
    import numpy as np

    from .constants import ANGSTROM_AU

    arry, attr = data_controller.data_dicts()

    attr["nk1"] = 3
    attr["nk2"] = 3
    attr["nk3"] = 1

    attr["nawf"] = 4
    attr["bnd"] = 4
    attr["shift"] = 100.0
    attr["Efermi"] = 0.0
    attr["nspin"] = 1
    attr["natoms"] = 2

    arry["naw"] = [2, 2]

    if "alat" not in params:
        alat = 1.0
    else:
        alat = params["alat"]

    attr["alat"] = alat * ANGSTROM_AU

    t = params["t"]
    soc_par = params["soc_par"]
    r_par = params["r_par"]
    v_par = params["v_par"]

    arry["HRs"] = np.zeros(
        (
            attr["nawf"],
            attr["nawf"],
            attr["nk1"],
            attr["nk2"],
            attr["nk3"],
            attr["nspin"],
        ),
        dtype=complex,
    )

    # H00
    arry["HRs"][0, 0, 0, 0, 0, 0] = t * v_par
    arry["HRs"][1, 1, 0, 0, 0, 0] = t * v_par
    arry["HRs"][2, 2, 0, 0, 0, 0] = -t * v_par
    arry["HRs"][3, 3, 0, 0, 0, 0] = -t * v_par

    # H00
    arry["HRs"][0, 2, 0, 0, 0, 0] = t
    arry["HRs"][1, 3, 0, 0, 0, 0] = t
    arry["HRs"][2, 0, 0, 0, 0, 0] = t
    arry["HRs"][3, 1, 0, 0, 0, 0] = t

    # H10
    arry["HRs"][2, 0, 1, 0, 0, 0] = t
    arry["HRs"][3, 1, 1, 0, 0, 0] = t

    arry["HRs"][0, 0, 1, 0, 0, 0] = -complex(0.0, soc_par)
    arry["HRs"][1, 1, 1, 0, 0, 0] = complex(0.0, soc_par)
    arry["HRs"][2, 2, 1, 0, 0, 0] = complex(0.0, soc_par)
    arry["HRs"][3, 3, 1, 0, 0, 0] = -complex(0.0, soc_par)

    ##H20
    # arry['HRs'][:,:,2,0,0,0] = np.conj(arry['HRs'][:,:,1,0,0,0]).T

    # H01
    arry["HRs"][2, 0, 0, 1, 0, 0] = t
    arry["HRs"][3, 1, 0, 1, 0, 0] = t

    arry["HRs"][0, 0, 0, 1, 0, 0] = complex(0.0, soc_par)
    arry["HRs"][1, 1, 0, 1, 0, 0] = -complex(0.0, soc_par)
    arry["HRs"][2, 2, 0, 1, 0, 0] = -complex(0.0, soc_par)
    arry["HRs"][3, 3, 0, 1, 0, 0] = complex(0.0, soc_par)

    ##H02
    # arry['HRs'][:,:,0,2,0,0] = np.conj(arry['HRs'][:,:,0,1,0,0]).T

    # H21
    arry["HRs"][0, 0, 2, 1, 0, 0] = -complex(0.0, soc_par)
    arry["HRs"][1, 1, 2, 1, 0, 0] = complex(0.0, soc_par)
    arry["HRs"][2, 2, 2, 1, 0, 0] = complex(0.0, soc_par)
    arry["HRs"][3, 3, 2, 1, 0, 0] = -complex(0.0, soc_par)

    ##H12
    ##arry['HRs'][:,:,1,2,0,0] = np.conj(arry['HRs'][:,:,2,1,0,0]).T

    r3h = np.sqrt(3.0) / 2.0

    arry["HRs"][0, 3, 0, 0, 0, 0] += r_par * complex(
        -r3h, 0.5
    )  # 1j * r_par * (0.5 * 1 - r3h * -1j)
    arry["HRs"][1, 2, 0, 0, 0, 0] += r_par * complex(
        r3h, 0.5
    )  # 1j * r_par * (0.5 * 1 - r3h * 1j)
    arry["HRs"][3, 0, 0, 0, 0, 0] += r_par * complex(-r3h, -0.5)
    arry["HRs"][2, 1, 0, 0, 0, 0] += r_par * complex(r3h, -0.5)

    arry["HRs"][0, 3, 1, 0, 0, 0] += -r_par * complex(
        r3h, 0.5
    )  # -1j * r_par * (0.5 * 1 + r3h * -1j)
    arry["HRs"][1, 2, 1, 0, 0, 0] += -r_par * complex(
        -r3h, 0.5
    )  # -1j * r_par * (0.5 * 1 + r3h * 1j)

    arry["HRs"][0, 3, 0, 1, 0, 0] += complex(0.0, r_par)  # -1j * r_par * -1 * 1
    arry["HRs"][1, 2, 0, 1, 0, 0] += complex(0.0, r_par)  # -1j * r_par * -1 * 1

    # H02
    arry["HRs"][:, :, 0, 2, 0, 0] = np.conj(arry["HRs"][:, :, 0, 1, 0, 0]).T
    # H20
    arry["HRs"][:, :, 2, 0, 0, 0] = np.conj(arry["HRs"][:, :, 1, 0, 0, 0]).T
    # H12
    arry["HRs"][:, :, 1, 2, 0, 0] = np.conj(arry["HRs"][:, :, 2, 1, 0, 0]).T

    # Lattice Vectors
    arry["a_vectors"] = np.zeros((3, 3), dtype=float)
    arry["a_vectors"] = np.array([[1.0, 0, 0], [0.5, 3**0.5 / 2, 0], [0, 0, 10]])
    arry["a_vectors"] = arry["a_vectors"]

    # Spin properties
    arry["Sj"] = np.zeros((3, 4, 4), dtype=complex)

    arry["Sj"][2, 0, 0] = 0.5
    arry["Sj"][2, 1, 1] = -0.5
    arry["Sj"][2, 2, 2] = 0.5
    arry["Sj"][2, 3, 3] = -0.5

    arry["Sj"][0, 0, 1] = 0.5
    arry["Sj"][0, 1, 0] = 0.5
    arry["Sj"][0, 2, 3] = 0.5
    arry["Sj"][0, 3, 2] = 0.5

    arry["Sj"][1, 0, 1] = -complex(0.0, 0.5)
    arry["Sj"][1, 1, 0] = complex(0.0, 0.5)
    arry["Sj"][1, 2, 3] = -complex(0.0, 0.5)
    arry["Sj"][1, 3, 2] = complex(0.0, 0.5)
    # Atomic coordinates
    arry["tau"] = np.zeros((2, 3), dtype=float)
    arry["tau"][0] = np.dot([1 / 3, 1 / 3, 0.0], arry["a_vectors"])
    arry["tau"][1] = np.dot([2 / 3, 2 / 3, 0.0], arry["a_vectors"])

    # Reciprocal Lattice
    arry["b_vectors"] = np.zeros((3, 3), dtype=float)
    volume = np.dot(
        np.cross(arry["a_vectors"][0, :], arry["a_vectors"][1, :]),
        arry["a_vectors"][2, :],
    )
    arry["b_vectors"][0, :] = (
        np.cross(arry["a_vectors"][1, :], arry["a_vectors"][2, :])
    ) / volume
    arry["b_vectors"][1, :] = (
        np.cross(arry["a_vectors"][2, :], arry["a_vectors"][0, :])
    ) / volume
    arry["b_vectors"][2, :] = (
        np.cross(arry["a_vectors"][0, :], arry["a_vectors"][1, :])
    ) / volume

    attr["omega"] = alat**3 * arry["a_vectors"][0, :].dot(
        np.cross(arry["a_vectors"][1, :], arry["a_vectors"][2, :])
    )

    arry["species"] = ["KM", "KM"]


def build_TB_model(data_controller, parameters):
    if parameters["label"].upper() == "GRAPHENE":
        graphene(data_controller, parameters)
    elif parameters["label"].upper() == "GRAPHENE2":
        graphene2(data_controller, parameters)
    elif parameters["label"].upper() == "CUBIUM":
        cubium(data_controller, parameters)
    elif parameters["label"].upper() == "CUBIUM2":
        cubium2(data_controller, parameters)
    elif parameters["label"].upper() == "KANE_MELE":
        Kane_Mele(data_controller, parameters)
    elif parameters["label"].upper() == "SLATER_KOSTER":
        Slater_Koster(data_controller, parameters)
    else:
        print(f'ERROR: Label "{parameters["label"]}" not found in builtin models.')
