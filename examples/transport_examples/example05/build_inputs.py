"""Generate the Quantum ESPRESSO inputs for the Bi2Se3(0001) surface spectrum.

Bi2Se3 is R-3m (#166). Its *primitive* cell is the 5-atom rhombohedral one, but
that cell cannot be used here: the PAOFLOW transport partition stacks principal
layers along the third lattice vector and treats a1/a2 as the in-plane
periodicity, so a1 and a2 must span the surface plane and a3 must be the surface
normal. In the rhombohedral setting all three vectors carry the same z
component, so none of that holds.

The 15-atom *hexagonal* conventional cell is therefore the smallest cell
compatible with a (0001) surface: a1/a2 lie in the surface plane, a3 = c is the
surface normal, and one cell is exactly three quintuple layers (QLs).

Two structural points that this script exists to make auditable:

1.  The R-centring translations (0,0,0), (2/3,1/3,1/3), (1/3,2/3,2/3) applied to
    the Wyckoff sites Se(3a) and Bi/Se(6c) generate all 15 atoms.
2.  The origin is then shifted so that the *cell boundary falls in the van der
    Waals gap*. This is not cosmetic. The NEGF surface Green's function
    terminates the semi-infinite stack at the cell boundary, so an unshifted
    cell (whose z=0 plane sits at the centre of a QL) would model a surface
    created by cleaving *through* a quintuple layer -- dangling bonds, no
    topological Dirac cone. After the shift the stack terminates at the vdW gap,
    which is how Bi2Se3 actually cleaves.

Structural parameters are the room-temperature refinement of Nakajima,
J. Phys. Chem. Solids 24, 479 (1963), the same values the WannierTools Bi2Se3
example uses.

Usage
-----
    python build_inputs.py

Rewrites ``scf.in`` and ``nscf.in`` in place and prints the resulting layer
stacking so the termination can be checked by eye.
"""

from __future__ import annotations

import numpy as np

# --- Structure (hexagonal setting, angstrom) --------------------------------
A_HEX = 4.138
C_HEX = 28.64
# Wyckoff 6c free parameters. Se(1) sits on 3a and has no free parameter.
Z_BI = 0.4008
Z_SE2 = 0.2117

# --- Calculation parameters -------------------------------------------------
PREFIX = 'bi2se3'
OUTDIR = './output/qe'
PSEUDO_DIR = '../../../PSEUDOS/nc-fr-04_pbe_standard'
PSEUDOS = {'Bi': 'Bi.upf', 'Se': 'Se.upf'}

# PseudoDojo nc-fr-04 "standard": Bi carries 5d10 6s2 6p3 (15 e), Se carries
# 3d10 4s2 4p4 (16 e), so the cell has 6*15 + 9*16 = 234 electrons. These are
# hard norm-conserving potentials; 80 Ry sits above the high-accuracy hint for
# both elements. Run a convergence test before quoting numbers.
ECUTWFC = 80.0
ECUTRHO = 320.0

# With spin-orbit every band holds one electron, so 234 bands are occupied. The
# PAO basis has 18 spinor orbitals per atom (d(3/2,5/2) + s(1/2) + p(1/2,3/2)),
# i.e. nawf = 15*18 = 270; nbnd must cover that manifold for the projection to
# be complete.
NBND = 300

# Small smearing keeps a fermi_energy in the XML (PAOFLOW references all
# energies to it) without touching the occupations of a ~0.3 eV-gap insulator.
DEGAUSS = 0.005

# SCF mesh: symmetry-reduced, so it can afford to be denser than the nscf mesh.
K_SCF = (9, 9, 3)

# NSCF mesh: this one sets the PAO real-space range, and symmetry must be off.
#
#   In-plane (6 x 6)  -> transverse R-vectors span -2..3, i.e. ~3a ~ 12 A. This
#     is the knob to turn for a smoother in-plane dispersion; 9x9 or 12x12 cost
#     (N/6)^2 more in the nscf step and nothing at all in the NEGF step.
#
#   Along z (3)       -> DO NOT CHANGE. The transport code builds H_00 from
#     R_z = 0 and the principal-layer coupling H_01 from R_z = 1, and silently
#     discards everything beyond. With nk_z = 3 the R_z grid is exactly
#     {-1, 0, 1}, so the truncation throws nothing away and the principal-layer
#     approximation is exact for this Hamiltonian. nk_z = 5 would generate
#     R_z = +/-2 blocks that are then dropped, an uncontrolled error.
K_NSCF = (6, 6, 3)


def bi2se3_positions() -> list[tuple[str, np.ndarray]]:
    """Return the 15 (element, crystal coordinate) pairs of the hexagonal cell.

    Coordinates are shifted so that z = 0 / z = 1 lies in the van der Waals gap.
    """
    centering = [(0.0, 0.0, 0.0), (2 / 3, 1 / 3, 1 / 3), (1 / 3, 2 / 3, 2 / 3)]
    # Se(1) on 3a, Bi and Se(2) on 6c = {(0,0,z), (0,0,-z)}.
    generators = [('Se', 0.0), ('Bi', Z_BI), ('Bi', -Z_BI), ('Se', Z_SE2), ('Se', -Z_SE2)]

    atoms = [
        (element, np.array([cx, cy, (z + cz) % 1.0]))
        for element, z in generators
        for cx, cy, cz in centering
    ]
    atoms.sort(key=lambda item: item[1][2])

    # Locate the widest interlayer spacing -- the vdW gap -- and move it onto the
    # cell boundary. Comparing against the wrap-around spacing keeps this correct
    # no matter where the gap starts out.
    z = np.array([position[2] for _, position in atoms])
    spacings = np.append(np.diff(z), 1.0 + z[0] - z[-1])
    widest = int(np.argmax(spacings))
    z_cut = z[widest] + 0.5 * spacings[widest]

    shifted = [(element, np.array([p[0], p[1], (p[2] - z_cut) % 1.0])) for element, p in atoms]
    shifted.sort(key=lambda item: item[1][2])
    return shifted


def _cards(positions: list[tuple[str, np.ndarray]]) -> str:
    lines = ['ATOMIC_SPECIES']
    for element, pseudo in PSEUDOS.items():
        lines.append(f' {element}  0.0  {pseudo}')
    lines.append('ATOMIC_POSITIONS crystal')
    for element, (x, y, z) in positions:
        lines.append(f' {element}  {x:.10f}  {y:.10f}  {z:.10f}')
    return '\n'.join(lines)


def _input_file(calculation: str, kmesh: tuple[int, int, int], positions) -> str:
    extra_system = ''
    if calculation == 'nscf':
        extra_system = f'    nbnd = {NBND}\n    nosym = .true.\n    noinv = .true.\n'
    extra_control = "    wf_collect = .true.\n" if calculation == 'nscf' else '    tprnfor = .true.\n'

    return f""" &control
    calculation = '{calculation}'
    restart_mode = 'from_scratch'
    prefix = '{PREFIX}'
    pseudo_dir = '{PSEUDO_DIR}'
    outdir = '{OUTDIR}'
    verbosity = 'high'
{extra_control} /
 &system
    ibrav = 4
    A = {A_HEX}
    C = {C_HEX}
    nat = {len(positions)}, ntyp = {len(PSEUDOS)}
    ecutwfc = {ECUTWFC}
    ecutrho = {ECUTRHO}
    noncolin = .true.
    lspinorb = .true.
    occupations = 'smearing', smearing = 'marzari-vanderbilt', degauss = {DEGAUSS}
{extra_system} /
 &electrons
    mixing_mode = 'plain'
    mixing_beta = 0.3
    conv_thr = 1.0d-8
    electron_maxstep = 200
 /
{_cards(positions)}
K_POINTS (automatic)
  {kmesh[0]} {kmesh[1]} {kmesh[2]}  0 0 0
"""


def main() -> None:
    positions = bi2se3_positions()

    with open('scf.in', 'w') as handle:
        handle.write(_input_file('scf', K_SCF, positions))
    with open('nscf.in', 'w') as handle:
        handle.write(_input_file('nscf', K_NSCF, positions))

    print(f'wrote scf.in ({K_SCF[0]}x{K_SCF[1]}x{K_SCF[2]}) and '
          f'nscf.in ({K_NSCF[0]}x{K_NSCF[1]}x{K_NSCF[2]}, nosym)')
    print()
    print('layer stacking along z (Se-Bi-Se-Bi-Se = one quintuple layer):')
    z = np.array([position[2] for _, position in positions])
    spacings = np.append(np.diff(z), 1.0 + z[0] - z[-1]) * C_HEX
    for index, (element, position) in enumerate(positions):
        marker = '   <-- van der Waals gap' if spacings[index] > 2.0 else ''
        boundary = ' | cell boundary' if index == len(positions) - 1 else ''
        print(f'  {element:2s}  z = {position[2]:.6f}  ({position[2] * C_HEX:6.3f} A)'
              f'   d = {spacings[index]:5.3f} A{marker}{boundary}')


if __name__ == '__main__':
    main()
