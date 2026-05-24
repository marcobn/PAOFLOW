"""Run a bare (no Hubbard) QE scf + bands calculation for GaAs.

Uses a separate prefix ``GaAs_bare`` and outdir ``bare/`` so the
converged eACBN0 ``GaAs.save`` is untouched.  Produces
``GaAs_bare.bands.dat.gnu`` for the superimposed plot.
"""
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
QE = '/Users/marco/Local/Programs/qe-7.4.1/bin/'
MPI = 'mpirun -np 8'
PREFIX_BARE = 'GaAs_bare'
OUTDIR = './bare/'

(HERE / 'bare').mkdir(exist_ok=True)


def _strip_hubbard(text: str) -> str:
    """Drop the HUBBARD card (and everything after the header until EOF
    or the next blank-line + ALLCAPS card)."""
    out, skip = [], False
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith('HUBBARD'):
            skip = True
            continue
        if skip:
            # Stop skipping once we hit a blank line followed by another
            # card header, or EOF.  Our inputs end with the HUBBARD
            # block, so just skip to the end.
            continue
        out.append(line)
    return '\n'.join(out) + '\n'


def _rewrite(src: str, dst: str, replacements: dict[str, str]) -> None:
    txt = (HERE / src).read_text()
    txt = _strip_hubbard(txt)
    for k, v in replacements.items():
        txt = txt.replace(k, v)
    (HERE / dst).write_text(txt)


def _run(cmd, stdin, stdout):
    print(f'>>> {cmd} < {stdin} > {stdout}', flush=True)
    with open(HERE / stdin, 'rb') as fin, open(HERE / stdout, 'w') as fout:
        subprocess.run(cmd.split(), stdin=fin, stdout=fout,
                       stderr=subprocess.STDOUT, check=True, cwd=HERE)


# ---- build bare inputs (clone of GaAs.* with HUBBARD stripped) ----
# scf
_rewrite(
    'GaAs.scf.in',
    'bare_scf.in',
    {
        "prefix = 'GaAs'": f"prefix = '{PREFIX_BARE}'\n  outdir = '{OUTDIR}'",
    },
)
# bands
_rewrite(
    'GaAs.bands.in',
    'bare_bands.in',
    {
        "prefix = 'GaAs'": f"prefix = '{PREFIX_BARE}'\n  outdir = '{OUTDIR}'",
    },
)
# bands.x post-processing
bandsx = (HERE / 'GaAs.bandsx.in').read_text()
bandsx = bandsx.replace("prefix = 'GaAs'",
                        f"prefix = '{PREFIX_BARE}'")
bandsx = bandsx.replace("outdir = './'", f"outdir = '{OUTDIR}'")
bandsx = bandsx.replace("filband = 'GaAs.bands.dat'",
                        f"filband = '{PREFIX_BARE}.bands.dat'")
(HERE / 'bare_bandsx.in').write_text(bandsx)


# ---- run pw.x (scf), pw.x (bands), bands.x ----
_run(f'{MPI} {QE}pw.x', 'bare_scf.in', 'bare_scf.out')
_run(f'{MPI} {QE}pw.x', 'bare_bands.in', 'bare_bands.out')
_run(f'{MPI} {QE}bands.x', 'bare_bandsx.in', 'bare_bandsx.out')

print(f'\nDone. Wrote {PREFIX_BARE}.bands.dat[.gnu]')
