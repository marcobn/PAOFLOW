#!/usr/bin/env python3
"""Verdicts, stability temperature and diagnostics of an SSCHA campaign (template: copy to <study>/analysis.py).

    python analysis.py      # -> tables/runs.csv, tables/scan.csv, tables/stability.md, tables/diagnostics.md

Reads the run directories below RUN_ROOT (read-only, through SSCHARun) and the experimental references in data/;
writes one row per run (settings, stage, Hessian verdict, softest mode on the supercell grid and on FINE_MESH), the
scan with the stability temperature (bracket, local fit) and the next proposed temperature, and the diagnostics of
every run. Playbook stages 4-6: PAOFLOW/agentic/sscha/AGENT_WORKFLOW.md.
"""

from pathlib import Path

import pandas as pd

from PAOFLOW.agentic.sscha import find_runs, next_temperature, scan_table, stability_temperature

HERE = Path(__file__).resolve().parent
RUN_ROOT = Path('<run folder>/output')  # read-only
FINE_MESH = (8, 8, 8)  # 3-4x the supercell grid
TARGET_BRACKET_K = 100.0
EXPERIMENT = {  # what the measured temperature means, with its source (None if the study has none)
    'temperature': None,
    'low': None,
    'high': None,
    'label': '<eutectoid / transition / decomposition (source)>',
}
TABLES = HERE / 'tables'


def main() -> None:
    TABLES.mkdir(exist_ok=True)
    runs = find_runs(str(RUN_ROOT))
    if not runs:
        raise SystemExit('no SSCHA runs below %s' % RUN_ROOT)

    rows = []
    for run, scan in zip(runs, scan_table(runs, fine_mesh=FINE_MESH)):
        status = run.status()
        last = status['history'][-1] if status['history'] else {}
        rows.append(dict(
            temperature_k=run.temperature, electronic_temperature_k=run.electronic_temperature,
            supercell='x'.join(map(str, run.supercell)), stage=status['stage'],
            populations=len(status['history']), last_size=last.get('size'),
            free_energy_ry=last.get('free_energy_ry'), free_energy_error_ry=last.get('free_energy_error_ry'),
            kong_liu=last.get('kong_liu'), stable=scan['stable'], softest_grid_cm1=scan['lowest_cm1'],
            softest_grid_q=scan['lowest_q'], softest_fine_cm1=scan['fine_lowest_cm1'],
            softest_fine_q=scan['fine_lowest_q'], run_dir=run.sscha_dir,
        ))  # fmt: skip
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / 'runs.csv', index=False)

    done = table.dropna(subset=['softest_fine_cm1'])
    lines = [
        '# Stability temperature',
        '',
        'Fine mesh %s; softest mode of the free-energy Hessian.' % (FINE_MESH,),
        '',
    ]
    if len(done) >= 2:
        done[['temperature_k', 'softest_grid_cm1', 'softest_fine_cm1', 'stable']].to_csv(
            TABLES / 'scan.csv', index=False
        )
        result = stability_temperature(done['temperature_k'], done['softest_fine_cm1'])
        proposal = next_temperature(
            done['temperature_k'], done['softest_fine_cm1'], TARGET_BRACKET_K
        )
        lines += ['- T_s = %.0f K (%s), bracket %s' % (result['t_stability'], 'extrapolated' if result['extrapolated']
                                                       else 'interpolated', result['bracket']),
                  '- local fit over %s K: T_s = %.0f K, slope %.2f cm^-2/K, rms %.0f cm^-2' % (
                      result['fit_temperatures'], result['t_fit'], result['slope'], result['rms']),
                  '- next temperature: %s' % ('none (bracket below %g K)' % TARGET_BRACKET_K if proposal is None
                                              else '%.0f K' % proposal)]  # fmt: skip
        if EXPERIMENT['temperature'] is not None:
            lines.append(
                '- experiment: %g K (%s)' % (EXPERIMENT['temperature'], EXPERIMENT['label'])
            )
    else:
        lines.append(
            'fewer than two temperatures with an exported Hessian: no stability temperature yet'
        )
    (TABLES / 'stability.md').write_text('\n'.join(lines) + '\n')

    report = ['# Diagnostics', '']
    for run in runs:
        report.append('## %g K (%s)' % (run.temperature, run.sscha_dir))
        findings = run.diagnostics(fine_mesh=FINE_MESH)
        report += ['- [%s] %s' % (f['level'], f['message']) for f in findings] or ['- none']
        report.append('')
    (TABLES / 'diagnostics.md').write_text('\n'.join(report))
    print(table.to_string(index=False))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
