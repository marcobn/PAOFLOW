"""Weak experimental peaks: magnetic breakdown or frequency mixing? (collaborator package)

numpy + standard library only; fixed random seeds.
A. weak peaks without a direct orbit vs breakdown-network orbits (if data/network_orbits.csv exists)
B. weak peaks as harmonics/combinations (2F_i, F_i+F_j, |F_i-F_j|) of significant peaks, same angle
C. for combinations with both parents assigned to calculated groups: do the groups touch in k-space?
   (meta.json "contact_groups", from the breakdown gap map)
Writes weak_peak_classification.csv.  Usage: python weak_peak_tests.py
"""

import csv

import numpy as np

import plot_figures as pf

T, META = pf.T, pf.META
CONTACT = {frozenset(p) for p in META.get('contact_groups', [])}


def direct_group(c, a, f):
    m = (c['copies'] >= T['direct_min_copies']) & (np.abs(c['angle'] - a) <= T['angle_win'])
    if not m.any():
        return np.inf, ''
    i = np.flatnonzero(m)[np.argmin(np.abs(c['F'][m] - f) / f)]
    d = abs(c['F'][i] - f) / f
    return d, (c['group'][i] if d <= T['direct_tol'] else '')


def best_combo(e, sig, a, f):
    idx = np.flatnonzero(sig & (e['angle'] == a))
    best = None
    for k, i in enumerate(idx):
        cands = (
            [(2 * e['F'][i], (i, i), '2x')]
            + [(e['F'][i] + e['F'][j], (i, j), '+') for j in idx[k + 1 :]]
            + [(abs(e['F'][i] - e['F'][j]), (i, j), '-') for j in idx[k + 1 :]]
        )
        for v, par, op in cands:
            d = abs(v - f) / f
            if v > 15 and d <= T['combo_tol'] and (best is None or d < best[0]):
                best = (d, par, op)
    return best


def chance(fn, angles, freqs, all_angles, seed, trials=300):
    rng = np.random.default_rng(seed)
    sh = [
        np.mean([fn(a, f) for a, f in zip(rng.choice(all_angles, len(freqs)), freqs)])
        for _ in range(trials)
    ]
    lu = [
        np.mean(
            [
                fn(a, f)
                for a, f in zip(angles, np.exp(rng.uniform(np.log(15), np.log(8000), len(freqs))))
            ]
        )
        for _ in range(trials)
    ]
    return np.mean(sh), np.std(sh), np.mean(lu), np.std(lu)


def main():
    e, c, n, _ = pf.load()
    sig, weak = e['mag'] >= T['sig_mag'], e['mag'] < T['sig_mag']
    angles = np.unique(e['angle'])
    dirs = [direct_group(c, a, f) for a, f in zip(e['angle'], e['F'])]
    explained = np.array([d <= T['direct_tol'] for d, _ in dirs])
    un = weak & ~explained
    if n is not None:
        print('A. weak peaks without a direct orbit vs network orbits (3 %, same angle)')
        for name, m in (
            ('robust', (n['copies'] >= T['net_min_copies']) & (n['R'] >= T['net_min_R'])),
            ('open-sheet, any', n['open'] & (n['copies'] >= 3)),
            ('all', n['copies'] >= 3),
        ):

            def fn(a, f, m=m):
                s = m & (n['angle'] == a)
                return bool(s.any() and np.min(np.abs(n['F'][s] - f) / f) <= T['match_tol'])

            obs = np.mean([fn(a, f) for a, f in zip(e['angle'][un], e['F'][un])])
            ms, ss, ml, sl = chance(fn, e['angle'][un], e['F'][un], angles, 7)
            print(
                f'   {name:16s} {100 * obs:4.0f} %  | chance angle-shuffled {100 * ms:.0f}+/-{100 * ss:.0f} %, '
                f'log-uniform {100 * ml:.0f}+/-{100 * sl:.0f} %'
            )
    print('B. weak peaks = 2F_i, F_i+F_j or |F_i-F_j| of significant peaks at the same angle')

    def fn(a, f):
        return best_combo(e, sig, a, f) is not None

    hits = np.array([fn(a, f) for a, f in zip(e['angle'][weak], e['F'][weak])])
    ms, ss, ml, sl = chance(fn, e['angle'][weak], e['F'][weak], angles, 11)
    print(
        f'   {hits.sum()}/{len(hits)} ({100 * hits.mean():.0f} %)  | chance angle-shuffled {100 * ms:.0f}+/-'
        f'{100 * ss:.0f} %, log-uniform {100 * ml:.0f}+/-{100 * sl:.0f} %'
    )
    rows, rel = [], {'contact': 0, 'same group': 0, 'no contact': 0}
    for i in np.flatnonzero(weak):
        b = best_combo(e, sig, e['angle'][i], e['F'][i])
        expr = par = verdict = ''
        if b is not None:
            _, (p, q), op = b
            gp, gq = dirs[p][1], dirs[q][1]
            expr = f'2 x {e["F"][p]:.0f}' if op == '2x' else f'{e["F"][p]:.0f} {op} {e["F"][q]:.0f}'
            par = f'{gp or "?"}/{gq or "?"}'
            if op != '2x' and gp and gq:
                verdict = (
                    'same group'
                    if gp == gq
                    else ('contact' if frozenset((gp, gq)) in CONTACT else 'no contact')
                )
                rel[verdict] += 1
        rows.append(
            dict(
                angle_deg=e['angle'][i],
                freq_T=e['F'][i],
                rel_mag=e['mag'][i],
                direct_dev_pct=round(100 * dirs[i][0], 1) if np.isfinite(dirs[i][0]) else '',
                direct_group=dirs[i][1],
                combination=expr,
                parent_groups=par,
                parent_relation=verdict,
            )
        )
    print(f"C. combinations with both parents assigned: {rel} (breakdown requires 'contact')")
    with open(pf.HERE / 'weak_peak_classification.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print('written weak_peak_classification.csv')


if __name__ == '__main__':
    main()
