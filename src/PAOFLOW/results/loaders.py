"""Loaders of the results layer: files of a :class:`DatasetRef` to a :class:`Dataset`.

One function per family.  Loaders read with :mod:`PAOFLOW.results.columns` (and the existing
PAOFLOW readers where they apply), attach units from the catalogue, and expose the choices a
figure offers (temperature, tensor component, grouping, column group) as ``Dataset.options``;
the chosen values come back in ``options`` when the figure is rebuilt.  Colour roles follow the
entity: spin up / down, tensor components and kernel levels keep their slot, temperatures take
the sequential ramp.
"""

from __future__ import annotations

import json
import os
import re
from collections import OrderedDict
from typing import Any, Callable

import numpy as np

from .catalogue import KINDS
from .columns import header_parameters, read_blocks, read_columns
from .model import (
    CATEGORICAL,
    MAX_ORDINAL,
    Dataset,
    DatasetRef,
    Kind,
    Series,
    axis_title,
    ordinal_roles,
    spread,
)

KIND_BY_NAME = {kind.name: kind for kind in KINDS}
MAX_SERIES = 8
HARTREE_MEV = 27211.386245988
SPIN_LABELS = {'0': 'spin ↑', '1': 'spin ↓'}
COMPONENT_ORDER = ('xx', 'yy', 'zz', 'xy', 'xz', 'yz', 'yx', 'zx', 'zy')
TENSOR_COLUMNS = ('xx', 'yy', 'zz', 'xy', 'xz', 'yz')


def _path(root: str, relative: str) -> str:
    return os.path.join(root, relative)


def _choice(options: dict, name: str, label: str, choices: list, default: Any) -> Any:
    """The selected value of an option (the default when missing or not among the choices)."""
    values = [c[0] if isinstance(c, (list, tuple)) else c for c in choices]
    selected = options.get(name, default)
    if selected not in values:
        selected = default
    return selected


def _option(dataset: Dataset, name: str, label: str, choices: list, value: Any) -> None:
    dataset.options[name] = {
        'label': label,
        'choices': [list(c) if isinstance(c, tuple) else c for c in choices],
        'value': value,
    }


def _labels_roles(ref: DatasetRef) -> list[str]:
    """Categorical roles in the fixed order of the dataset's labels (stable per entity)."""
    ordered = sorted(set(ref.labels), key=_label_order)
    slot = {label: CATEGORICAL[i % len(CATEGORICAL)] for i, label in enumerate(ordered)}
    return [slot[label] for label in ref.labels]


def _label_order(label: str) -> tuple:
    for index, component in enumerate(COMPONENT_ORDER):
        if label == component or label.startswith(component + ' '):
            return (0, index, label)
    if label in ('spin ↑', 'DFT', 'DOS'):
        return (0, -1, label)
    return (1, 0, label)


def _kind(ref: DatasetRef) -> Kind:
    return KIND_BY_NAME[ref.kind]


def _axes(kind: Kind, ref: DatasetRef) -> tuple[str, str]:
    y_label, y_unit = kind.y
    if kind.describe is not None:
        _title, y_label, y_unit = kind.describe(ref.group)
    return axis_title(*kind.x), axis_title(y_label, y_unit)


def _new(ref: DatasetRef, **kw) -> Dataset:
    kind = _kind(ref)
    x_label, y_label = _axes(kind, ref)
    values = dict(x_label=x_label, y_label=y_label, zero=kind.zero)
    values.update(kw)
    return Dataset(ref=ref, title=ref.title, family=kind.family, **values)


# --------------------------------------------------------------------------------- k paths
def read_kpath(folder: str, count: int | None = None) -> tuple[list[tuple[float, str]], bool]:
    """Labelled ticks of ``kpath_points.txt`` in ``folder``.

    Two layouts exist: the dense ``label npoints`` lines (ticks at cumulative k indices, the
    layout of :func:`PAOFLOW.inputs.read_pao_output.read_band_path_PAO`) and the sparse-backend
    ``position label`` lines (ticks at path distances).

    Parameters
    ----------
    folder : str
        Folder of the band file.
    count : int, optional
        Number of k points of the band file (ticks beyond it are dropped).

    Returns
    -------
    ticks : list of (float, str)
    by_distance : bool
        Whether the ticks are distances (sparse layout).
    """
    path = os.path.join(folder, 'kpath_points.txt')
    if not os.path.isfile(path):
        return [], False
    entries = []
    with open(path, 'r', encoding='utf-8', errors='replace') as handle:
        for line in handle:
            tokens = line.split()
            if not tokens:
                break
            entries.append(tokens)
    if not entries:
        return [], False
    pretty = lambda tag: 'Γ' if tag in ('G', 'gG', 'Gamma', r'$\Gamma$') else tag  # noqa: E731
    try:
        float(entries[0][0])
        by_distance = True
    except ValueError:
        by_distance = False
    if by_distance:
        return [(float(t[0]), pretty(t[1] if len(t) > 1 else '')) for t in entries], True
    ticks: list[tuple[float, str]] = []
    position = 0
    for index, tokens in enumerate(entries):
        tag = pretty(tokens[0])
        npoints = int(tokens[1]) if len(tokens) > 1 else 0
        if ticks and ticks[-1][0] == position and index > 0:
            ticks[-1] = (position, ticks[-1][1] + '|' + tag)
        else:
            ticks.append((position, tag))
        position += npoints
    if count is not None:
        ticks = [t for t in ticks if t[0] <= count - 1 + 1e-9]
    return ticks, False


# ------------------------------------------------------------------------------- families
def load_curve(ref: DatasetRef, root: str, options: dict) -> Dataset:
    dataset = _new(ref)
    roles = _labels_roles(ref)
    for file, label, role in zip(ref.files, ref.labels, roles):
        data, _header = read_columns(_path(root, file))
        if data.shape[0] == 0 or data.shape[1] < 2:
            dataset.notes.append('%s: no numeric data' % os.path.basename(file))
            continue
        dataset.series.append(
            Series(label or ref.title, data[:, 0], data[:, 1], role=role, meta={'file': file})
        )
    if ref.kind == 'berry_curvature':
        dataset.ticks, _ = read_kpath(_path(root, ref.folder))
    if ref.kind == 'projectability':
        for series in dataset.series:
            series.mode = 'markers'
    return dataset


def _column_names(kind: Kind, header: list[str], count: int) -> list[str]:
    if kind.column_names and len(kind.column_names) == count:
        return list(kind.column_names)
    for line in header:
        match = re.search(r'=\s*([-+\d.eE\s]+)$', line)
        if match:
            values = match.group(1).split()
            if len(values) == count:
                symbol = 'η' if 'eta' in line.lower() else ''
                return ['%s = %s' % (symbol, v) if symbol else v for v in values]
        tokens = [t for t in re.split(r'\s{2,}|\t|,\s*', line.strip()) if t]
        if len(tokens) == count + 1:
            return tokens[1:]
    return ['column %d' % (i + 1) for i in range(count)]


def _x_from_header(kind: Kind, header: list[str]) -> str | None:
    """Axis title of the first column from the header (``# frequency(cm-1)  total_dos``)."""
    if kind.x[0] and kind.x[1]:
        return None
    for line in header:
        token = re.split(r'\s{2,}|,\s', line.strip())[0] if line.strip() else ''
        if token:
            match = re.match(r'^\s*([A-Za-z_|]+[^()\[]*?)\s*[(\[]([^)\]]+)[)\]]', token)
            if match:
                return axis_title(kind.x[0] or match.group(1).strip(), match.group(2).strip())
            return kind.x[0] or token
    return None


def load_columns(ref: DatasetRef, root: str, options: dict) -> Dataset:
    kind = _kind(ref)
    dataset = _new(ref)
    file = ref.files[0]
    data, header = read_columns(_path(root, file))
    if data.shape[0] == 0:
        dataset.text = _text(_path(root, file))
        dataset.family = 'text'
        return dataset
    names = _column_names(kind, header, data.shape[1] - 1)
    x_title = _x_from_header(kind, header)
    if x_title:
        dataset.x_label = x_title
    columns = list(range(1, data.shape[1]))
    if kind.columns:
        groups = [
            (title, unit, tuple(c for c in cols if c < data.shape[1]))
            for title, unit, cols in kind.columns
        ]
        groups = [g for g in groups if g[2]]
        titles = [g[0] for g in groups]
        chosen = _choice(options, 'quantity', 'Quantity', titles, titles[0])
        _option(dataset, 'quantity', 'Quantity', titles, chosen)
        title, unit, columns = next(g for g in groups if g[0] == chosen)
        dataset.y_label = axis_title(title, unit)
        columns = list(columns)
    elif not kind.y[0]:
        dataset.y_label = names[0] if len(names) == 1 else 'Value'
    if len(columns) > MAX_SERIES:
        chosen = _choice(options, 'column', 'Column', names, names[0])
        _option(dataset, 'column', 'Column', names, chosen)
        columns = [names.index(chosen) + 1]
        dataset.notes.append('%d columns: pick one above.' % len(names))
    for index, column in enumerate(columns):
        dataset.series.append(
            Series(
                names[column - 1],
                data[:, 0],
                data[:, column],
                role=CATEGORICAL[index % 8],
                meta={'file': file},
            )
        )
    if header:
        dataset.notes.append(' · '.join(header)[:400])
    if ref.kind == 'max_eigenvalue':
        dataset.notes.append('Tc where the largest eigenvalue crosses 1.')
    return dataset


def load_temperature_columns(ref: DatasetRef, root: str, options: dict) -> Dataset:
    dataset = _new(ref)
    tables = []
    for file, label in zip(ref.files, ref.labels):
        data, header = read_columns(_path(root, file))
        if data.shape[0]:
            tables.append(
                (
                    float(ref.group.get('temperature') or re.findall(r'[\d.]+', label)[0]),
                    file,
                    data,
                    header,
                )
            )
    if not tables:
        dataset.notes.append('no numeric data')
        return dataset
    tables.sort(key=lambda t: t[0])
    shown = _shown_temperatures(dataset, options, [t[0] for t in tables])
    tables = [t for t in tables if t[0] in shown]
    count = tables[0][2].shape[1] - 1
    names = _column_names(_kind(ref), tables[0][3], count)
    if count > 1:
        default = names[min(len(names) - 1, len(names) // 2)] if 'η' in names[0] else names[0]
        chosen = _choice(options, 'column', 'Column', names, default)
        _option(dataset, 'column', 'Column', names, chosen)
        column = names.index(chosen) + 1
    else:
        column = 1
    roles = ordinal_roles(len(tables))
    for (temperature, file, data, _header), role in zip(tables, roles):
        if data.shape[1] > column:
            dataset.series.append(
                Series(
                    '%g K' % temperature,
                    data[:, 0],
                    data[:, column],
                    role=role,
                    meta={'file': file},
                )
            )
    if ref.kind == 'phonon_assisted_absorption':
        dataset.options['log_y'] = {
            'label': 'Log scale',
            'choices': [True, False],
            'value': options.get('log_y', True),
        }
    return dataset


def load_bands(ref: DatasetRef, root: str, options: dict) -> Dataset:
    dataset = _new(ref)
    roles = _labels_roles(ref)
    count = None
    by_distance = False
    for file, label, role in zip(ref.files, ref.labels, roles):
        data, _header = read_columns(_path(root, file))
        if data.shape[0] == 0:
            continue
        first = data[:, 0]
        integer_index = np.allclose(first, np.round(first)) and np.all(np.diff(first) == 1)
        x = first if integer_index or data.shape[1] > 1 else np.arange(len(first))
        by_distance = by_distance or not integer_index
        bands = data[:, 1:] if data.shape[1] > 1 else data
        count = len(x)
        dash = 'dash' if label in ('spin ↓',) else 'solid'
        dataset.series.append(
            Series(label or 'bands', x, bands, role=role, dash=dash, meta={'file': file})
        )
    ticks, distance_ticks = read_kpath(_path(root, ref.folder), count)
    if ticks and distance_ticks == by_distance:
        dataset.ticks = ticks
    if dataset.series:
        energies = np.concatenate([s.y.ravel() for s in dataset.series])
        low = float(np.nanmin(energies))
        high = float(np.nanmax(energies))
        default = (max(low, -10.0), min(high, 6.0)) if dataset.zero else (low, high)
        dataset.options['window'] = {
            'label': 'Energy window',
            'range': [low, high],
            'value': options.get('window') or [round(default[0], 3), round(default[1], 3)],
        }
    return dataset


def _phonon_unit(header: list[str]) -> str:
    for line in header:
        match = re.search(r'\((cm-1|cm\^-1|THz|meV)\)', line)
        if match:
            return {'cm-1': 'cm⁻¹', 'cm^-1': 'cm⁻¹'}.get(match.group(1), match.group(1))
    return ''


def load_phonon_bands(ref: DatasetRef, root: str, options: dict) -> Dataset:
    dataset = _new(ref)
    file = ref.files[0]
    data, header = read_columns(_path(root, file))
    unit = _phonon_unit(header)
    dataset.y_label = axis_title('Frequency', unit)
    if data.shape[0]:
        dataset.series.append(
            Series('phonons', data[:, 0], data[:, 1:], role='series-1', meta={'file': file})
        )
        if float(np.nanmin(data[:, 1:])) < -1e-3 * max(1.0, float(np.nanmax(np.abs(data[:, 1:])))):
            dataset.notes.append('Imaginary (unstable) modes are plotted as negative frequencies.')
    labels = re.sub(r'_band\.dat$', '_band.labels', _path(root, file))
    if os.path.isfile(labels):
        ticks = []
        with open(labels, 'r', encoding='utf-8', errors='replace') as handle:
            for line in handle:
                tokens = line.split(None, 1)
                if len(tokens) == 2 and not line.startswith('#'):
                    try:
                        label = tokens[1].strip().replace('$\\Gamma$', 'Γ').replace('$', '')
                        ticks.append((float(tokens[0]), 'Γ' if label in ('G', 'Gamma') else label))
                    except ValueError:
                        continue
        merged: list[tuple[float, str]] = []
        for position, label in ticks:
            if merged and abs(merged[-1][0] - position) < 1e-9:
                merged[-1] = (position, merged[-1][1] + '|' + label)
            else:
                merged.append((position, label))
        dataset.ticks = merged
    return dataset


def load_transport(ref: DatasetRef, root: str, options: dict) -> Dataset:
    dataset = _new(ref)
    tables = []
    for file, label in zip(ref.files, ref.labels):
        data, _header = read_columns(_path(root, file))
        if data.shape[0] and data.shape[1] >= 8:
            tables.append((label, file, data))
    if not tables:
        dataset.notes.append('no transport table (T, E and six tensor components expected)')
        return dataset
    temperatures = sorted({float(t) for _, _, d in tables for t in d[:, 0]})
    components = ['trace/3', *TENSOR_COLUMNS]
    component = _choice(options, 'component', 'Component', components, 'trace/3')
    _option(dataset, 'component', 'Component', components, component)
    shown = _shown_temperatures(dataset, options, temperatures)
    roles = dict(zip(shown, ordinal_roles(len(shown))))
    for label, file, data in tables:
        for temperature in shown:
            rows = data[np.isclose(data[:, 0], temperature)]
            if component == 'trace/3':
                values = rows[:, 2:5].mean(axis=1)
            else:
                values = rows[:, 2 + TENSOR_COLUMNS.index(component)]
            name = '%g K' % temperature + (' · %s' % label if len(tables) > 1 else '')
            dataset.series.append(
                Series(
                    name,
                    rows[:, 1],
                    values,
                    role=roles[temperature],
                    dash='dash' if label == 'spin ↓' else 'solid',
                    meta={'file': file},
                )
            )
    return dataset


def _shown_temperatures(dataset: Dataset, options: dict, temperatures: list[float]) -> list[float]:
    """The temperatures drawn (at most five, chosen by the user or evenly spread)."""
    chosen = options.get('temperatures')
    if isinstance(chosen, list) and chosen:
        shown = [t for t in temperatures if any(abs(t - float(c)) < 1e-6 for c in chosen)]
    else:
        shown = spread(temperatures)
    if len(shown) > MAX_ORDINAL:
        dataset.notes.append('At most %d temperatures at once (one colour ramp).' % MAX_ORDINAL)
        shown = spread(shown)
    shown = shown or temperatures[:1]
    dataset.options['temperatures'] = {
        'label': 'Temperatures (K)',
        'choices': temperatures,
        'value': shown,
        'multiple': True,
        'max': MAX_ORDINAL,
    }
    return shown


def load_hall(ref: DatasetRef, root: str, options: dict) -> Dataset:
    dataset = _new(ref)
    tables = []
    for file, label in zip(ref.files, ref.labels):
        data, _header = read_columns(_path(root, file))
        if data.shape[0] and data.shape[1] >= 3:
            tables.append((file, label, data))
    if not tables:
        return dataset
    temperatures = sorted({float(t) for _, _, d in tables for t in d[:, 0]})
    shown = _shown_temperatures(dataset, options, temperatures)
    for file, label, data in tables:
        for temperature, role in zip(shown, ordinal_roles(len(shown))):
            rows = data[np.isclose(data[:, 0], temperature)]
            name = '%g K' % temperature + (' · %s' % label if len(ref.files) > 1 else '')
            dataset.series.append(
                Series(name, rows[:, 1], rows[:, 2], role=role, meta={'file': file})
            )
    return dataset


ORBITAL = re.compile(r'^(?P<element>[A-Z][a-z]?)(?P<site>\d+)_(?P<n>\d)(?P<l>[spdf])(?P<m>\w*)$')


def _pdos_key(label: str, grouping: str) -> str:
    match = ORBITAL.match(label)
    if not match:
        name = 'orbital %s' % label if label.isdigit() else label
        return name if grouping == 'orbital' else 'orbitals'
    element, site, n, l, m = match.group('element', 'site', 'n', 'l', 'm')
    return {
        'orbital': label,
        'site': element + site,
        'shell': '%s%s %s%s' % (element, site, n, l),
        'element': element,
        'element and l': '%s %s' % (element, l),
    }[grouping]


def load_pdos(ref: DatasetRef, root: str, options: dict) -> Dataset:
    dataset = _new(ref)
    labels = ref.labels
    structured = any(ORBITAL.match(label) for label in labels)
    groupings = (
        ['element and l', 'element', 'site', 'shell', 'orbital'] if structured else ['orbital']
    )
    grouping = _choice(options, 'group', 'Group by', groupings, groupings[0])
    _option(dataset, 'group', 'Group by', groupings, grouping)
    displays = ['lines', 'stacked']
    display = _choice(options, 'display', 'Display', displays, 'lines')
    _option(dataset, 'display', 'Display', displays, display)
    sums: 'OrderedDict[str, np.ndarray]' = OrderedDict()
    energy = None
    for file, label in zip(ref.files, labels):
        data, _header = read_columns(_path(root, file))
        if data.shape[0] == 0:
            continue
        energy = data[:, 0] if energy is None else energy
        key = _pdos_key(label, grouping)
        sums[key] = sums.get(key, 0.0) + data[:, 1]
    if energy is None:
        return dataset
    keys = sorted(
        sums,
        key=lambda k: -float(np.trapezoid(sums[k], energy))
        if hasattr(np, 'trapezoid')
        else -float(np.trapz(sums[k], energy)),
    )
    if len(keys) > MAX_SERIES:
        kept, rest = keys[: MAX_SERIES - 1], keys[MAX_SERIES - 1 :]
        other = sum(sums[k] for k in rest)
        keys = kept + ['other (%d)' % len(rest)]
        sums['other (%d)' % len(rest)] = other
        dataset.notes.append(
            '%d groups: the smallest %d are summed in "other".' % (len(sums) - 1, len(rest))
        )
    order = sorted(keys)
    for key in keys:
        role = 'neutral' if key.startswith('other') else CATEGORICAL[order.index(key) % 8]
        meta = {'stack': 'pdos'} if display == 'stacked' else {}
        dataset.series.append(Series(key, energy, sums[key], role=role, meta=meta))
    total = os.path.join(
        _path(root, ref.folder),
        re.sub(r'^.*_pdos(dk)?_', r'pdos\1_sum_', os.path.basename(ref.files[0])),
    )
    if os.path.isfile(total):
        data, _ = read_columns(total)
        if data.shape[0]:
            dataset.series.append(
                Series('total', data[:, 0], data[:, 1], role='neutral', dash='dot')
            )
    dataset.notes.append('spin %s' % ref.group.get('spin', '0'))
    return dataset


def load_texture(ref: DatasetRef, root: str, options: dict) -> Dataset:
    dataset = _new(ref)
    blocks, _header = read_blocks(_path(root, ref.files[0]))
    if not blocks or blocks[0].shape[1] < 5:
        dataset.notes.append('expected blocks of k, E, Sx, Sy, Sz')
        return dataset
    components = ['Sx', 'Sy', 'Sz'] if ref.group.get('op', 'spin') == 'spin' else ['Lx', 'Ly', 'Lz']
    component = _choice(options, 'component', 'Component', components, components[2])
    _option(dataset, 'component', 'Component', components, component)
    column = 2 + components.index(component)
    x = np.concatenate([np.append(b[:, 0], np.nan) for b in blocks])
    y = np.concatenate([np.append(b[:, 1], np.nan) for b in blocks])
    c = np.concatenate([np.append(b[:, column], np.nan) for b in blocks])
    dataset.series.append(
        Series(
            component,
            x,
            y,
            mode='markers',
            role='diverging',
            meta={'color': c, 'color_label': '⟨%s⟩ (ħ/2)' % component},
        )
    )
    dataset.ticks, _ = read_kpath(_path(root, ref.folder))
    return dataset


def load_weighted_bands(ref: DatasetRef, root: str, options: dict) -> Dataset:
    dataset = _new(ref)
    data, _header = read_columns(_path(root, ref.files[0]))
    if data.shape[0] == 0 or data.shape[1] < 3:
        return dataset
    dataset.series.append(
        Series(
            'weight',
            data[:, 0],
            data[:, 1],
            mode='markers',
            role='sequential',
            meta={'color': data[:, 2], 'color_label': 'site weight'},
        )
    )
    dataset.ticks, _ = read_kpath(_path(root, ref.folder))
    return dataset


def load_eliashberg(ref: DatasetRef, root: str, options: dict) -> Dataset:
    dataset = _new(ref)
    quantities = ['α²F(ω)', 'λ(ω) = 2∫α²F/ω dω']
    quantity = _choice(options, 'quantity', 'Quantity', quantities, quantities[0])
    _option(dataset, 'quantity', 'Quantity', quantities, quantity)
    for file, label, role in zip(ref.files, ref.labels, _labels_roles(ref)):
        data, header = read_columns(_path(root, file))
        if data.shape[0] == 0:
            continue
        omega, a2f = data[:, 0], data[:, 1]
        name = 'transport α²F_tr' if 'tr' in os.path.basename(file) else 'α²F'
        if quantity == quantities[0]:
            dataset.series.append(Series(name, omega, a2f, role=role, meta={'file': file}))
            dataset.y_label = 'α²F(ω)'
        else:
            integrand = np.where(omega > 0, 2.0 * a2f / np.where(omega > 0, omega, 1.0), 0.0)
            cumulative = np.concatenate(
                [[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(omega))]
            )
            dataset.series.append(
                Series(name.replace('α²F', 'λ'), omega, cumulative, role=role, meta={'file': file})
            )
            dataset.y_label = 'λ(ω)'
        for name_, value in header_parameters(header).items():
            unit = 'K' if name_.lower().startswith(('tc', 'w_log')) else ''
            dataset.table.append(
                {
                    'name': name_ + (' (transport)' if 'tr' in file else ''),
                    'value': value,
                    'unit': unit,
                }
            )
    return dataset


def load_map(ref: DatasetRef, root: str, options: dict) -> Dataset:
    dataset = _new(ref)
    path = _path(root, ref.files[0])
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        if {'kpoints', 'w', 'intensity'} <= keys:
            kpoints = archive['kpoints']
            distance = (
                np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(kpoints, axis=0), axis=1))])
                if len(kpoints) > 1
                else np.zeros(1)
            )
            intensity = archive['intensity']
            kinds = ['intensity'] + (['raw'] if 'raw' in keys else [])
            which = _choice(options, 'array', 'Array', kinds, 'intensity')
            _option(dataset, 'array', 'Array', kinds, which)
            values = archive[which]
            dataset.surface = {
                'x': distance,
                'y': archive['w'],
                'z': values.T,
                'x_label': 'k along the path',
                'y_label': axis_title('E − E_F', 'eV'),
                'z_label': 'Intensity (a.u.)',
                'diverging': False,
            }
            if 'energies' in keys and archive['energies'].size:
                energies = archive['energies']
                bands = energies.reshape(len(distance), -1)
                dataset.series.append(
                    Series('bare bands', distance, bands, role='neutral', dash='dot')
                )
            if 'metadata' in keys:
                dataset.notes.append(str(archive['metadata'])[:600])
            del intensity
        elif {'kdist', 'energy_ev', 'spectral'} <= keys:
            dataset.surface = {
                'x': archive['kdist'],
                'y': archive['energy_ev'],
                'z': np.asarray(archive['spectral']).T
                if archive['spectral'].shape[0] == len(archive['kdist'])
                else archive['spectral'],
                'x_label': 'k path',
                'y_label': axis_title('Energy', 'eV'),
                'z_label': 'A(k, E)',
                'diverging': False,
            }
            if 'sym_tick_distance' in keys and 'sym_tick_label' in keys:
                dataset.ticks = [
                    (float(d), 'Γ' if str(l) in ('G', 'gG', 'Gamma') else str(l))
                    for d, l in zip(archive['sym_tick_distance'], archive['sym_tick_label'])
                ]
            if 'temperature_k' in keys:
                dataset.table.append(
                    {
                        'name': 'temperature',
                        'value': float(np.asarray(archive['temperature_k']).ravel()[0]),
                        'unit': 'K',
                    }
                )
        else:
            dataset.family = 'npz'
            return load_npz(ref, root, options)
    dataset.family = 'map'
    return dataset


def load_npz(ref: DatasetRef, root: str, options: dict) -> Dataset:
    dataset = _new(ref)
    dataset.family = 'npz'
    path = _path(root, ref.files[0])
    rows = []
    plottable = []
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            try:
                value = archive[key]
            except ValueError:
                rows.append({'name': key, 'value': '(object array)', 'unit': ''})
                continue
            if value.ndim == 0:
                scalar = value.item()
                rows.append(
                    {
                        'name': key,
                        'value': scalar if isinstance(scalar, (int, float)) else str(scalar)[:200],
                        'unit': '',
                    }
                )
            else:
                rows.append(
                    {
                        'name': key,
                        'value': 'array %s %s' % ('×'.join(map(str, value.shape)), value.dtype),
                        'unit': '',
                    }
                )
                if value.ndim == 1 and value.size > 1 and np.issubdtype(value.dtype, np.number):
                    plottable.append(key)
        dataset.table = rows
        if len(plottable) >= 2:
            x_key = _choice(options, 'x', 'x', plottable, plottable[0])
            y_default = next(
                (k for k in plottable if k != x_key and archive[k].shape == archive[x_key].shape),
                plottable[1],
            )
            y_key = _choice(options, 'y', 'y', plottable, y_default)
            _option(dataset, 'x', 'x array', plottable, x_key)
            _option(dataset, 'y', 'y array', plottable, y_key)
            if archive[x_key].shape == archive[y_key].shape:
                dataset.series.append(Series(y_key, archive[x_key], archive[y_key]))
                dataset.x_label, dataset.y_label = x_key, y_key
            else:
                dataset.notes.append('%s and %s have different lengths' % (x_key, y_key))
    return dataset


def _text(path: str, limit: int = 400) -> str:
    lines = []
    with open(path, 'r', encoding='utf-8', errors='replace') as handle:
        for index, line in enumerate(handle):
            if index >= limit:
                lines.append('... (truncated)')
                break
            lines.append(line.rstrip('\n'))
    return '\n'.join(lines)


def load_text(ref: DatasetRef, root: str, options: dict) -> Dataset:
    dataset = _new(ref)
    dataset.family = 'text'
    parts = []
    for file in ref.files:
        header = '── %s' % file if len(ref.files) > 1 else ''
        parts.append((header + '\n' if header else '') + _text(_path(root, file)))
    dataset.text = '\n\n'.join(parts)
    return dataset


def load_sscha(ref: DatasetRef, root: str, options: dict) -> Dataset:
    dataset = _new(ref, x_label='Population', y_label=axis_title('Free energy', 'Ry'))
    folder = _path(root, ref.folder)
    with open(_path(root, ref.files[0]), 'r', encoding='utf-8') as handle:
        state = json.load(handle)
    for key, value in state.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            dataset.table.append({'name': key, 'value': value, 'unit': ''})
    populations = []
    for name in sorted(os.listdir(folder)):
        match = re.match(r'pop(\d+)$', name)
        summary = os.path.join(folder, name, 'summary.json')
        if match and os.path.isfile(summary):
            with open(summary, 'r', encoding='utf-8') as handle:
                populations.append((int(match.group(1)), json.load(handle)))
    if populations:
        populations.sort()
        x = np.array([p for p, _ in populations], dtype=float)
        energy = np.array([s.get('free_energy_ry', np.nan) for _, s in populations], dtype=float)
        dataset.series.append(Series('free energy', x, energy, mode='lines+markers'))
        error = [s.get('free_energy_error_ry') for _, s in populations]
        if all(e is not None for e in error):
            dataset.series[0].meta['error'] = np.array(error, dtype=float)
        for population, summary in populations:
            for key in ('kong_liu', 'lowest_cm1'):
                if key in summary:
                    dataset.table.append(
                        {'name': 'pop %d %s' % (population, key), 'value': summary[key], 'unit': ''}
                    )
    hessian = os.path.join(folder, 'hessian', 'summary.json')
    if os.path.isfile(hessian):
        with open(hessian, 'r', encoding='utf-8') as handle:
            for key, value in json.load(handle).items():
                if isinstance(value, (int, float, str, bool)):
                    dataset.table.append({'name': 'hessian %s' % key, 'value': value, 'unit': ''})
    return dataset


def load_dhva(ref: DatasetRef, root: str, options: dict) -> Dataset:
    dataset = _new(ref)
    angles = ['polar θ', 'azimuthal φ']
    angle = _choice(options, 'angle', 'Angle', angles, angles[0])
    _option(dataset, 'angle', 'Angle', angles, angle)
    for file, label, role in zip(ref.files, ref.labels, _labels_roles(ref)):
        data, header = read_columns(_path(root, file))
        if data.shape[0] == 0 or data.shape[1] < 3:
            continue
        x = data[:, 0] if angle == angles[0] else data[:, 1]
        dataset.series.append(
            Series(label or 'orbits', x, data[:, 2], mode='markers', role=role, meta={'file': file})
        )
    dataset.x_label = axis_title('Field angle ' + angle.split()[1], 'deg')
    return dataset


def load_manifest(ref: DatasetRef, root: str, options: dict) -> Dataset:
    dataset = _new(ref)
    dataset.family = 'text'
    with open(_path(root, ref.files[0]), 'r', encoding='utf-8') as handle:
        runs = json.load(handle).get('runs', [])
    if not runs:
        dataset.text = '(no run recorded)'
        return dataset
    choices = ['%d: %s' % (index + 1, run.get('finished', '?')) for index, run in enumerate(runs)]
    chosen = _choice(options, 'run', 'Run', choices, choices[-1])
    _option(dataset, 'run', 'Run', choices, chosen)
    run = runs[choices.index(chosen)]
    for key, value in run.items():
        if key != 'files':
            unit = 'eV' if key.endswith('_ev') else ''
            dataset.table.append({'name': key, 'value': value, 'unit': unit})
    lines = ['%-42s %-26s %s' % ('file', 'kind', 'bytes')]
    for entry in run.get('files', []):
        lines.append(
            '%-42s %-26s %d' % (entry['file'], entry.get('kind', '?'), entry.get('bytes', 0))
        )
    dataset.text = '\n'.join(lines)
    return dataset


def load_volume(ref: DatasetRef, root: str, options: dict) -> Dataset:
    from .volumes import load_volume as load

    return load(ref, root, options, _new(ref))


LOADERS: dict[str, Callable[[DatasetRef, str, dict], Dataset]] = {
    'curve': load_curve,
    'columns': load_columns,
    'temperature_columns': load_temperature_columns,
    'bands': load_bands,
    'phonon_bands': load_phonon_bands,
    'transport': load_transport,
    'hall': load_hall,
    'pdos': load_pdos,
    'texture': load_texture,
    'weighted_bands': load_weighted_bands,
    'eliashberg': load_eliashberg,
    'map': load_map,
    'npz': load_npz,
    'text': load_text,
    'sscha': load_sscha,
    'dhva': load_dhva,
    'volume': load_volume,
    'manifest': load_manifest,
}


def load(ref: DatasetRef, root: str, options: dict | None = None) -> Dataset:
    """Load the files of a reference.

    Parameters
    ----------
    ref : DatasetRef
        The dataset found by discovery.
    root : str
        Folder the reference paths are relative to.
    options : dict, optional
        Selected options (temperature, component, grouping, ...).

    Returns
    -------
    Dataset
    """
    return LOADERS[ref.family](ref, root, options or {})
