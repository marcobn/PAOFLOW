"""Colours of the results figures, by role, for the light and the dark theme.

One table for every renderer: the GUI resolves the roles of a figure specification in the browser
(``/api/results/palette``), the static report embeds the table, and the matplotlib export reads
the light column.  The categorical order is fixed (colour follows the entity, never its rank) and
was checked with the palette validator of the data-visualisation guidelines on the GUI surfaces
(``#ffffff`` light, ``#171b23`` dark): adjacent CVD ΔE >= 8.4, normal-vision ΔE >= 19.3; three light
slots sit below 3:1 contrast, so every figure keeps its legend and a table view.  Ordered series
(temperatures) use five steps of one hue (adjacent ΔL >= 0.06 in both themes); colour scales are
one hue (sequential) or blue to red through a neutral grey (diverging).
"""

from __future__ import annotations

PALETTE: dict[str, dict] = {
    'light': {
        'surface': '#ffffff',
        'text': '#1c2330',
        'muted': '#5d6778',
        'grid': '#e9ecf0',
        'axis': '#c3c8d0',
        'neutral': '#898781',
        'diverging': '#7a7f88',
        'sequential': '#2a78d6',
        'series': [
            '#2a78d6',
            '#eb6834',
            '#1baf7a',
            '#eda100',
            '#e87ba4',
            '#008300',
            '#6250d6',
            '#e34948',
        ],
        'ordinal': ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281'],
        'scales': {
            'sequential': [
                [0.0, '#ffffff'],
                [0.15, '#cde2fb'],
                [0.35, '#86b6ef'],
                [0.55, '#3987e5'],
                [0.75, '#1c5cab'],
                [1.0, '#0d366b'],
            ],
            'sequential-mesh': [[0.0, '#9ec5f4'], [0.5, '#2a78d6'], [1.0, '#0d366b']],
            'diverging': [
                [0.0, '#104281'],
                [0.25, '#5598e7'],
                [0.5, '#f0efec'],
                [0.75, '#e66767'],
                [1.0, '#a8302f'],
            ],
        },
    },
    'dark': {
        'surface': '#171b23',
        'text': '#e4e8ef',
        'muted': '#a3acbb',
        'grid': '#262c37',
        'axis': '#3a4250',
        'neutral': '#898781',
        'diverging': '#9aa0a8',
        'sequential': '#3987e5',
        'series': [
            '#3987e5',
            '#d95926',
            '#199e70',
            '#c98500',
            '#d55181',
            '#008300',
            '#9085e9',
            '#e66767',
        ],
        'ordinal': ['#184f95', '#2a78d6', '#5598e7', '#86b6ef', '#b7d3f6'],
        'scales': {
            'sequential': [
                [0.0, '#171b23'],
                [0.15, '#0d366b'],
                [0.35, '#184f95'],
                [0.55, '#2a78d6'],
                [0.75, '#6da7ec'],
                [1.0, '#cde2fb'],
            ],
            'sequential-mesh': [[0.0, '#184f95'], [0.5, '#3987e5'], [1.0, '#cde2fb']],
            'diverging': [
                [0.0, '#3987e5'],
                [0.25, '#1f3d66'],
                [0.5, '#383835'],
                [0.75, '#7a2b2b'],
                [1.0, '#e66767'],
            ],
        },
    },
}


def resolve(role: str, theme: str = 'light') -> str:
    """Colour of a role (``series-3``, ``ord-2``, ``neutral``, ``axis``, ...).

    Parameters
    ----------
    role : str
        The role.
    theme : str, optional
        ``'light'`` or ``'dark'``.

    Returns
    -------
    str
        A hex colour (the role itself when it is already a colour).
    """
    table = PALETTE[theme]
    if '@' in role:  # 'series-3@0.8': the colour with an opacity
        base, alpha = role.split('@', 1)
        colour = resolve(base, theme).lstrip('#')
        red, green, blue = (int(colour[i : i + 2], 16) for i in (0, 2, 4))
        return 'rgba(%d, %d, %d, %s)' % (red, green, blue, alpha)
    if role.startswith('series-'):
        return table['series'][(int(role.split('-')[1]) - 1) % len(table['series'])]
    if role.startswith('ord-'):
        return table['ordinal'][(int(role.split('-')[1]) - 1) % len(table['ordinal'])]
    return table.get(role, role) if isinstance(table.get(role, role), str) else role


def resolve_spec(spec: dict, theme: str = 'light') -> dict:
    """A figure specification with every role replaced by its colour (deep copy).

    Parameters
    ----------
    spec : dict
        Specification of :func:`PAOFLOW.results.figures.figure_spec`.
    theme : str, optional
        ``'light'`` or ``'dark'``.

    Returns
    -------
    dict
    """
    import copy

    table = PALETTE[theme]
    spec = copy.deepcopy(spec)

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ('color', 'fillcolor') and isinstance(value, str):
                    node[key] = resolve(value, theme)
                elif key == 'colorscale' and isinstance(value, str) and value.startswith('role:'):
                    node[key] = table['scales'][value[5:]]
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(spec)
    return spec
