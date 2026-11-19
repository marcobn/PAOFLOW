"""Number lists written as small Python expressions, evaluated without ``eval``.

The generator accepts lists such as ``'[n for n in range(450, 650, 5)]'`` (the resonance-Raman
laser wavelengths).  The expression is parsed with :mod:`ast` and only numbers, the arithmetic
operators, list and tuple displays, ``range`` and one comprehension over ``range`` are evaluated;
anything else (names, attributes, other calls, subscripts) is rejected.  This makes the input safe
to accept from the browser GUI as well as from the terminal.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

#: Largest list an expression may produce (a guard against ``range(10**9)``).
MAX_ITEMS = 100000

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _range(arguments: list[Any]) -> range:
    if not 1 <= len(arguments) <= 3 or not all(isinstance(a, int) for a in arguments):
        raise ValueError('range() takes one to three integers')
    values = range(*arguments)
    if len(values) > MAX_ITEMS:
        raise ValueError('range() longer than %d items' % MAX_ITEMS)
    return values


def _evaluate(node: ast.AST, names: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, names)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if isinstance(node.value, bool):
            raise ValueError('booleans are not numbers here')
        return node.value
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand, names))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _BINARY[type(node.op)](_evaluate(node.left, names), _evaluate(node.right, names))
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_evaluate(element, names) for element in node.elts]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'range'
        and not node.keywords
    ):
        return _range([_evaluate(argument, names) for argument in node.args])
    if isinstance(node, ast.ListComp) and len(node.generators) == 1:
        generator = node.generators[0]
        if not isinstance(generator.target, ast.Name) or generator.is_async:
            raise ValueError('the comprehension must run one name over range()')
        iterable = _evaluate(generator.iter, names)
        if not isinstance(iterable, (range, list)):
            raise ValueError('the comprehension must run over range() or a list')
        values = []
        for value in iterable:
            scope = dict(names)
            scope[generator.target.id] = value
            if all(_evaluate(condition, scope) for condition in generator.ifs):
                values.append(_evaluate(node.elt, scope))
        return values
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        comparisons = {
            ast.Lt: operator.lt,
            ast.LtE: operator.le,
            ast.Gt: operator.gt,
            ast.GtE: operator.ge,
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
        }
        if type(node.ops[0]) in comparisons:
            left = _evaluate(node.left, names)
            right = _evaluate(node.comparators[0], names)
            return comparisons[type(node.ops[0])](left, right)
    raise ValueError('unsupported expression: %s' % ast.dump(node)[:80])


def evaluate_number_list(expression: str) -> list[float]:
    """Numbers of a list expression (numbers, arithmetic, lists, ``range``, one comprehension).

    Parameters
    ----------
    expression : str
        For example ``'[n for n in range(450, 650, 5)]'``, ``'range(400, 700, 50)'`` or
        ``'[488, 514.5, 2 * 266]'``.

    Returns
    -------
    list of float

    Raises
    ------
    ValueError
        If the expression is not valid Python or uses anything outside the allowed subset.
    """
    try:
        tree = ast.parse(expression.strip(), mode='eval')
    except SyntaxError as error:
        raise ValueError('not a list expression: %s' % error.msg) from None
    value = _evaluate(tree, {})
    if isinstance(value, (int, float)):
        return [float(value)]
    try:
        return [float(item) for item in value]
    except TypeError:
        raise ValueError('the expression does not give a list of numbers') from None
