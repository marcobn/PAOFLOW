"""Field help from the PAOFLOW docstrings and the system.py comment tables."""

from __future__ import annotations

from PAOFLOW.gen.questions import WORKFLOWS
from PAOFLOW.gen.questions.docs import FIELD_DOCS, docstring_parameters, field_doc, plain_text
from PAOFLOW.gen.questions.model import Context
from PAOFLOW.gen.questions.resolve import resolve_form


def numpy_style(a, b, c):
    """Summary.

    Parameters
    ----------
    a, b : float
        First line of a and b
        continued.
    c : int

    Other Parameters
    ----------------
    d : str
        Documented later.

    Returns
    -------
    None
    """


def argument_style(x, y):
    """
    Arguments:
        x (float): the x value
            over two lines.
        y, z: shared text.
    Returns:
        None
    """


def test_parsers():
    assert docstring_parameters(numpy_style) == {
        'a': 'First line of a and b continued.',
        'b': 'First line of a and b continued.',
        'd': 'Documented later.',
    }
    assert docstring_parameters(argument_style) == {
        'x': 'the x value over two lines.',
        'y': 'shared text.',
        'z': 'shared text.',
    }


def test_plain_text():
    assert (
        plain_text(r'width of :math:`\delta` in ``eV`` (:mod:`PAOFLOW.x.y`)')
        == 'width of δ in eV (y)'
    )


def test_every_reference_resolves():
    missing = [
        (w, k)
        for w, table in FIELD_DOCS.items()
        for k, ref in table.items()
        if field_doc(ref) is None
    ]
    assert missing == []
    for workflow, table in FIELD_DOCS.items():
        keys = {q.key for q in WORKFLOWS[workflow].questions()}
        assert set(table) <= keys, set(table) - keys


def test_resolved_items_carry_the_docs(tmp_path):
    form = resolve_form(
        WORKFLOWS['run'], {'properties': ['nonlinear']}, Context(workdir=str(tmp_path))
    )
    eta = next(i for i in form.items if i.get('key') == 'nlo_eta')
    assert eta['doc']['source'] == 'nonlinear_optics' and 'δ' in eta['doc']['text']
