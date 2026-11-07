"""Form resolution of the declarative questions: the browser GUI's counterpart of the CLI runner.

:func:`resolve_form` walks the items of a workflow like the CLI does, but with the answers the
browser holds (raw field values) instead of prompts.  For every item it reports whether it is
shown, the value to display, the default, the allowed choices and the error, and, when nothing is
in error, the assembled configuration.  Blank fields mean "the default", as a blank answer does on
the terminal, so a form left untouched gives the configuration of a CLI run that accepts every
default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .model import Answers, Context, Note, Question, Workflow

_MISSING = object()


@dataclass
class ResolvedForm:
    """The state of a workflow form for a set of raw answers.

    Parameters
    ----------
    items : list of dict
        One description per item (see :func:`describe_question`), in CLI order.
    answers : dict
        Parsed answers of the shown questions (the input of ``assemble``).
    errors : dict
        Error message per question key.
    config : dict or None
        The assembled configuration when there is no error.
    """

    items: list[dict] = field(default_factory=list)
    answers: Answers = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    config: dict | None = None

    @property
    def complete(self) -> bool:
        """Whether the configuration could be assembled."""
        return self.config is not None

    def to_json(self) -> dict:
        """JSON-ready description (the configuration excluded).

        Returns
        -------
        dict
        """
        return {'items': self.items, 'errors': self.errors, 'complete': self.complete}


def _display(question: Question, value: Any) -> Any:
    """Value of a question as the browser field shows it."""
    if value is None:
        return ''
    if question.kind in ('int', 'float', 'text'):
        return (
            value
            if isinstance(value, str)
            else repr(value)
            if question.kind == 'float'
            else str(value)
        )
    if question.kind == 'rows':
        return [list(row) for row in value]
    return value


def label_and_hint(question: Question) -> tuple[str, str]:
    """Short label and hint of a question for the GUI.

    An explicit ``label`` wins (the prompt stays the tooltip); otherwise a trailing parenthesis of
    the prompt is split off as the hint: ``'Coarse q-grid per axis (ph.x; must divide the k-grid)'``
    gives ``('Coarse q-grid per axis', 'ph.x; must divide the k-grid')``.

    Parameters
    ----------
    question : Question
        The question.

    Returns
    -------
    tuple of str
    """
    import re

    prompt = question.prompt.strip()
    if question.label:
        return question.label, ''
    match = re.match(r'^(.*?)\s*\((.+)\)\s*([?:]?)\s*$', prompt, re.S)
    if match and match.group(1):
        return match.group(1) + match.group(3).replace(':', ''), match.group(2)
    return prompt.rstrip(':'), ''


def describe_question(question: Question, workflow: str = '') -> dict:
    """Static description of a question (what does not depend on the answers).

    Parameters
    ----------
    question : Question
        The question.
    workflow : str, optional
        Workflow name (selects the documentation table).

    Returns
    -------
    dict
        ``type``, ``key``, ``kind``, ``label``, ``prompt``, ``help``, ``unit``, ``section``,
        ``level``, ``columns``, ``gui_only``, ``widget`` and ``doc`` (the documentation of the
        PAOFLOW parameter it feeds, :mod:`PAOFLOW.gen.questions.docs`).
    """
    label, hint = label_and_hint(question)
    return {
        'type': 'question',
        'key': question.key,
        'kind': question.kind,
        'label': label,
        'hint': hint,
        'prompt': question.prompt,
        'help': question.help,
        'unit': question.unit,
        'section': question.section,
        'level': question.level,
        'columns': list(question.columns),
        'gui_only': not question.cli,
        'widget': question.widget,
        'doc': _doc(workflow, question),
    }


def _doc(workflow: str, question: Question) -> dict | None:
    from .docs import question_doc

    found = question_doc(workflow, question.key, question.doc)
    return {'text': found[0], 'source': found[1]} if found else None


def _number(question: Question, raw: Any) -> Any:
    if isinstance(raw, bool):
        raise ValueError('not a number')
    if question.kind == 'int':
        if isinstance(raw, int):
            return raw
        if isinstance(raw, float) and raw.is_integer():
            return int(raw)
        try:
            return int(str(raw).strip())
        except ValueError:
            raise ValueError('not an integer: %r' % raw) from None
    if isinstance(raw, (int, float)):
        return float(raw)
    try:
        return float(str(raw).strip())
    except ValueError:
        raise ValueError('not a number: %r' % raw) from None


def _blank(raw: Any) -> bool:
    return raw is _MISSING or raw is None or (isinstance(raw, str) and not raw.strip())


def interpret(question: Question, raw: Any, default: Any, choices: list) -> Any:
    """Parsed value of a raw field value (blank means the default).

    Parameters
    ----------
    question : Question
        The question.
    raw : Any
        The browser value (strings for text and numbers, booleans, lists for multi-selects and
        rows), or the missing marker.
    default : Any
        The default for the current answers.
    choices : list
        Allowed values (``choice``, ``names``) or ``(key, label)`` pairs (``menu``).

    Returns
    -------
    Any
        The value the CLI would have stored.

    Raises
    ------
    ValueError
        With the message to show next to the field.
    """
    kind = question.kind
    if kind in ('int', 'float'):
        return default if _blank(raw) else _number(question, raw)
    if kind == 'bool':
        if _blank(raw):
            return default
        if isinstance(raw, str):
            return raw.strip().lower().startswith('y') or raw.strip().lower() == 'true'
        return bool(raw)
    if kind == 'choice':
        if _blank(raw):
            return default
        if raw not in choices:
            raise ValueError('choose one of %s' % ', '.join(map(str, choices)))
        return raw
    if kind == 'menu':
        if _blank(raw):
            return []
        keys = [key for key, _label in choices]
        return [key for key in dict.fromkeys(raw) if key in keys]
    if kind == 'names':
        if _blank(raw):
            return list(default or [])
        if isinstance(raw, str):
            return question.parse(raw) if question.parse else raw.split(',')
        return list(dict.fromkeys(str(name).strip() for name in raw if str(name).strip()))
    if kind == 'rows':
        rows = []
        for row in (default or []) if _blank(raw) else raw:
            cells = [
                str(cell).strip() for cell in (row if isinstance(row, (list, tuple)) else [row])
            ]
            line = ' '.join(cell for cell in cells if cell)
            if line:
                rows.append(question.parse(line) if question.parse else line)
        return rows
    text = default if _blank(raw) else str(raw)
    text = '' if text is None else text
    return question.parse(text) if question.parse else text


def resolve_form(workflow: Workflow, raw_answers: dict, context: Context) -> ResolvedForm:
    """Resolve a workflow form: shown items, values, defaults, errors and the configuration.

    Parameters
    ----------
    workflow : Workflow
        The workflow.
    raw_answers : dict
        Field values by question key (missing or blank: the default).
    context : Context
        The session context (working directory and overrides such as the elphon ``out``).

    Returns
    -------
    ResolvedForm
    """
    form = ResolvedForm()
    for item in workflow.items:
        if isinstance(item, Note):
            shown = item.applies(form.answers, context)
            form.items.append(
                {
                    'type': 'note',
                    'text': item.message(form.answers, context).strip('\n') if shown else '',
                    'visible': shown,
                    'section': item.section,
                }
            )
            continue
        entry = describe_question(item, workflow.name)
        shown = item.applies(form.answers, context)
        entry['visible'] = shown
        form.items.append(entry)
        if not shown:
            continue
        default = item.default_value(form.answers, context)
        choices = item.choice_values(form.answers, context)
        raw = raw_answers.get(item.key, _MISSING)
        error = None
        try:
            value = interpret(item, raw, default, choices)
        except ValueError as exc:
            error = str(exc).strip()
            try:
                value = interpret(item, _MISSING, default, choices)
            except ValueError:
                value = None
        if error is None and item.validate is not None:
            message = item.validate(value, form.answers, context)
            error = message.strip() if message else None
        if error is None and item.abort_if_empty and not value:
            error = item.abort_if_empty
        entry.update(
            {
                'value': _display(item, raw if not _blank(raw) else default)
                if item.kind not in ('menu', 'rows')
                else _display(item, value),
                'default': _display(item, default),
                'is_default': _blank(raw),
                'choices': [list(c) if isinstance(c, tuple) else c for c in choices],
                'error': error,
            }
        )
        if error:
            form.errors[item.key] = error
        if not item.cli and (_blank(raw) or value == interpret(item, _MISSING, default, choices)):
            continue  # GUI-only refinement left at the builder default
        form.answers[item.key] = value
    if not form.errors:
        try:
            form.config = workflow.assemble(dict(form.answers), context)
        except (KeyError, ValueError, TypeError) as exc:
            form.errors['_assemble'] = 'cannot assemble the configuration: %s' % exc
    return form
