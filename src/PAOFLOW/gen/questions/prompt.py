"""CLI runner of the declarative questions: the interactive prompts of ``paoflow-gen``.

The runner walks the items of a workflow in order and asks each applicable question with the
prompt helpers of :mod:`PAOFLOW.gen.paoflow_driver` (``ask``, ``ask_int``, ...), so the text the
user sees and the answers a script can pipe in are those of the hand-written prompts it replaces.
Invalid answers (a parse error or a validation message) are reported and asked again.
"""

from __future__ import annotations

from typing import Any, Sequence

from .model import Answers, Context, Item, Note, Question

#: Prompt of the numbered multi-select (the property menu of the run workflow).
MENU_ENTRY_PROMPT = "Enter a comma/space separated list (e.g. '1 2 8'): "


def ask_menu(title: str, choices: Sequence[tuple[str, str]]) -> list[str]:
    """Numbered multi-select: print the menu, read the numbers, return the chosen keys.

    Parameters
    ----------
    title : str
        Heading printed after a blank line.
    choices : sequence of (str, str)
        ``(key, label)`` pairs.

    Returns
    -------
    list of str
        Chosen keys in the order given, without duplicates; unknown numbers are ignored.
    """
    from .. import paoflow_driver as pd

    print('\n%s' % title)
    for index, (_key, label) in enumerate(choices, 1):
        print(f'  {index}: {label}')
    raw = pd._input(MENU_ENTRY_PROMPT).strip()
    chosen: list[str] = []
    for token in raw.replace(',', ' ').split():
        try:
            index = int(token)
        except ValueError:
            continue
        if 1 <= index <= len(choices):
            key = choices[index - 1][0]
            if key not in chosen:
                chosen.append(key)
    return chosen


def _ask_once(question: Question, answers: Answers, context: Context) -> Any:
    """One prompt for ``question`` (raises :class:`ValueError` on an unparsable answer)."""
    from .. import paoflow_driver as pd

    default = question.default_value(answers, context)
    kind = question.kind
    if kind == 'int':
        return pd.ask_int(question.prompt, default)
    if kind == 'float':
        return pd.ask_float(question.prompt, default)
    if kind == 'bool':
        return pd.ask_yes_no(question.prompt, default)
    if kind == 'choice':
        return pd.ask_choice(question.prompt, question.choice_values(answers, context), default)
    if kind == 'menu':
        return ask_menu(question.prompt, question.choice_values(answers, context))
    if kind == 'names':
        shown = ', '.join(default) if isinstance(default, (list, tuple)) else default
        raw = pd.ask(question.prompt, shown)
        return question.parse(raw) if question.parse else raw
    if kind == 'rows':
        rows = []
        while True:
            line = pd._input(question.row_prompt).strip()
            if not line:
                return rows
            rows.append(question.parse(line) if question.parse else line)
    raw = pd.ask(question.prompt, default)
    return question.parse(raw) if question.parse else raw


def ask_question(question: Question, answers: Answers, context: Context) -> Any:
    """Ask ``question`` until the answer parses and validates.

    Parameters
    ----------
    question : Question
        The question.
    answers : dict
        Answers so far.
    context : Context
        The session context.

    Returns
    -------
    Any
        The stored value.
    """
    while True:
        try:
            value = _ask_once(question, answers, context)
        except ValueError as error:
            print('  %s' % error)
            continue
        message = question.validate(value, answers, context) if question.validate else None
        if message is None:
            return value
        print(message)


def overridden_value(question: Question, context: Context) -> Any:
    """The command-line value answering ``question``, or ``None``.

    Parameters
    ----------
    question : Question
        The question.
    context : Context
        The session context.

    Returns
    -------
    Any
        The parsed override (``None`` when the option is not given).
    """
    if not question.override:
        return None
    value = context.overrides.get(question.override)
    if not value:
        return None
    if isinstance(value, str) and question.parse is not None:
        return question.parse(value)
    return value


def ask_items(
    items: Sequence[Item], context: Context, answers: Answers | None = None
) -> Answers | None:
    """Ask the applicable items in order (the CLI of a workflow).

    Parameters
    ----------
    items : sequence of Question or Note
        The workflow items.
    context : Context
        The session context.
    answers : dict, optional
        Answers known beforehand (they seed the predicates; asked keys are overwritten).

    Returns
    -------
    dict or None
        The answers; ``None`` when a question with ``abort_if_empty`` got an empty answer (its
        message is printed).
    """
    answers = dict(answers or {})
    for item in items:
        if isinstance(item, Note):
            if item.applies(answers, context):
                print(item.message(answers, context))
            continue
        if not item.cli or not item.applies(answers, context):
            continue
        value = overridden_value(item, context)
        if value is None:
            value = ask_question(item, answers, context)
        answers[item.key] = value
        if item.abort_if_empty and not value:
            print(item.abort_if_empty)
            return None
    return answers
