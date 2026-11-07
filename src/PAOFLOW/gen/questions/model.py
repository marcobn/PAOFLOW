"""Data model of the ``paoflow-gen`` questions, shared by the CLI prompts and the browser GUI.

A workflow is an ordered list of :class:`Question` and :class:`Note` items plus an ``assemble``
function that turns the answers into the configuration dictionary the script builders read.  The
CLI asks the items in order (:mod:`PAOFLOW.gen.questions.prompt`); the GUI renders the same items
as a form and resolves visibility, defaults and validation on the server.  Every predicate and
default is a Python callable of the answers so far and of the :class:`Context` (the working
directory and what was detected in it), so nothing is duplicated in JavaScript.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

Answers = dict[str, Any]

#: Kinds of questions.  ``text`` answers are strings (optionally parsed), ``menu`` is the numbered
#: multi-select of the property run, ``names`` a comma-separated multi-select by name and ``rows``
#: a list of lines entered until a blank one (the ACBN0 manifolds).
QUESTION_KINDS = ('text', 'int', 'float', 'bool', 'choice', 'menu', 'names', 'rows')


@dataclass
class Context:
    """Working directory, command-line overrides and cached detections of one generator session.

    Parameters
    ----------
    workdir : str
        Directory scanned for the Quantum ESPRESSO artefacts.
    overrides : Mapping[str, Any], optional
        Values given on the command line (``savedir``, ``prefix``, ``upf``, ``out``).  A question
        whose ``override`` names a truthy entry is not asked.
    cache : dict, optional
        Results of the detections (computed once per session).
    """

    workdir: str
    overrides: Mapping[str, Any] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_args(cls, args: Any, workdir: str, names: Sequence[str]) -> 'Context':
        """Context of a CLI run.

        Parameters
        ----------
        args : argparse.Namespace or any object
            The command-line options; missing attributes count as not given.
        workdir : str
            The working directory (used as given).
        names : sequence of str
            Option names that may override a question.

        Returns
        -------
        Context
        """
        overrides = {name: getattr(args, name, None) for name in names}
        return cls(workdir=workdir, overrides=overrides)

    def cached(self, key: str, compute: Callable[[], Any]) -> Any:
        """Value of ``compute()``, computed once per context.

        Parameters
        ----------
        key : str
            Cache key.
        compute : callable
            Function without arguments.

        Returns
        -------
        Any
        """
        if key not in self.cache:
            self.cache[key] = compute()
        return self.cache[key]


Predicate = Callable[[Answers, Context], bool]


@dataclass(frozen=True)
class Question:
    """One question of a generator workflow.

    Parameters
    ----------
    key : str
        Name of the answer (usually the configuration key it feeds).
    prompt : str
        CLI prompt, verbatim; the GUI label unless ``label`` is set.
    kind : str
        One of :data:`QUESTION_KINDS`.
    default : Any or callable
        Default value, or ``default(answers, context)``.  ``names`` defaults are lists.
    choices : sequence or callable, optional
        Allowed values (``choice``, ``names``) or ``(key, label)`` pairs (``menu``).
    when : callable, optional
        ``when(answers, context)``: the question is asked (shown) only if true.
    parse : callable, optional
        Text answer to stored value (``text``, ``names`` and every ``rows`` line); raises
        :class:`ValueError` with the message shown to the user.
    validate : callable, optional
        ``validate(value, answers, context)`` returns an error message (the CLI prints it and asks
        again) or ``None``.
    abort_if_empty : str, optional
        Message printed when the answer is empty; the workflow then generates nothing.
    override : str, optional
        Command-line option that answers the question when given.
    row_prompt : str, optional
        Prompt of each line of a ``rows`` question.
    label : str, optional
        Short GUI label (the prompt is used when empty).
    help : str, optional
        Longer GUI help.
    unit : str, optional
        Unit shown next to the GUI field.
    section : str, optional
        GUI section title.
    level : str, optional
        ``'basic'`` or ``'advanced'`` (collapsed in the GUI; the CLI asks both).
    columns : sequence of str, optional
        Column titles of a ``rows`` question in the GUI.
    cli : bool, optional
        Asked by the CLI (default).  ``False`` marks a GUI-only refinement: an optional builder
        setting that the CLI leaves at the builder default; the GUI passes it on only when the
        user changes it, so a GUI run with defaults writes the same files as the CLI.
    """

    key: str
    prompt: str
    kind: str = 'text'
    default: Any = None
    choices: Any = None
    when: Predicate | None = None
    parse: Callable[[str], Any] | None = None
    validate: Callable[[Any, Answers, Context], str | None] | None = None
    abort_if_empty: str | None = None
    override: str | None = None
    row_prompt: str = ''
    label: str = ''
    help: str = ''
    unit: str = ''
    section: str = ''
    level: str = 'basic'
    columns: Sequence[str] = ()
    cli: bool = True

    def __post_init__(self) -> None:
        if self.kind not in QUESTION_KINDS:
            raise ValueError('unknown question kind %r (%s)' % (self.kind, self.key))

    def applies(self, answers: Answers, context: Context) -> bool:
        """Whether the question is asked for these answers.

        Parameters
        ----------
        answers : dict
            Answers so far.
        context : Context
            The session context.

        Returns
        -------
        bool
        """
        return self.when is None or bool(self.when(answers, context))

    def default_value(self, answers: Answers, context: Context) -> Any:
        """The default for these answers (callables evaluated).

        Parameters
        ----------
        answers : dict
            Answers so far.
        context : Context
            The session context.

        Returns
        -------
        Any
        """
        return self.default(answers, context) if callable(self.default) else self.default

    def choice_values(self, answers: Answers, context: Context) -> list:
        """The allowed values for these answers (callables evaluated).

        Parameters
        ----------
        answers : dict
            Answers so far.
        context : Context
            The session context.

        Returns
        -------
        list
        """
        if self.choices is None:
            return []
        values = self.choices(answers, context) if callable(self.choices) else self.choices
        return list(values)


@dataclass(frozen=True)
class Note:
    """Text printed by the CLI (and shown in the GUI) at its place in the workflow.

    Parameters
    ----------
    text : str or callable
        The message, or ``text(answers, context)``.
    when : callable, optional
        ``when(answers, context)``: shown only if true.
    section : str, optional
        GUI section title.
    """

    text: str | Callable[[Answers, Context], str]
    when: Predicate | None = None
    section: str = ''

    def applies(self, answers: Answers, context: Context) -> bool:
        """Whether the note is shown for these answers.

        Parameters
        ----------
        answers : dict
            Answers so far.
        context : Context
            The session context.

        Returns
        -------
        bool
        """
        return self.when is None or bool(self.when(answers, context))

    def message(self, answers: Answers, context: Context) -> str:
        """The text for these answers.

        Parameters
        ----------
        answers : dict
            Answers so far.
        context : Context
            The session context.

        Returns
        -------
        str
        """
        return self.text(answers, context) if callable(self.text) else self.text


Item = Question | Note


@dataclass(frozen=True)
class Workflow:
    """A generator workflow: its questions and the assembly of the configuration.

    Parameters
    ----------
    name : str
        Identifier (``run``, ``acbn0``, ``phonon``, ``elphon``, ``sscha``).
    title : str
        Display name.
    description : str
        One-line description.
    items : sequence of Question or Note
        The questions in CLI order.
    assemble : callable
        ``assemble(answers, context)`` returns the configuration dictionary of the builders.
    overrides : sequence of str, optional
        Command-line options that can answer questions of this workflow.
    """

    name: str
    title: str
    description: str
    items: Sequence[Item]
    assemble: Callable[[Answers, Context], dict]
    overrides: Sequence[str] = ()

    def questions(self) -> list[Question]:
        """The questions of the workflow (notes excluded).

        Returns
        -------
        list of Question
        """
        return [item for item in self.items if isinstance(item, Question)]


def gated(items: Sequence[Item], condition: Predicate) -> tuple[Item, ...]:
    """Copies of ``items`` that also require ``condition``.

    Parameters
    ----------
    items : sequence of Question or Note
        Items with their own ``when``.
    condition : callable
        Extra ``condition(answers, context)``.

    Returns
    -------
    tuple of Question or Note
    """
    from dataclasses import replace

    def combined(own: Predicate | None) -> Predicate:
        if own is None:
            return condition
        return lambda answers, context: condition(answers, context) and own(answers, context)

    return tuple(replace(item, when=combined(item.when)) for item in items)


# ------------------------------------------------------------------------------- parsers
def split_names(text: str) -> list[str]:
    """Comma- or space-separated names, duplicates removed in order.

    Parameters
    ----------
    text : str
        Answer text.

    Returns
    -------
    list of str
    """
    names = [name.strip() for name in text.replace(',', ' ').split() if name.strip()]
    return list(dict.fromkeys(names))


def comma_list(text: str) -> list[str]:
    """Comma-separated items, blanks dropped.

    Parameters
    ----------
    text : str
        Answer text.

    Returns
    -------
    list of str
    """
    return [item.strip() for item in text.split(',') if item.strip()]


def float_list(text: str) -> list[float]:
    """Comma- or space-separated numbers.

    Parameters
    ----------
    text : str
        Answer text.

    Returns
    -------
    list of float

    Raises
    ------
    ValueError
        If an item is not a number.
    """
    try:
        return [float(item) for item in text.replace(',', ' ').split() if item.strip()]
    except ValueError:
        raise ValueError('not a list of numbers: %r' % text) from None


def int_list(text: str) -> list[int]:
    """Comma- or space-separated integers.

    Parameters
    ----------
    text : str
        Answer text.

    Returns
    -------
    list of int

    Raises
    ------
    ValueError
        If an item is not an integer.
    """
    try:
        return [int(item) for item in text.replace(',', ' ').split()]
    except ValueError:
        raise ValueError('not a list of integers: %r' % text) from None


def number(text: str) -> float:
    """A number given as text (the questions that keep a string default).

    Parameters
    ----------
    text : str
        Answer text.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If the text is not a number.
    """
    try:
        return float(text)
    except ValueError:
        raise ValueError('not a number: %r' % text) from None


def optional_number(text: str) -> float | None:
    """A number, or ``None`` for a blank answer.

    Parameters
    ----------
    text : str
        Answer text.

    Returns
    -------
    float or None

    Raises
    ------
    ValueError
        If the text is neither blank nor a number.
    """
    return number(text) if text.strip() else None


def optional_text(text: str) -> str | None:
    """The text, or ``None`` for a blank answer.

    Parameters
    ----------
    text : str
        Answer text.

    Returns
    -------
    str or None
    """
    return text or None


def divisible_by(divisor_of: Callable[[Answers], int], what: str) -> Callable:
    """Validator: a positive multiple of another answer.

    Parameters
    ----------
    divisor_of : callable
        ``divisor_of(answers)``: the required divisor.
    what : str
        Name shown in the message.

    Returns
    -------
    callable
        ``validate(value, answers, context)``.
    """

    def validate(value: int, answers: Answers, context: Context) -> str | None:
        divisor = divisor_of(answers)
        if value > 0 and value % divisor == 0:
            return None
        return '  %s must be a positive multiple of %d.' % (what, divisor)

    return validate
