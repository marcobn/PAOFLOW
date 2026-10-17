"""Back-compatibility shim: ``PAOFLOW.qo_analysis`` moved to :mod:`PAOFLOW.agentic.qo_analysis`.

Imports of the old paths,

    from PAOFLOW.qo_analysis import QuantumOscillationAnalysis
    from PAOFLOW.qo_analysis.field_geometry import ...

keep working: the package and each submodule are aliased to the new location
with a ``FutureWarning`` (shown by default, unlike ``DeprecationWarning``).
"""

from __future__ import annotations

import sys as _sys
import warnings as _warnings
from importlib import import_module as _import_module
from importlib.abc import Loader as _Loader
from importlib.abc import MetaPathFinder as _MetaPathFinder
from importlib.util import spec_from_loader as _spec_from_loader

_OLD = __name__  # 'PAOFLOW.qo_analysis'
_NEW = 'PAOFLOW.agentic.qo_analysis'

_warnings.warn(f'{_OLD} is deprecated; import from {_NEW} instead', FutureWarning, stacklevel=2)


class _AliasLoader(_Loader):
    def __init__(self, target_name: str) -> None:
        self._target_name = target_name

    def create_module(self, spec):
        module = _import_module(self._target_name)
        _sys.modules[spec.name] = module
        return module

    def exec_module(self, module):  # pragma: no cover - aliased module already loaded
        return None


class _AliasFinder(_MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if not fullname.startswith(_OLD + '.'):
            return None
        new_name = _NEW + fullname[len(_OLD) :]
        _warnings.warn(
            f'{fullname} is deprecated; import from {new_name} instead', FutureWarning, stacklevel=2
        )
        return _spec_from_loader(fullname, _AliasLoader(new_name))


if not any(isinstance(f, _AliasFinder) for f in _sys.meta_path):
    _sys.meta_path.insert(0, _AliasFinder())

_target = _import_module(_NEW)
__all__ = list(getattr(_target, '__all__', ()))


def __getattr__(name):
    return getattr(_target, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_target)))
