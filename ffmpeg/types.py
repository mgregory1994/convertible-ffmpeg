from __future__ import annotations

from typing import IO, Callable, Iterable, TypeVar, Union


Numeric = Union[int, float]

T = Union[str, Numeric]
Option = Union[Iterable[T], T]

Stream = Union[bytes, IO[bytes]]

Handler = TypeVar("Handler", bound=Callable[..., None])
