from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:
    from ffmpeg import types
    from ffmpeg.options import Option


@dataclass(frozen=True)
class File:
    url: str
    options: list[Option] = field(default_factory=list)

    @property
    def option_pairs(self) -> dict[str, Optional[types.Option]]:
        pairs = {}

        for option in self.options:
            pairs.update(option.pair)

        return pairs

    def build(self) -> Iterable[str]:
        raise NotImplementedError()


@dataclass(frozen=True)
class InputFile(File):
    def build(self) -> Iterable[str]:
        for option in self.options:
            yield from option.build()

        yield from ["-i", self.url]


@dataclass(frozen=True)
class OutputFile(File):
    def build(self) -> Iterable[str]:
        for option in self.options:
            yield from option.build()

        yield self.url
