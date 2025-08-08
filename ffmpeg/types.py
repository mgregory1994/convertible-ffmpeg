from __future__ import annotations

import asyncio
from typing import IO, Callable, Iterable, TypeVar, Union, ReadOnly, Required, TypedDict


Numeric = Union[int, float]

T = Union[str, Numeric]
Option = Union[Iterable[T], T]

Stream = Union[bytes, IO[bytes]]
StreamDefinition = dict[str, int | str | dict[str, str]]
AsyncStream = Union[bytes, asyncio.StreamReader]

Handler = TypeVar("Handler", bound=Callable[..., None])


class InputSettings(TypedDict, total=False):
    y: ReadOnly[None]
    n: ReadOnly[None]
    hwaccel: ReadOnly[str]
    ss: float
    display_rotation: int
    display_vflip: None
    display_hflip: None
    itsoffset: float


class GeneralSettings(TypedDict, total=False):
    to: float
    movflags: str


class VideoSettings(
    TypedDict(
        "codec_settings",
        {
            "c": Required[ReadOnly[str]],
            "crf": float,
            "qp": float,
            "b": int,
            "minrate": int,
            "maxrate": int,
            "bufsize": int,
            "preset": str,
            "tune": str,
            "nal-hrd": str,
            "pass": int,
            "split_encode_mode": str,
            "lookahead_level": str,
            "intra-refresh": bool,
            "ldkfs": bool,
            "dpb_size": bool,
            "b_ref_mode": str,
            "weighted_pred": bool,
            "aq-strength": int,
            "strict_gop": bool,
            "nonref_p": bool,
            "zerolatency": bool,
            "temporal-aq": bool,
            "spatial-aq": bool,
            "b_adapt": bool,
            "forced-idr": bool,
            "no-scenecut": bool,
            "qp_cr_offset": int,
            "qp_cb_offset": int,
            "cq": int,
            "rc-lookahead": int,
            "surfaces": int,
            "tile-columns": int,
            "tile-rows": int,
            "highbitdepth": bool,
            "multipass": str,
            "rc": str,
            "tier": int,
            "level": str,
            "constrained-encoding": bool,
            "max_slice_size": int,
            "single-slice-intra-refresh": bool,
            "coder": str,
            "init_qpI": int,
            "init_qpP": int,
            "init_qpB": int,
            "bluray-compat": bool,
            "aud": bool,
            "cbr": bool,
            "2pass": bool,
            "unidir_b": bool,
            "tf_level": bool,
            "dolbyvision": bool,
            "svtav1-params": str,
            "tiles": str,
            "tile_groups": int,
            "blbrc": bool,
            "rc_mode": str,
            "max_frame_size": int,
            "low_power": bool,
            "async_depth": int,
            "b_depth": int,
            "idr_interval": int,
            "quality": int,
            "loop_filter_level": int,
            "loop_filter_sharpness": int,
            "noise_reduction": int,
            "sc_threshold": int,
            "chromaoffset": int,
            "b_strategy": int,
            "motion-est": str,
            "slice-max-size": int,
            "direct-pred": str,
            "partitions": str,
            "cplxblur": bool,
            "deblock": bool,
            "mbtree": bool,
            "fast-pskip": bool,
            "8x8dct": bool,
            "mixed-refs": bool,
            "b-pyramid": str,
            "b-bias": int,
            "weightp": str,
            "weightb": bool,
            "psr-rd": tuple[float, float],
            "psy": bool,
            "aq-mode": str,
            "crf_max": float,
            "wpredp": bool,
            "fastfirstpass": bool,
            "passlogfile": str,
            "stats": str,
            "sharpness": int,
            "rc_lookahead": int,
            "arnr_max_frames": int,
            "arnr_strength": int,
            "arnr_type": str,
            "speed": int,
            "cpu-used": int,
            "auto-alt-ref": bool,
            "noise-sensitivity": int,
            "drop-threshold": int,
            "static-thresh": int,
            "max-intra-rate": int,
            "deadline": str,
            "lag-in-frames": int,
            "min-gf-interval": int,
            "enable-tpl": bool,
            "tune-content": str,
            "row-mt": bool,
            "frame-parallel": bool,
            "lossless": bool,
            "tile_rows": int,
            "tile_cols": int,
            "look_ahead_depth": int,
            "max_frame_size_i": int,
            "max_frame_size_b": int,
            "ext_brc": bool,
            "adaptive_i": bool,
            "adaptive_b": bool,
            "look_ahead_downsampling": str,
            "look_ahead": bool,
            "scenario": str,
            "p_strategy": int,
            "min_qp_i": int,
            "max_qp_i": int,
            "min_qp_p": int,
            "max_qp_p": int,
            "min_qp_b": int,
            "max_qp_b": int,
            "rdo": bool,
        },
        total=False,
    )
): ...


class VideoFilters(TypedDict, total=False):
    crop: tuple[int, int, int, int]  # width, height, x pad, y pad
    deband: None
    deflicker: None
    dejudder: None
    delogo: tuple[bool, int, int, int, int]
    deshake: None
    framerate: float
    pixelize: tuple[int, int]
    transpose: str
    scale: tuple[int, int]  # width, height
    thumbnail: None
    vflip: None
    hflip: None
    subtitles: str
    yadif: None
    bwdif: None
    estdif: None
    kerndeint: None
    linblenddeint: None
    cubicipoldeint: None
    mediandeint: None
    ffmpegdeint: None
    w3fdif: None


class AudioSettings(TypedDict, total=False):
    c: Required[ReadOnly[str]]
    b: int
    ac: int
    ar: int


class AudioFilters(TypedDict, total=False):
    adeclick: None
    adeclip: None
    loudnorm: None
    dialoguenhance: str
    dynaudnorm: None
    extrastereo: float
    volume: float


class SubtitleSettings(TypedDict, total=False):
    c: Required[ReadOnly[str]]
