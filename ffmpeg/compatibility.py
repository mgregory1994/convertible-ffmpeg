from __future__ import annotations

from typing import Optional

from ffmpeg import FFmpeg, types
from ffmpeg.errors import FFmpegError


def _get_test_ffmpeg() -> FFmpeg:
    return FFmpeg().input("smptebars", f="lavfi")


def is_h264_nvenc_supported() -> bool:
    try:
        options: dict[str, Optional[types.Option]] = {
            "c:v": "h264_nvenc",
            "vframes": "1",
            "f": "null",
        }
        _get_test_ffmpeg().output("-", options).execute()

        return True
    except FFmpegError:
        return False


def is_hevc_nvenc_supported() -> bool:
    try:
        options: dict[str, Optional[types.Option]] = {
            "c:v": "hevc_nvenc",
            "vframes": "1",
            "f": "null",
        }
        _get_test_ffmpeg().output("-", options).execute()

        return True
    except FFmpegError:
        return False


def is_options_supported(options: dict[str, Optional[types.Option]]) -> bool:
    try:
        options = options.copy()
        options.update({
            "vframes": "1",
            "f": "null"
        })
        _get_test_ffmpeg().output("-", options).execute()

        return True
    except FFmpegError:
        return False
