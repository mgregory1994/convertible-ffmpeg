from __future__ import annotations

import os
import json

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TypedDict, ReadOnly, Required, Any
from queue import Queue

from watchdog import observers
from watchdog.events import (
    FileSystemEventHandler,
    FileCreatedEvent,
    FileDeletedEvent,
    DirCreatedEvent,
    DirDeletedEvent,
)
from watchdog.observers.api import BaseObserver

from ffmpeg import FFmpeg, Progress, FFmpegError


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


@dataclass
class _Stream:
    _stream_dict: dict[str, int | str | dict[str, str]]

    @property
    def id(self) -> str:
        return ""

    @property
    def index(self) -> int:
        index = self._stream_dict["index"]

        if isinstance(index, int):
            return index

        raise ValueError

    @property
    def map_index(self) -> str:
        return f"0:{self.index}"

    @property
    def codec_type(self) -> str:
        return str(self._stream_dict["codec_type"])

    @property
    def codec_name(self) -> str:
        return str(self._stream_dict["codec_name"])

    @property
    def language(self) -> str:
        try:
            stream_tags = self._stream_dict["tags"]

            if isinstance(stream_tags, dict):
                language = stream_tags["language"]
            else:
                raise ValueError

            if language == "und":
                raise KeyError

            return language
        except KeyError:
            return "N/A"

    def __hash__(self) -> int:
        return hash((self.index, self.codec_name, self.codec_type))

    def __eq__(self, value) -> bool:
        if isinstance(value, _Stream):
            is_index_eq = self.index == value.index
            is_codec_type_eq = self.codec_type == value.codec_type
            is_codec_name_eq = self.codec_name == value.codec_name

            return is_index_eq and is_codec_type_eq and is_codec_name_eq
        return False


class VideoStream(_Stream):
    def __init__(self, stream_dict: dict[str, int | str | dict[str, str]]):
        super().__init__(stream_dict)

        self._settings: list[VideoSettings] = []
        self._filters: list[VideoFilters] = []
        self._init_settings_and_filters()

    def _init_settings_and_filters(self) -> None:
        self._settings.append({"c": "libx264"})
        self._filters.append({})

    @property
    def settings(self) -> list[VideoSettings]:
        return self._settings

    @property
    def filters(self) -> list[VideoFilters]:
        return self._filters

    @property
    def id(self) -> str:
        return "v"

    @property
    def width(self) -> int:
        width = self._stream_dict["width"]

        if isinstance(width, int):
            return width

        raise ValueError

    @property
    def height(self) -> int:
        height = self._stream_dict["height"]

        if isinstance(height, int):
            return height

        raise ValueError

    @property
    def aspect_ratio(self) -> str:
        aspect_ratio = self._stream_dict["display_aspect_ratio"]

        if isinstance(aspect_ratio, str):
            return aspect_ratio

        raise ValueError

    @property
    def pixel_format(self) -> str:
        pixel_format = self._stream_dict["pix_fmt"]

        if isinstance(pixel_format, str):
            return pixel_format

        raise ValueError

    @property
    def frame_rate(self) -> float:
        frame_rate_value = self._stream_dict["r_frame_rate"]

        if isinstance(frame_rate_value, str):
            values = frame_rate_value.split("/")
            dividend = int(values[0])
            divisor = int(values[1])

            return round(dividend / divisor, 3)

        raise ValueError

    @property
    def field_order(self) -> str:
        field_order = self._stream_dict["field_order"]

        if isinstance(field_order, str):
            return field_order

        raise ValueError


class AudioStream(_Stream):
    def __init__(self, stream_dict: dict[str, int | str | dict[str, str]]):
        super().__init__(stream_dict)

        self._settings: list[AudioSettings] = []
        self._filters: list[AudioFilters] = []
        self._init_settings_and_filters()

    def _init_settings_and_filters(self) -> None:
        self.settings.append({"c": "aac"})
        self.filters.append({})

    @property
    def settings(self) -> list[AudioSettings]:
        return self._settings

    @property
    def filters(self) -> list[AudioFilters]:
        return self._filters

    @property
    def id(self) -> str:
        return "a"

    @property
    def channels(self) -> int:
        channels = self._stream_dict["channels"]

        if isinstance(channels, int):
            return channels

        raise ValueError

    @property
    def channel_layout(self) -> str:
        channel_layout = self._stream_dict["channel_layout"]

        if isinstance(channel_layout, str):
            return channel_layout

        raise ValueError

    @property
    def sample_rate(self) -> str:
        sample_rate = self._stream_dict["sample_rate"]

        if isinstance(sample_rate, str):
            return sample_rate

        raise ValueError

    def __str__(self):
        return ", ".join([self.codec_name, str(self.channels)])

    def __repr__(self):
        return ", ".join(
            [self.language, str(self.channels), self.channel_layout, self.sample_rate]
        )


class SubtitleStream(_Stream):
    def __init__(self, stream_dict: dict[str, int | str | dict[str, str]]):
        super().__init__(stream_dict)

        self._settings: list[SubtitleSettings] = []

    @property
    def id(self) -> str:
        return "s"

    @property
    def settings(self) -> list[SubtitleSettings]:
        return self._settings


class _MediaFile:
    def __init__(self, path: Path) -> None:
        self._path = path.resolve()
        self._video_streams: list[VideoStream] = []
        self._audio_streams: list[AudioStream] = []
        self._subtitle_streams: list[SubtitleStream] = []

    @property
    def name(self) -> str:
        return self._path.stem

    @property
    def directory(self) -> str:
        return os.path.dirname(self.path)

    @property
    def extension(self) -> str:
        return self._path.suffix

    @property
    def path(self) -> str:
        return str(self._path)

    @property
    def format_name(self) -> str:
        return self._media_info["format"]["format_long_name"]

    @property
    def duration(self) -> float:
        return float(self._media_info["format"]["duration"])

    @property
    def size(self) -> int:
        return int(self._media_info["format"]["size"])

    @property
    def bitrate(self) -> int:
        return int(self._media_info["format"]["bit_rate"])

    @property
    def num_streams(self) -> int:
        return self._media_info["format"]["nb_streams"]

    @property
    def is_video(self) -> bool:
        if self.video_streams:
            return self.video_streams[0].codec_name != "mjpeg"
        return False

    @property
    def is_audio(self) -> bool:
        return bool(self.audio_streams) and not self.is_video

    @property
    def video_streams(self) -> list[VideoStream]:
        return self._video_streams

    @property
    def audio_streams(self) -> list[AudioStream]:
        return self._audio_streams

    @property
    def subtitle_streams(self) -> list[SubtitleStream]:
        return self._subtitle_streams

    def populate_media_info(self):
        ffprobe = FFmpeg(executable="ffprobe").input(
            self.path,
            print_format="json",
            show_streams=None,
            show_format=None,
        )
        self._media_info = json.loads(ffprobe.execute())

    def populate_streams(self):
        for stream in self._media_info["streams"]:
            if stream["codec_type"] == "video":
                self._video_streams.append(VideoStream(stream))
            elif stream["codec_type"] == "audio":
                self._audio_streams.append(AudioStream(stream))
            elif stream["codec_type"] == "subtitle":
                self._subtitle_streams.append(SubtitleStream(stream))


class InputFile(_MediaFile):
    def __init__(self, path: Path) -> None:
        super().__init__(path)

        self.populate_media_info()
        self.populate_streams()


class OutputFile(_MediaFile):
    @property
    def name(self) -> str:
        return super().name

    @name.setter
    def name(self, name: str):
        path = Path(os.path.join(self.directory, name + self.extension))
        self._path = path.resolve()

    @property
    def directory(self) -> str:
        return super().directory

    @directory.setter
    def directory(self, directory: str):
        path = Path(os.path.join(directory, self.name + self.extension))
        self._path = path

    @property
    def extension(self) -> str:
        return super().extension

    @extension.setter
    def extension(self, extension: str):
        path = Path(os.path.join(self.directory, self.name + extension))
        self._path = path.resolve()


class MediaFolder:
    def __init__(self, path: Path, recursive: bool = False) -> None:
        self._path = path.resolve()
        self._recursive = recursive
        self._list_lock: Lock = Lock()
        self._media_tasks: list[MediaTask] = []
        self._observer = observers.Observer()

    @property
    def path(self) -> str:
        return str(self._path)

    @property
    def recursive(self) -> bool:
        return self._recursive

    @property
    def media_tasks(self) -> list[MediaTask]:
        with self._list_lock:
            return self._media_tasks.copy()

    @property
    def observer(self) -> BaseObserver:
        return self._observer

    @property
    def size(self) -> int:
        size = 0

        for task in self.media_tasks:
            size += task.input_file.size

        return size

    def add_media_task(self, task: MediaTask) -> None:
        with self._list_lock:
            self._media_tasks.append(task)

    def remove_media_task(self, task: MediaTask) -> None:
        try:
            with self._list_lock:
                self._media_tasks.remove(task)
        except ValueError:
            pass

    def schedule_event_handler(self, event_handler: FileSystemEventHandler) -> None:
        self.observer.schedule(event_handler, self.path, recursive=self.recursive)


class _Task:
    def __init__(self, output_path: Path) -> None:
        self._output_file: OutputFile = OutputFile(output_path)
        self.input_settings: InputSettings = {"y": None, "hwaccel": "auto"}
        self.general_settings: GeneralSettings = {}
        self.is_video_offset: bool = False
        self.is_audio_offset: bool = False
        self._status_lock: Lock = Lock()
        self._is_started: bool = False
        self._is_stopped: bool = False
        self._is_paused: bool = False
        self._is_done: bool = False
        self._is_error: bool = False
        self._progress: Progress | None = None

    @property
    def output_file(self) -> OutputFile:
        return self._output_file

    @property
    def is_started(self) -> bool:
        with self._status_lock:
            return self._is_started

    @is_started.setter
    def is_started(self, is_enabled: bool) -> None:
        with self._status_lock:
            self._is_started = is_enabled

    @property
    def is_stopped(self) -> bool:
        with self._status_lock:
            return self._is_stopped

    @is_stopped.setter
    def is_stopped(self, is_enabled: bool) -> None:
        with self._status_lock:
            self._is_stopped = is_enabled

    @property
    def is_paused(self) -> bool:
        with self._status_lock:
            return self._is_paused

    @is_paused.setter
    def is_paused(self, is_enabled: bool) -> None:
        with self._status_lock:
            self._is_paused = is_enabled

    @property
    def is_done(self) -> bool:
        with self._status_lock:
            return self._is_done

    @is_done.setter
    def is_done(self, is_enabled: bool) -> None:
        with self._status_lock:
            self._is_done = is_enabled

    @property
    def is_error(self) -> bool:
        with self._status_lock:
            return self._is_error

    @is_error.setter
    def is_error(self, is_enabled: bool) -> None:
        with self._status_lock:
            self._is_error = is_enabled

    @property
    def progress(self) -> Progress | None:
        with self._status_lock:
            return self._progress

    @progress.setter
    def progress(self, progress: Progress) -> None:
        with self._status_lock:
            self._progress = progress


class MediaTask(_Task):
    def __init__(self, input_path: Path, output_path: Path) -> None:
        super().__init__(output_path)

        self._input_file: InputFile = InputFile(input_path)

    @property
    def input_file(self) -> InputFile:
        return self._input_file

    def get_video_stream(self, index: int) -> VideoStream | None:
        try:
            return self.input_file.video_streams[index]
        except IndexError:
            return None

    def get_audio_stream(self, index: int) -> AudioStream | None:
        try:
            return self.input_file.audio_streams[index]
        except IndexError:
            return None

    def get_subtitle_stream(self, index: int) -> SubtitleStream | None:
        try:
            return self.input_file.subtitle_streams[index]
        except IndexError:
            return None

    def get_video_settings(
        self, stream: VideoStream, index: int
    ) -> tuple[VideoSettings, VideoFilters] | None:
        try:
            settings = stream.settings[index]
            filters = stream.filters[index]

            return settings, filters
        except (IndexError, ValueError):
            return None

    def get_audio_settings(
        self, stream: AudioStream, index: int
    ) -> tuple[AudioSettings, AudioFilters] | None:
        try:
            settings = stream.settings[index]
            filters = stream.filters[index]

            return settings, filters
        except (IndexError, ValueError):
            return None

    def get_subtitle_settings(
        self, stream: SubtitleStream, index: int
    ) -> SubtitleSettings | None:
        try:
            return stream.settings[index]
        except (IndexError, ValueError):
            return None

    def set_video_settings(
        self,
        stream: VideoStream,
        index: int,
        settings: VideoSettings | None = None,
        filters: VideoFilters | None = None,
    ) -> None:
        if settings is not None:
            stream.settings[index] = settings

        if filters is not None:
            stream.filters[index] = filters

    def set_audio_settings(
        self,
        stream: AudioStream,
        index: int,
        settings: AudioSettings | None = None,
        filters: AudioFilters | None = None,
    ) -> None:
        if settings is not None:
            stream.settings[index] = settings

        if filters is not None:
            stream.filters[index] = filters

    def set_subtitle_settings(
        self, stream: SubtitleStream, index: int, settings: SubtitleSettings
    ) -> None:
        if settings is not None:
            stream.settings[index] = settings

    def add_video_settings(
        self, stream: VideoStream, settings: VideoSettings, filters: VideoFilters = {}
    ) -> int:
        stream.settings.append(settings)
        stream.filters.append(filters)

        return stream.settings.index(settings)

    def add_audio_settings(
        self, stream: AudioStream, settings: AudioSettings, filters: AudioFilters = {}
    ) -> int:
        stream.settings.append(settings)
        stream.filters.append(filters)

        return stream.settings.index(settings)

    def add_subtitle_settings(
        self, stream: SubtitleStream, settings: SubtitleSettings
    ) -> int:
        stream.settings.append(settings)

        return stream.settings.index(settings)

    def remove_video_settings(self, stream: VideoStream, index: int):
        stream.settings.pop(index)
        stream.filters.pop(index)

    def remove_audio_settings(self, stream: AudioStream, index: int):
        stream.settings.pop(index)
        stream.filters.pop(index)

    def remove_subtitle_settings(self, stream: SubtitleStream, index: int):
        stream.settings.pop(index)

    def get_progress_percent(self) -> float | None:
        if self.progress is None:
            return None

        trim_duration = self.general_settings.get("to")

        if trim_duration:
            duration = trim_duration
        else:
            duration = self.input_file.duration

        time_position = self.progress.time.total_seconds()
        progress_percent = (time_position / float(duration)) * 100.0

        return round(progress_percent, 2)

    def get_time_remaining(self) -> float | None:
        try:
            if self.progress is None:
                return None

            trim_duration = self.general_settings.get("to")

            if trim_duration:
                duration = trim_duration
            else:
                duration = self.input_file.duration

            time_position = self.progress.time.total_seconds()
            time_difference = duration - time_position
            time_remaining = time_difference / self.progress.speed

            return round(time_remaining, 2)
        except ZeroDivisionError:
            return None


class FolderTask(_Task):
    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        recursive: bool = False,
        watch_folder: bool = False,
    ) -> None:
        super().__init__(output_path)

        self._media_folder: MediaFolder = MediaFolder(input_path, recursive=recursive)
        self._task_queue: Queue[MediaTask] = Queue()
        self._event_handler: _MediaFolderEventHandler = _MediaFolderEventHandler(self)
        self._watch_folder: bool = watch_folder
        self.video_settings: VideoSettings = {"c": "libx264"}
        self.video_filters: VideoFilters = {}
        self.audio_settings: AudioSettings = {"c": "aac"}
        self.audio_filters: AudioFilters = {}
        self.subtitle_settings: SubtitleSettings = {"c": "copy"}
        self.is_no_video: bool = False
        self.is_no_audio: bool = False
        self.is_no_subtitle: bool = False
        self.is_crop_detect: bool = False

        self.populate_media_tasks()
        self.media_folder.schedule_event_handler(self._event_handler)

    @property
    def media_folder(self) -> MediaFolder:
        return self._media_folder

    @property
    def task_queue(self) -> Queue[MediaTask]:
        return self._task_queue

    @property
    def watch_folder(self) -> bool:
        return self._watch_folder

    def populate_media_tasks(self) -> None:
        pass


class _MediaFolderEventHandler(FileSystemEventHandler):
    def __init__(self, folder_task: FolderTask) -> None:
        super().__init__()

        self._folder_task = folder_task

    def on_created(self, event: DirCreatedEvent | FileCreatedEvent) -> None:
        try:
            file_path = str(event.src_path)

            if not Path(file_path).is_dir():
                task = TaskHelper.create_folder_media_task(Path(file_path), self._folder_task)
                self._folder_task.media_folder.add_media_task(task)
        except FFmpegError:
            pass

    def on_deleted(self, event: DirDeletedEvent | FileDeletedEvent) -> None:
        path = str(event.src_path)

        for task in self._folder_task.media_folder.media_tasks:
            if path == task.input_file.path or path == task.input_file.directory:
                self._folder_task.media_folder.remove_media_task(task)


class TaskHelper:
    @staticmethod
    def create_folder_media_task(input_path: Path, folder_task: FolderTask) -> MediaTask:
        output_path = os.path.join(
            folder_task.output_file.directory,
            folder_task.output_file.name,
            input_path.name + folder_task.output_file.extension,
        )
        task = MediaTask(input_path, Path(output_path))
        task.input_settings = folder_task.input_settings
        task.general_settings = folder_task.general_settings
        video_stream = task.get_video_stream(0)
        video_filters = folder_task.video_filters.copy()
        audio_stream = task.get_audio_stream(0)
        subtitle_stream = task.get_subtitle_stream(0)

        if folder_task.is_crop_detect:
            video_filters["crop"] = FFmpegHelper.get_crop_detect(task)

        if video_stream and folder_task.is_no_video:
            task.remove_video_settings(video_stream, 0)
        elif video_stream:
            task.set_video_settings(
                video_stream, 0, folder_task.video_settings, video_filters
            )

        if audio_stream and folder_task.is_no_audio:
            task.remove_audio_settings(audio_stream, 0)
        elif audio_stream:
            task.set_audio_settings(
                audio_stream, 0, folder_task.audio_settings, folder_task.audio_filters
            )

        if subtitle_stream and folder_task.is_no_subtitle:
            task.remove_subtitle_settings(subtitle_stream, 0)
        elif subtitle_stream:
            task.set_subtitle_settings(
                subtitle_stream, 0, folder_task.subtitle_settings
            )

        return task


class FFmpegHelper:
    BITRATE_ARGS: set[str] = {"b", "minrate", "maxrate", "bufsize"}
    CROP_DETECT_SPLITS: int = 4

    @staticmethod
    def get_crop_detect(task: MediaTask) -> tuple[int, int, int, int]:
        crop_values: list[int] = []

        for split in range(1, FFmpegHelper.CROP_DETECT_SPLITS):
            start_time = task.input_file.duration * split
            start_time /= FFmpegHelper.CROP_DETECT_SPLITS
            ffmpeg = (
                FFmpeg()
                .input(task.input_file.path)
                .option("ss", start_time)
                .output("pipe:1", vf="cropdetect", to=5, f="null")
            )

            @ffmpeg.on("progress")
            def capture_crop_detect(progress: Progress):
                if progress.crop[0] <= 0:
                    return

                if not crop_values:
                    crop_values.extend(list(progress.crop))

                width, height, x, y = zip(crop_values, progress.crop)
                crop_values[0] = max(width)
                crop_values[1] = max(height)
                crop_values[2] = min(x)
                crop_values[3] = min(y)

            ffmpeg.execute()

        width, height, x, y = crop_values
        return width, height, x, y

    @staticmethod
    def start_ffmpeg(task: MediaTask, ffmpeg: FFmpeg) -> None:
        try:
            task.is_started = True
            ffmpeg.execute()

            task.output_file.populate_media_info()
            task.output_file.populate_streams()
        except FFmpegError as exception:
            task.is_error = True

            raise exception
        finally:
            task.is_done = True

    @staticmethod
    def generate_ffmpeg(task: MediaTask) -> FFmpeg:
        ffmpeg = FFmpeg()
        FFmpegHelper._set_ffmpeg_input_options(task, ffmpeg)
        FFmpegHelper._set_ffmpeg_output(task, ffmpeg)

        return ffmpeg

    @staticmethod
    def _set_ffmpeg_input_options(task: MediaTask, ffmpeg: FFmpeg) -> None:
        if task.input_settings.get("itsoffset"):
            input_settings = task.input_settings.copy()
            input_settings.pop("itsoffset")
            ffmpeg.input(task.input_file.path, input_settings)  # type: ignore

        ffmpeg.input(task.input_file.path, task.input_settings)  # type: ignore

    @staticmethod
    def _set_ffmpeg_output(task: MediaTask, ffmpeg: FFmpeg) -> None:
        output_settings: dict[str, Any] = (
            FFmpegHelper._get_stream_settings(task) | task.general_settings
        )
        ffmpeg.output(
            task.output_file.path, output_settings, map=FFmpegHelper._get_map_args(task)
        )

    @staticmethod
    def _get_stream_settings(task: MediaTask) -> dict[str, Any]:
        ffmpeg_args: dict[str, Any] = {}
        streams = task.input_file.video_streams + task.input_file.audio_streams

        for stream in streams:
            for index in range(len(stream.settings)):
                ffmpeg_args.update(FFmpegHelper._get_stream_ffmpeg_args(stream, index))

        return ffmpeg_args

    @staticmethod
    def _get_stream_ffmpeg_args(
        stream: VideoStream | AudioStream, index: int
    ) -> dict[str, Any]:
        stream_settings = stream.settings[index].items()
        stream_filters = stream.filters[index].items()
        stream_args: dict[str, Any] = {}
        filter_args: list[str] = []

        for arg, value in stream_settings:
            if isinstance(value, tuple):
                arg_pair = ",".join(map(str, value))
            elif arg in FFmpegHelper.BITRATE_ARGS:
                arg_pair = f"{value}k"
            else:
                arg_pair = value

            stream_args[f"{arg}:{stream.id}:{index}"] = arg_pair

        for arg, value in stream_filters:
            if isinstance(value, tuple):
                arg_pair = f"{arg}={':'.join(map(str, value))}"
            elif value is None:
                arg_pair = arg
            else:
                arg_pair = f"{arg}={value}"

            filter_args.append(arg_pair)

        if filter_args:
            stream_args[f"filter:{stream.id}:{index}"] = ",".join(filter_args)

        return stream_args

    @staticmethod
    def _get_map_args(task: MediaTask) -> list[str]:
        map_args: list[str] = []

        for stream in task.input_file.video_streams:
            for settings in stream.settings:
                map_args.append(f"{int(task.is_video_offset)}:{stream.index}")

        for stream in task.input_file.audio_streams:
            for settings in stream.settings:
                map_args.append(f"{int(task.is_audio_offset)}:{stream.index}")

        return map_args
