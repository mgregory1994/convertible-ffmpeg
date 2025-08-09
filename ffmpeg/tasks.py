from __future__ import annotations

import os
import json

from pathlib import Path
from threading import Lock
from typing import Any
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

from ffmpeg import FFmpeg, FFmpegError, Progress
from ffmpeg.streams import VideoStream, AudioStream, SubtitleStream
from ffmpeg.types import (
    InputSettings,
    GeneralSettings,
    VideoSettings,
    VideoFilters,
    AudioSettings,
    AudioFilters,
    SubtitleSettings,
)


class _Task:
    def __init__(self, output_file: str) -> None:
        self._output_file: OutputMediaFile = OutputMediaFile(output_file)
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
        self._ffmpeg: FFmpeg = FFmpeg()
        self._progress: Progress | None = None

    @property
    def output_file(self) -> OutputMediaFile:
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
    def ffmpeg(self) -> FFmpeg:
        return self._ffmpeg

    @property
    def progress(self) -> Progress | None:
        with self._status_lock:
            return self._progress

    @progress.setter
    def progress(self, progress: Progress) -> None:
        with self._status_lock:
            self._progress = progress


class MediaTask(_Task):
    def __init__(self, input_file: str, output_file: str) -> None:
        super().__init__(output_file)

        self._input_file: InputMediaFile = InputMediaFile(input_file)

    @property
    def input_file(self) -> InputMediaFile:
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

    def get_progress_time_remaining(self) -> float | None:
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

    def execute_ffmpeg(self) -> None:
        try:
            self.is_started = True
            self.ffmpeg.execute()

            self.output_file.populate_media_info()
            self.output_file.populate_streams()
        except FFmpegError as exception:
            self.is_error = True

            raise exception
        finally:
            self.is_done = True


class FolderTask(_Task):
    def __init__(
        self,
        input_file: str,
        output_file: str,
        recursive: bool = False,
        watch_folder: bool = False,
    ) -> None:
        super().__init__(output_file)

        self._media_folder: MediaFolder = MediaFolder(input_file, recursive=recursive)
        self._task_lock: Lock = Lock()
        self._media_tasks: list[MediaTask] = []
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

        self.media_folder.schedule_event_handler(self._event_handler)

    def initialize_media_tasks(self) -> None:
        for input_file in self.media_folder.media_files:
            task = TaskHelper.create_folder_media_task(input_file.path, self)

            if task:
                self.add_media_task(task)

    def initialize_task_queue(self) -> None:
        for task in self.media_tasks:
            self.task_queue.put(task)

    @property
    def media_folder(self) -> MediaFolder:
        return self._media_folder

    @property
    def media_tasks(self) -> list[MediaTask]:
        with self._task_lock:
            return self._media_tasks.copy()

    @property
    def task_queue(self) -> Queue[MediaTask]:
        return self._task_queue

    @property
    def watch_folder(self) -> bool:
        return self._watch_folder

    def add_media_task(self, task: MediaTask) -> None:
        with self._task_lock:
            self._media_tasks.append(task)

    def remove_media_task(self, task: MediaTask) -> None:
        try:
            with self._task_lock:
                self._media_tasks.remove(task)
        except ValueError:
            pass


class _MediaFolderEventHandler(FileSystemEventHandler):
    def __init__(self, folder_task: FolderTask) -> None:
        super().__init__()

        self._folder_task = folder_task

    def on_created(self, event: DirCreatedEvent | FileCreatedEvent) -> None:
        try:
            file_path = str(event.src_path)

            if not Path(file_path).is_dir():
                input_file = InputMediaFile(file_path)

                if input_file.is_video or input_file.is_audio:
                    self._folder_task.media_folder.add_media_file(input_file)

                if self._folder_task.is_started:
                    task = TaskHelper.create_folder_media_task(input_file.path, self._folder_task)

                    if task:
                        self._folder_task.add_media_task(task)
                        self._folder_task.task_queue.put(task)
        except FFmpegError:
            pass

    def on_deleted(self, event: DirDeletedEvent | FileDeletedEvent) -> None:
        path = str(event.src_path)

        for media_file in self._folder_task.media_folder.media_files:
            if path == media_file.path or path == media_file.directory:
                self._folder_task.media_folder.remove_media_file(media_file)


class _MediaFile:
    def __init__(self, file_path: str) -> None:
        self._path = Path(os.fspath(file_path)).resolve()
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


class InputMediaFile(_MediaFile):
    def __init__(self, file_path: str) -> None:
        super().__init__(file_path)

        self.populate_media_info()
        self.populate_streams()


class OutputMediaFile(_MediaFile):
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
    def __init__(self, folder_path: str, recursive: bool = False) -> None:
        self._path = Path(os.fspath(folder_path)).resolve()
        self._is_recursive = recursive
        self._list_lock: Lock = Lock()
        self._media_files: list[InputMediaFile] = []
        self._observer = observers.Observer()

        self._initialize_media_files()

    def _initialize_media_files(self) -> None:
        for root, folders, file_names in os.walk(self._path):
            for name in file_names:
                file_path = os.path.join(root, name)
                input_file = InputMediaFile(file_path)

                if input_file.is_video or input_file.is_audio:
                    self.add_media_file(input_file)

            if not self._is_recursive:
                break

    @property
    def path(self) -> str:
        return str(self._path)

    @property
    def is_recursive(self) -> bool:
        return self._is_recursive

    @property
    def media_files(self) -> list[InputMediaFile]:
        with self._list_lock:
            return self._media_files.copy()

    @property
    def observer(self) -> BaseObserver:
        return self._observer

    @property
    def size(self) -> int:
        size = 0

        for media_file in self.media_files:
            size += media_file.size

        return size

    def add_media_file(self, media_file: InputMediaFile) -> None:
        with self._list_lock:
            self._media_files.append(media_file)

    def remove_media_file(self, media_file: InputMediaFile) -> None:
        try:
            with self._list_lock:
                self._media_files.remove(media_file)
        except ValueError:
            pass

    def schedule_event_handler(self, event_handler: FileSystemEventHandler) -> None:
        self.observer.schedule(event_handler, self.path, recursive=self.is_recursive)


class TaskHelper:
    @staticmethod
    def create_folder_media_task(input_file: str, folder_task: FolderTask) -> MediaTask | None:
        try:
            output_file = TaskHelper._get_media_task_output_file(input_file, folder_task)
            task = MediaTask(input_file, output_file)
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

            return task if task.input_file.is_video or task.input_file.is_audio else None
        except FFmpegError:
            return None

    @staticmethod
    def _get_media_task_output_file(input_file: str, folder_task: FolderTask) -> str:
        input_path = Path(os.fspath(input_file)).resolve()
        output_file = os.path.join(
            folder_task.output_file.directory,
            folder_task.output_file.name,
            (input_path.name + folder_task.output_file.extension),
        )
        return output_file


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
    def initialize_ffmpeg(task: MediaTask) -> None:
        FFmpegHelper._init_input_options(task)
        FFmpegHelper._init_output_options(task)

    @staticmethod
    def _init_input_options(task: MediaTask) -> None:
        if task.input_settings.get("itsoffset"):
            input_settings = task.input_settings.copy()
            input_settings.pop("itsoffset")
            task.ffmpeg.input(task.input_file.path, input_settings)  # type: ignore

        task.ffmpeg.input(task.input_file.path, task.input_settings)  # type: ignore

    @staticmethod
    def _init_output_options(task: MediaTask) -> None:
        output_settings: dict[str, Any] = (
            FFmpegHelper._get_stream_options(task) | task.general_settings
        )
        task.ffmpeg.output(
            task.output_file.path, output_settings, map=FFmpegHelper._get_map_args(task)
        )

    @staticmethod
    def _get_stream_options(task: MediaTask) -> dict[str, Any]:
        ffmpeg_args: dict[str, Any] = {}
        streams = task.input_file.video_streams + task.input_file.audio_streams

        for stream in streams:
            for index in range(len(stream.settings)):
                ffmpeg_args.update(FFmpegHelper._get_stream_args(stream, index))

        return ffmpeg_args

    @staticmethod
    def _get_stream_args(
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
