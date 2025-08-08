from dataclasses import dataclass

from ffmpeg.types import (
    StreamDefinition,
    VideoSettings,
    VideoFilters,
    AudioSettings,
    AudioFilters,
    SubtitleSettings,
)


@dataclass
class _Stream:
    _stream_dict: StreamDefinition

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
