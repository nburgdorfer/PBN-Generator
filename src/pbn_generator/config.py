from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_origin, get_type_hints


@dataclass
class InputConfig:
    image: str | Path
    palette: str | Path | None = None


@dataclass
class CanvasConfig:
    size: tuple[float, float] = (8.0, 10.0)
    dpi: int = 150


@dataclass
class ImagePreprocessorConfig:
    name: str = "bilateral"
    filter_size: int = 7


@dataclass
class ClusterLabelerConfig:
    name: str = "kmeans"
    allow_unused_palette_colors: bool = True


@dataclass
class LabelProcessorConfig:
    name: str = "morphological_open_merge"
    min_area: float = 0.005
    min_width_inches: float = 0.10
    merge_passes: int = 1
    label_smooth_size: int = 3
    label_smooth_passes: int = 2


@dataclass
class PaletteGeneratorConfig:
    num_colors: int = 25


@dataclass
class OutputConfig:
    directory: str | Path = "output"
    border_thickness: int = 0
    write_intermediate: bool = True


@dataclass
class PBNConfig:
    input: InputConfig
    canvas: CanvasConfig = field(default_factory=CanvasConfig)
    image_preprocessor: ImagePreprocessorConfig = field(
        default_factory=ImagePreprocessorConfig
    )
    cluster_labeler: ClusterLabelerConfig = field(default_factory=ClusterLabelerConfig)
    label_processor: LabelProcessorConfig = field(default_factory=LabelProcessorConfig)
    palette_generator: PaletteGeneratorConfig = field(
        default_factory=PaletteGeneratorConfig
    )
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "PBNConfig":
        return _dataclass_from_mapping(cls, values)


def _dataclass_from_mapping(dataclass_type: type[Any], values: dict[str, Any]) -> Any:
    kwargs = {}
    type_hints = get_type_hints(dataclass_type)
    for field in fields(dataclass_type):
        if field.name not in values:
            continue

        value = values[field.name]
        field_type = type_hints.get(field.name, field.type)
        if is_dataclass(field_type) and isinstance(value, dict):
            kwargs[field.name] = _dataclass_from_mapping(field_type, value)
        elif get_origin(field_type) is tuple and isinstance(value, list):
            kwargs[field.name] = tuple(value)
        else:
            kwargs[field.name] = value

    return dataclass_type(**kwargs)
