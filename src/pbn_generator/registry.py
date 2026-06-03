from __future__ import annotations

from pbn_generator.config import (
    ClusterLabelerConfig,
    ImagePreprocessorConfig,
    LabelProcessorConfig,
)
from pbn_generator.label_processors import (
    DistanceTransformMergeProcessor,
    LabelProcessor,
    MorphologicalOpenMergeProcessor,
    SmoothMergeProcessor,
)
from pbn_generator.labelers import ClusterLabeler, KMeansLabeler
from pbn_generator.preprocessors import BilateralPreprocessor, ImagePreprocessor


IMAGE_PREPROCESSORS = {
    "bilateral": BilateralPreprocessor,
}

CLUSTER_LABELERS = {
    "kmeans": KMeansLabeler,
}

LABEL_PROCESSORS = {
    "distance_transform_merge": DistanceTransformMergeProcessor,
    "morphological_open_merge": MorphologicalOpenMergeProcessor,
    "smooth_merge": SmoothMergeProcessor,
}


def build_image_preprocessor(
    config: ImagePreprocessorConfig,
) -> ImagePreprocessor:
    preprocessor_type = _get_registered(
        IMAGE_PREPROCESSORS,
        config.name,
        "image_preprocessor",
    )
    return preprocessor_type(filter_size=config.filter_size)


def build_cluster_labeler(config: ClusterLabelerConfig) -> ClusterLabeler:
    labeler_type = _get_registered(
        CLUSTER_LABELERS,
        config.name,
        "cluster_labeler",
    )
    return labeler_type(allow_unused_colors=config.allow_unused_palette_colors)


def build_label_processor(
    config: LabelProcessorConfig,
    dpi: int,
) -> LabelProcessor:
    processor_type = _get_registered(
        LABEL_PROCESSORS,
        config.name,
        "label_processor",
    )
    if processor_type is SmoothMergeProcessor:
        return processor_type(
            min_area=config.min_area,
            dpi=dpi,
            merge_passes=config.merge_passes,
            label_smooth_size=config.label_smooth_size,
            label_smooth_passes=config.label_smooth_passes,
        )

    return processor_type(
        min_width_px=config.min_width_px,
        merge_passes=config.merge_passes,
        label_smooth_size=config.label_smooth_size,
        label_smooth_passes=config.label_smooth_passes,
    )


def _get_registered(registry: dict[str, type], name: str, config_name: str) -> type:
    try:
        return registry[name]
    except KeyError as error:
        options = ", ".join(sorted(registry))
        raise ValueError(
            f"Unknown {config_name}.name {name!r}. Available options: {options}."
        ) from error
