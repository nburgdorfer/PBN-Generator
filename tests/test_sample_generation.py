from pathlib import Path

import numpy as np
from PIL import Image

from pbn_generator import (
    BilateralPreprocessor,
    CanvasConfig,
    ClusterLabelerConfig,
    DistanceTransformMergeProcessor,
    ImagePreprocessorConfig,
    InputConfig,
    KMeansLabeler,
    LabelProcessorConfig,
    MorphologicalOpenMergeProcessor,
    OutputConfig,
    PBNConfig,
    PBNGenerator,
    PaletteGeneratorConfig,
    SmoothMergeProcessor,
)
from pbn_generator.registry import build_label_processor


def test_sample_data_and_palette_generate_manual_outputs():
    repo_root = Path(__file__).resolve().parents[1]
    output_directory = Path("output")
    output_stem = "hills"
    config = PBNConfig(
        input=InputConfig(
            image="data/hills.jpg",
            palette="palettes/palette_1.yaml",
        ),
        canvas=CanvasConfig(size=(8.0, 10.0), dpi=150),
        image_preprocessor=ImagePreprocessorConfig(name="bilateral", filter_size=11),
        cluster_labeler=ClusterLabelerConfig(
            name="kmeans",
            allow_unused_palette_colors=True,
        ),
        label_processor=LabelProcessorConfig(
            name="smooth_merge",
            min_area=0.05,
            merge_passes=2,
            label_smooth_size=3,
            label_smooth_passes=2,
        ),
        output=OutputConfig(directory=output_directory, write_intermediate=True),
    )

    generator = PBNGenerator(config, base_dir=repo_root)

    assert isinstance(generator.image_preprocessor, BilateralPreprocessor)
    assert (
        generator.image_preprocessor.filter_size
        == config.image_preprocessor.filter_size
    )
    assert isinstance(generator.cluster_labeler, KMeansLabeler)
    assert (
        generator.cluster_labeler.allow_unused_colors
        == config.cluster_labeler.allow_unused_palette_colors
    )
    assert isinstance(generator.label_processor, SmoothMergeProcessor)
    assert generator.label_processor.min_area == config.label_processor.min_area
    assert generator.label_processor.dpi == config.canvas.dpi
    assert (
        generator.label_processor.merge_passes
        == config.label_processor.merge_passes
    )
    assert (
        generator.label_processor.label_smooth_size
        == config.label_processor.label_smooth_size
    )
    assert (
        generator.label_processor.label_smooth_passes
        == config.label_processor.label_smooth_passes
    )

    result = generator.run()
    expected_output_directory = repo_root / output_directory

    assert result.colored_path == expected_output_directory / (
        f"{output_stem}_colored.png"
    )
    assert result.outline_path == expected_output_directory / (
        f"{output_stem}_outline.png"
    )
    assert result.palette_path == expected_output_directory / (
        f"{output_stem}_palette.png"
    )
    assert result.intermediate_path == expected_output_directory / (
        f"{output_stem}{generator.image_preprocessor.output_suffix}"
    )
    assert result.colored_path.exists()
    assert result.outline_path.exists()
    assert result.palette_path.exists()
    assert result.intermediate_path.exists()

    colored = np.asarray(Image.open(result.colored_path).convert("RGB"))
    assert not np.any(np.all(colored == [0, 0, 0], axis=2))


def test_distance_transform_processor_merges_narrow_regions():
    labels = np.zeros((7, 7), dtype=np.int32)
    labels[:, 3] = 1
    colors_rgb = np.array(
        [
            [255, 255, 255],
            [255, 0, 0],
        ],
        dtype=np.uint8,
    )
    processor = DistanceTransformMergeProcessor(
        min_width_px=3,
        merge_passes=1,
        label_smooth_size=1,
        label_smooth_passes=0,
    )

    processed = processor.process(labels, colors_rgb)

    assert np.all(processed == 0)


def test_morphological_open_processor_merges_narrow_regions():
    labels = np.zeros((7, 7), dtype=np.int32)
    labels[:, 3] = 1
    colors_rgb = np.array(
        [
            [255, 255, 255],
            [255, 0, 0],
        ],
        dtype=np.uint8,
    )
    processor = MorphologicalOpenMergeProcessor(
        min_width_px=3,
        merge_passes=1,
        label_smooth_size=1,
        label_smooth_passes=0,
    )

    processed = processor.process(labels, colors_rgb)

    assert np.all(processed == 0)


def test_label_processor_min_width_inches_converts_to_pixels():
    processor = build_label_processor(
        LabelProcessorConfig(
            name="distance_transform_merge",
            min_width_inches=0.1,
            merge_passes=1,
            label_smooth_size=1,
            label_smooth_passes=0,
        ),
        dpi=300,
    )

    assert isinstance(processor, DistanceTransformMergeProcessor)
    assert processor.min_width_px == 30


def test_sample_data_can_generate_palette_from_image():
    repo_root = Path(__file__).resolve().parents[1]
    config = PBNConfig(
        input=InputConfig(
            image="data/hills.jpg",
            palette=None,
        ),
        canvas=CanvasConfig(size=(2.0, 2.0), dpi=30),
        label_processor=LabelProcessorConfig(
            min_area=0.05,
            merge_passes=1,
            label_smooth_passes=1,
        ),
        palette_generator=PaletteGeneratorConfig(num_colors=6),
        output=OutputConfig(directory="output", write_intermediate=False),
    )

    result = PBNGenerator(config, base_dir=repo_root).run()

    assert result.colored_path == repo_root / "output/hills_colored.png"
    assert result.outline_path == repo_root / "output/hills_outline.png"
    assert result.palette_path == repo_root / "output/hills_palette.png"
    assert result.intermediate_path is None
    assert result.colored_path.exists()
    assert result.outline_path.exists()
    assert result.palette_path.exists()
