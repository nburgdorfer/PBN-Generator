from pathlib import Path

from pbn_generator import (
    BilateralPreprocessor,
    CanvasConfig,
    ClusterLabelerConfig,
    ImagePreprocessorConfig,
    InputConfig,
    KMeansLabeler,
    LabelProcessorConfig,
    OutputConfig,
    PBNConfig,
    PBNGenerator,
    SmoothMergeProcessor,
)


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
