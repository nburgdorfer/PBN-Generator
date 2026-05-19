from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    import hydra
    from hydra.utils import to_absolute_path
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing dependency: hydra-core. Install it with: pip install hydra-core"
    ) from e

try:
    from ruamel.yaml import YAML
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing dependency: ruamel.yaml. Install it with: pip install ruamel.yaml"
    ) from e


def load_palette(palette_file):
    yaml = YAML(typ="safe", pure=True)
    with open(palette_file, "r") as pf:
        palette = yaml.load(pf)

    names = palette["names"]
    codes = palette["codes"]
    if len(names) != len(codes):
        raise ValueError("Palette must have the same number of names and codes.")

    colors = np.array([hex_to_rgb(code) for code in codes], dtype=np.uint8)
    return names, codes, colors


def hex_to_rgb(code):
    value = str(code).strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        raise ValueError(f"Invalid hex color code: {code}")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def load_canvas_image(image_file, canvas_size, dpi):
    width_in, height_in = canvas_size
    max_width_px = int(round(width_in * dpi))
    max_height_px = int(round(height_in * dpi))
    if max_width_px <= 0 or max_height_px <= 0:
        raise ValueError("Canvas dimensions must be positive.")

    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.LANCZOS

    image = Image.open(image_file).convert("RGB")
    source_width_px, source_height_px = image.size
    scale = min(
        max_width_px / source_width_px,
        max_height_px / source_height_px,
    )
    width_px = max(1, int(round(source_width_px * scale)))
    height_px = max(1, int(round(source_height_px * scale)))

    print(
        f"Resizing to {width_px}x{height_px} px "
        f"({width_px / dpi:.2f}x{height_px / dpi:.2f} in)"
    )
    image = image.resize((width_px, height_px), resample_filter)
    return np.asarray(image)


def bilateral_filter(image, filter_size):
    if filter_size <= 1:
        return image
    if filter_size % 2 == 0:
        filter_size += 1
    return cv2.bilateralFilter(image, filter_size, sigmaColor=55, sigmaSpace=55)


def palette_lab(colors_rgb):
    colors = colors_rgb.reshape(1, -1, 3)
    return cv2.cvtColor(colors, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)


def quantize_to_palette(image_rgb, colors_rgb):
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    pixels = image_lab.reshape(-1, 3).astype(np.float32)
    colors_lab = palette_lab(colors_rgb)
    labels = np.empty(pixels.shape[0], dtype=np.int32)

    chunk_size = 500_000
    for start in range(0, pixels.shape[0], chunk_size):
        stop = min(start + chunk_size, pixels.shape[0])
        chunk = pixels[start:stop]
        distances = np.sum((chunk[:, None, :] - colors_lab[None, :, :]) ** 2, axis=2)
        labels[start:stop] = np.argmin(distances, axis=1)

    return labels.reshape(image_rgb.shape[:2])


def cluster_image_lab(image_rgb, num_colors):
    if num_colors <= 0:
        raise ValueError("Number of generated palette colors must be positive.")

    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    pixels = image_lab.reshape(-1, 3).astype(np.float32)
    k = min(num_colors, pixels.shape[0])

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    _, cluster_labels, centers_lab = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS,
    )
    return cluster_labels.reshape(image_rgb.shape[:2]), centers_lab


def cluster_then_quantize_to_palette(image_rgb, colors_rgb, allow_unused_colors):
    cluster_labels, centers_lab = cluster_image_lab(image_rgb, len(colors_rgb))
    center_to_palette = map_cluster_centers_to_palette(
        centers_lab,
        colors_rgb,
        allow_unused_colors,
    )
    palette_labels = center_to_palette[cluster_labels.flatten()]
    return palette_labels.reshape(image_rgb.shape[:2])


def map_cluster_centers_to_palette(centers_lab, colors_rgb, allow_unused_colors):
    colors_lab = palette_lab(colors_rgb)
    distances = np.linalg.norm(
        centers_lab[:, None, :] - colors_lab[None, :, :],
        axis=2,
    )

    if allow_unused_colors:
        return np.argmin(distances, axis=1).astype(np.int32)

    assignments = np.full(centers_lab.shape[0], -1, dtype=np.int32)
    used_palette_colors = set()
    center_count, palette_count = distances.shape
    ranked_pairs = np.argsort(distances, axis=None)

    for pair_index in ranked_pairs:
        center_index, palette_index = np.unravel_index(pair_index, distances.shape)
        if assignments[center_index] != -1:
            continue
        if palette_index in used_palette_colors:
            continue
        assignments[center_index] = palette_index
        used_palette_colors.add(palette_index)
        if len(used_palette_colors) == min(center_count, palette_count):
            break

    for center_index in range(center_count):
        if assignments[center_index] == -1:
            assignments[center_index] = int(np.argmin(distances[center_index]))

    return assignments


def lab_centers_to_rgb(centers_lab):
    centers = np.clip(np.rint(centers_lab), 0, 255).astype(np.uint8)
    centers = centers.reshape(1, -1, 3)
    return cv2.cvtColor(centers, cv2.COLOR_LAB2RGB).reshape(-1, 3)


def rgb_to_hex(color):
    r, g, b = [int(channel) for channel in color]
    return f"#{r:02X}{g:02X}{b:02X}"


def sort_generated_palette(labels, colors_rgb):
    lightness = palette_lab(colors_rgb)[:, 0]
    order = np.argsort(lightness)
    remap = np.empty(len(order), dtype=np.int32)
    remap[order] = np.arange(len(order))
    return remap[labels], colors_rgb[order]


def generate_palette_from_image(image_rgb, num_colors):
    labels, centers_lab = cluster_image_lab(image_rgb, num_colors)
    colors_rgb = lab_centers_to_rgb(centers_lab)
    labels, colors_rgb = sort_generated_palette(labels, colors_rgb)
    names = [f"generated-{index:02d}" for index in range(1, len(colors_rgb) + 1)]
    codes = [rgb_to_hex(color) for color in colors_rgb]
    return names, codes, colors_rgb, labels


def assign_palette_labels(image_rgb, colors_rgb, quantize_mode, allow_unused_colors):
    if quantize_mode == "direct":
        return quantize_to_palette(image_rgb, colors_rgb)

    return cluster_then_quantize_to_palette(
        image_rgb,
        colors_rgb,
        allow_unused_colors,
    )


def smooth_label_edges(labels, num_colors, window_size, passes):
    if window_size <= 1 or passes <= 0:
        return labels
    if window_size % 2 == 0:
        window_size += 1

    labels = labels.copy()
    for _ in range(passes):
        best_count = np.full(labels.shape, -1.0, dtype=np.float32)
        smoothed = labels.copy()

        for color_index in range(num_colors):
            mask = (labels == color_index).astype(np.float32)
            counts = cv2.boxFilter(
                mask,
                ddepth=-1,
                ksize=(window_size, window_size),
                normalize=False,
                borderType=cv2.BORDER_REPLICATE,
            )
            counts[labels == color_index] += 0.01
            replace = counts > best_count
            smoothed[replace] = color_index
            best_count[replace] = counts[replace]

        labels = smoothed

    return labels


def merge_small_regions(labels, colors_rgb, min_area_px, max_passes):
    if min_area_px <= 1:
        return labels

    colors_lab = palette_lab(colors_rgb)
    kernel = np.ones((3, 3), dtype=np.uint8)
    labels = labels.copy()
    num_colors = len(colors_rgb)

    for pass_index in range(max_passes):
        changed = 0
        for color_index in range(num_colors):
            mask = (labels == color_index).astype(np.uint8)
            component_count, components, stats, _ = cv2.connectedComponentsWithStats(
                mask,
                connectivity=8,
            )

            for component_index in range(1, component_count):
                area = stats[component_index, cv2.CC_STAT_AREA]
                if area >= min_area_px:
                    continue

                component_mask = components == component_index
                neighbor_mask = cv2.dilate(
                    component_mask.astype(np.uint8),
                    kernel,
                    iterations=1,
                ).astype(bool)
                neighbor_mask &= ~component_mask

                neighbor_labels = labels[neighbor_mask]
                neighbor_labels = neighbor_labels[neighbor_labels != color_index]
                if neighbor_labels.size == 0:
                    continue

                next_label = choose_merge_label(
                    color_index,
                    neighbor_labels,
                    colors_lab,
                    num_colors,
                )
                labels[component_mask] = next_label
                changed += 1

        print(f"Merge pass {pass_index + 1}: merged {changed} undersized sections")
        if changed == 0:
            break

    return labels


def choose_merge_label(current_label, neighbor_labels, colors_lab, num_colors):
    counts = np.bincount(neighbor_labels, minlength=num_colors).astype(np.float32)
    color_distances = np.linalg.norm(colors_lab - colors_lab[current_label], axis=1)
    scores = counts / (1.0 + color_distances)
    scores[current_label] = -1.0
    return int(np.argmax(scores))


def build_border_mask(labels, thickness):
    border = np.zeros(labels.shape, dtype=bool)
    border[1:, :] |= labels[1:, :] != labels[:-1, :]
    border[:-1, :] |= labels[:-1, :] != labels[1:, :]
    border[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    border[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if thickness > 1:
        kernel = np.ones((3, 3), dtype=np.uint8)
        border = cv2.dilate(
            border.astype(np.uint8),
            kernel,
            iterations=thickness - 1,
        ).astype(bool)

    return border


def colorize_labels(labels, colors_rgb, border_mask):
    image = colors_rgb[labels].copy()
    image[border_mask] = np.array([0, 0, 0], dtype=np.uint8)
    return image


def build_numbered_outline(labels, border_mask, dpi, num_colors):
    outline = np.full((*labels.shape, 3), 255, dtype=np.uint8)
    outline[border_mask] = np.array([0, 0, 0], dtype=np.uint8)
    draw_region_numbers(outline, labels, dpi, num_colors)
    return outline


def draw_region_numbers(image, labels, dpi, num_colors):
    font = cv2.FONT_HERSHEY_SIMPLEX
    base_scale = max(0.35, dpi / 250.0)

    for color_index in range(num_colors):
        mask = (labels == color_index).astype(np.uint8)
        component_count, components, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

        for component_index in range(1, component_count):
            text = str(color_index + 1)
            x = stats[component_index, cv2.CC_STAT_LEFT]
            y = stats[component_index, cv2.CC_STAT_TOP]
            w = stats[component_index, cv2.CC_STAT_WIDTH]
            h = stats[component_index, cv2.CC_STAT_HEIGHT]
            roi = (components[y : y + h, x : x + w] == component_index).astype(np.uint8)
            center_x, center_y, radius = best_label_position(roi, x, y)

            thickness = 1
            (unit_width, unit_height), unit_baseline = cv2.getTextSize(
                text,
                font,
                1.0,
                thickness,
            )
            available = max(6.0, radius * 1.45)
            scale = min(
                base_scale,
                available / max(unit_width, unit_height + unit_baseline),
            )
            scale = max(0.25, scale)

            if scale > 0.75:
                thickness = 2

            (text_width, text_height), baseline = cv2.getTextSize(
                text,
                font,
                scale,
                thickness,
            )
            origin = (
                int(round(center_x - text_width / 2)),
                int(round(center_y + text_height / 2)),
            )
            cv2.putText(
                image,
                text,
                origin,
                font,
                scale,
                (0, 0, 0),
                thickness,
                lineType=cv2.LINE_AA,
            )


def best_label_position(component_roi, offset_x, offset_y):
    padded = np.pad(component_roi, 1, mode="constant", constant_values=0)
    distances = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    _, radius, _, max_location = cv2.minMaxLoc(distances)
    center_x = offset_x + max_location[0] - 1
    center_y = offset_y + max_location[1] - 1
    return center_x, center_y, radius

def segment_image(image, color_codes):
    k = len(color_codes)
    pixels = np.float32(image.reshape((-1,3)))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    retval, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((image.shape))

    return cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)


def save_rgb(path, image):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def format_hex_code(code):
    value = str(code).strip().lstrip("#").upper()
    return f"#{value}"


def build_palette_legend_image(names, codes, colors_rgb):
    row_height = 64
    swatch_size = 42
    padding = 20
    gap = 18
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.62
    thickness = 1

    labels = [
        f"{index}. {name}  {format_hex_code(code)}"
        for index, (name, code) in enumerate(zip(names, codes), start=1)
    ]
    text_widths = [
        cv2.getTextSize(label, font, font_scale, thickness)[0][0]
        for label in labels
    ]
    width = padding * 2 + swatch_size + gap + max(text_widths, default=240)
    height = padding * 2 + row_height * len(labels)
    image = np.full((height, width, 3), 255, dtype=np.uint8)

    for row, (label, color) in enumerate(zip(labels, colors_rgb)):
        y = padding + row * row_height
        swatch_left = padding
        swatch_top = y + (row_height - swatch_size) // 2
        swatch_right = swatch_left + swatch_size
        swatch_bottom = swatch_top + swatch_size

        image[swatch_top:swatch_bottom, swatch_left:swatch_right] = color
        cv2.rectangle(
            image,
            (swatch_left, swatch_top),
            (swatch_right, swatch_bottom),
            (0, 0, 0),
            1,
        )
        cv2.putText(
            image,
            label,
            (swatch_right + gap, y + 40),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            lineType=cv2.LINE_AA,
        )

    return image


def print_palette_legend(names, codes):
    print("Palette numbers:")
    for index, (name, code) in enumerate(zip(names, codes), start=1):
        print(f"{index}: {name} ({code})")


def has_palette_path(palette_file):
    if palette_file is None:
        return False
    return str(palette_file).strip().lower() not in ("", "none", "null")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    if cfg.regions.min_area < 0:
        raise ValueError("Minimum region area must be non-negative.")
    if cfg.canvas.dpi <= 0:
        raise ValueError("DPI must be positive.")
    if cfg.quantization.mode not in ("clustered", "direct"):
        raise ValueError("quantization.mode must be 'clustered' or 'direct'.")

    min_area_px = int(round(cfg.regions.min_area * cfg.canvas.dpi * cfg.canvas.dpi))
    border_thickness = cfg.output.border_thickness
    if border_thickness <= 0:
        border_thickness = max(1, int(round(cfg.canvas.dpi * 0.008)))

    print("Loading canvas image...")
    image = load_canvas_image(
        to_absolute_path(cfg.input.image),
        cfg.canvas.size,
        cfg.canvas.dpi,
    )
    image = bilateral_filter(image, cfg.smoothing.image_filter_size)
    save_rgb(to_absolute_path(f"{cfg.output.prefix}_bilat.png"), image)

    if has_palette_path(cfg.input.palette):
        names, codes, colors_rgb = load_palette(to_absolute_path(cfg.input.palette))
        print_palette_legend(names, codes)
        print(f"Assigning pixels to palette colors with {cfg.quantization.mode} mode...")
        labels = assign_palette_labels(
            image,
            colors_rgb,
            cfg.quantization.mode,
            cfg.quantization.allow_unused_palette_colors,
        )
    else:
        print(
            "Generating palette from image with "
            f"{cfg.quantization.generated_palette_size} colors..."
        )
        names, codes, colors_rgb, labels = generate_palette_from_image(
            image,
            cfg.quantization.generated_palette_size,
        )
        print_palette_legend(names, codes)

    print("Smoothing region borders...")
    labels = smooth_label_edges(
        labels,
        len(colors_rgb),
        cfg.smoothing.label_smooth_size,
        cfg.smoothing.label_smooth_passes,
    )

    print(
        "Merging sections smaller than "
        f"{cfg.regions.min_area} square inches ({min_area_px} px)..."
    )
    labels = merge_small_regions(
        labels,
        colors_rgb,
        min_area_px,
        cfg.regions.merge_passes,
    )
    labels = smooth_label_edges(
        labels,
        len(colors_rgb),
        cfg.smoothing.label_smooth_size,
        1 if cfg.smoothing.label_smooth_passes > 0 else 0,
    )
    labels = merge_small_regions(
        labels,
        colors_rgb,
        min_area_px,
        cfg.regions.merge_passes,
    )

    print("Drawing outputs...")
    border_mask = build_border_mask(labels, border_thickness)
    colored = colorize_labels(labels, colors_rgb, border_mask)
    outline = build_numbered_outline(
        labels,
        border_mask,
        cfg.canvas.dpi,
        len(colors_rgb),
    )

    colored_path = to_absolute_path(f"{cfg.output.prefix}_colored.png")
    outline_path = to_absolute_path(f"{cfg.output.prefix}_outline.png")
    palette_path = to_absolute_path(f"{cfg.output.prefix}_palette.png")
    save_rgb(colored_path, colored)
    save_rgb(outline_path, outline)
    save_rgb(palette_path, build_palette_legend_image(names, codes, colors_rgb))
    print(f"Wrote {colored_path}")
    print(f"Wrote {outline_path}")
    print(f"Wrote {palette_path}")


if __name__ == "__main__":
    main()
