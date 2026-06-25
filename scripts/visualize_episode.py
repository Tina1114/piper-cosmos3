#!/usr/bin/env python3
"""Save a quick RGB camera-history preview for one raw HDF5 episode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from piper_cosmos.data.hdf5_reader import HDF5EpisodeReader, find_hdf5_files, load_config


DEFAULT_DATA_ROOT = Path("/project/peilab/wam/physical_WM/data/pack_3_objects_plus/perfect")
DEFAULT_CONFIG = Path("configs/data/piper_dual_hdf5.yaml")
DEFAULT_OUTPUT = Path("reports/episode_preview.png")


def chw_to_uint8_image(array: np.ndarray) -> Image.Image:
    chw = np.asarray(array, dtype=np.float32)
    hwc = np.transpose(chw, (1, 2, 0))
    pixels = np.clip(hwc * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def make_grid(images: dict[str, np.ndarray], t: int) -> Image.Image:
    camera_names = list(images)
    history_frames = next(iter(images.values())).shape[0]
    tile_w = next(iter(images.values())).shape[3]
    tile_h = next(iter(images.values())).shape[2]
    label_h = 24
    grid = Image.new(
        "RGB",
        (len(camera_names) * tile_w, history_frames * (tile_h + label_h)),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(grid)
    for col, camera in enumerate(camera_names):
        history = images[camera]
        for row in range(history_frames):
            x = col * tile_w
            y = row * (tile_h + label_h)
            frame_index = t - history_frames + 1 + row
            grid.paste(chw_to_uint8_image(history[row]), (x, y + label_h))
            draw.text((x + 4, y + 4), f"{camera} t={frame_index}", fill=(0, 0, 0))
    return grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize three-camera RGB history from a raw Piper HDF5 episode."
    )
    parser.add_argument("--episode", type=Path, default=None, help="Specific HDF5 episode path.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory used when --episode is omitted.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Dataset YAML config.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="PNG output path.")
    parser.add_argument("--t", type=int, default=1, help="Timestep to visualize.")
    parser.add_argument("--history-frames", type=int, default=2, help="Number of history frames.")
    parser.add_argument("--image-size", type=int, default=224, help="Preview resize size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    episode = args.episode
    if episode is None:
        episodes = find_hdf5_files(args.data_root)
        if not episodes:
            raise SystemExit(f"No HDF5 episodes found under {args.data_root}")
        episode = episodes[0]

    reader = HDF5EpisodeReader(episode, config, image_size=args.image_size)
    images = reader.read_image_history(args.t, args.history_frames)
    grid = make_grid(images, args.t)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output)
    print(f"Wrote {args.output} from {episode} at t={args.t}")


if __name__ == "__main__":
    main()
