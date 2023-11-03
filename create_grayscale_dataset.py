import argparse
import gc
import logging
import os

from PIL import Image
from rich.traceback import install
from tqdm.auto import tqdm

from src.utils import get_entries


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--colored_path",
        type=str,
        required=True,
        help="Path to the colored input directory",
    )
    parser.add_argument(
        "--grayscale_path",
        type=str,
        required=True,
        help="Path to the grayscale output directory",
    )
    args = parser.parse_args()
    return args


def main(args):
    # Create output directory
    logging.info("Creating output directory")
    os.makedirs(args.grayscale_path, exist_ok=True)

    # Conversion
    logging.info("Start conversion")
    colored_files = get_entries(f"{args.colored_path}/*.png")
    with tqdm(colored_files, unit="sample") as tbatch:
        for sample, colored_file in enumerate(tbatch):
            img = Image.open(colored_file)
            img = img.convert("L")

            path = f"{args.grayscale_path}/{sample:06}.png"
            img.save(path)

    # Clean up
    gc.collect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    install(show_locals=False)
    args = parse_args()
    main(args)
