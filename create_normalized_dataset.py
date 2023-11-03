import argparse
import gc
import logging
import os
import pathlib

from rich.traceback import install
from tqdm.auto import tqdm

from src.utils import get_entries


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the un-normalized dataset",
    )
    args = parser.parse_args()
    return args


def main(args):
    # Conversion
    logging.info("Start conversion")

    files = get_entries(f"{args.path}/**")
    with tqdm(files, unit="sample") as tbatch:
        for sample, file in enumerate(tbatch):
            file_path = pathlib.Path(file)
            os.rename(file, f"{args.path}/{sample:06}{file_path.suffix}.renamed")

    files = get_entries(f"{args.path}/**")
    with tqdm(files, unit="sample") as tbatch:
        for sample, file in enumerate(tbatch):
            file_path = pathlib.Path(file)
            os.rename(file, file_path.with_suffix(""))

    # Clean up
    gc.collect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    install(show_locals=False)
    args = parse_args()
    main(args)
