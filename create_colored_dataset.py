import argparse
import gc
import glob
import logging
import os

from PIL import Image, ImageStat
from rich.traceback import install
from tqdm.auto import tqdm

from src.utils import resize_max


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_path",
        type=str,
        required=True,
        help="Path to the raw input directory",
    )
    parser.add_argument(
        "--colored_path",
        type=str,
        required=True,
        help="Path to the colored output directory",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        help="Max size of the shortest dimension",
        default=768,
    )
    args = parser.parse_args()
    return args


def main(args):
    # Create output directory
    logging.info("Creating output directory")
    os.makedirs(args.colored_path, exist_ok=True)

    # Conversion
    logging.info("Start conversion")
    raw_files = sorted(glob.glob(f"{args.raw_path}/**"))
    i, acp, exp, rej = 0, 0, 0, 0
    with tqdm(raw_files, unit="sample") as tbatch:
        for sample, raw_file in enumerate(tbatch):
            try:
                img = Image.open(raw_file)
                img = img.convert("RGB")
                img = resize_max(img, args.max_size)

                stat = ImageStat.Stat(img)
                if sum(stat.sum) / 3 != stat.sum[0]:
                    acp += 1
                    path = f"{args.results_path}/{i:06}.png"
                    img.save(path)
                else:
                    rej += 1
            except:
                exp += 1
            finally:
                tbatch.set_postfix(acp=acp, rej=rej, exp=exp)
                i += 1

    # Clean up
    gc.collect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    install(show_locals=False)
    args = parse_args()
    main(args)
