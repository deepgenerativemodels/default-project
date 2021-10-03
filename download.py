import os
import argparse

import util


def parse_args():
    r"""
    Parses command line arguments.
    """

    root_dir = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(root_dir, "data"),
        help=(
            "Path to dataset directory. "
            "A new dataset will be downloaded if the directory does not exist."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(root_dir, "out"),
        help=(
            "Path to output directory. "
            "A new one will be created if the directory does not exist."
        ),
    )

    return parser.parse_args()


def download(args):
    r"""
    Downloads dataset and baseline models.
    """

    util.download_data(
        args.data_dir, url="http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
    )

    util.download_data(
        args.out_dir,
        url="https://github.com/deepgenerativemodels/default-project/releases/download/f.2021.v2/baselines-150k.tar",
    )
    
    util.download_data(
        args.out_dir,
        url="https://github.com/deepgenerativemodels/default-project/releases/download/f.2021.v2/baselines-295k.tar",
    )


if __name__ == "__main__":
    download(parse_args())
