import os
import argparse
import tarfile
import tempfile
import urllib.request

from tqdm import tqdm


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


def download_data(dst_dir, url):
    r"""
    Downloads and uncompresses the specified dataset.
    """

    def update_download_progress(blk_n=1, blk_sz=1, total_sz=None):
        assert update_download_progress.pbar is not None
        pbar = update_download_progress.pbar
        if total_sz is not None:
            pbar.total = total_sz
        pbar.update(blk_n * blk_sz - pbar.n)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, "compressed_data")
        url_name = url.split("/")[-1]
        with tqdm(unit="B", unit_scale=True, desc=f"Downloading {url_name}") as pbar:
            update_download_progress.pbar = pbar
            urllib.request.urlretrieve(
                url, tmp_path, reporthook=update_download_progress
            )
        with tarfile.open(tmp_path, "r") as f:
            for member in tqdm(f.getmembers(), desc=f"Extracting {url_name}"):
                f.extract(member, path=dst_dir)


def download(args):
    r"""
    Downloads dataset and baseline models.
    """

    download_data(
        args.data_dir, url="http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
    )

    download_data(
        args.out_dir,
        url="https://github.com/deepgenerativemodels/default-project/releases/download/f.2021.v2/baselines-150k.tar",
    )

    download_data(
        args.out_dir,
        url="https://github.com/deepgenerativemodels/default-project/releases/download/f.2021.v2/baselines-295k.tar",
    )


if __name__ == "__main__":
    download(parse_args())
