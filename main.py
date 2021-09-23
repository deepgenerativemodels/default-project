import urllib.request
import os
import tarfile
import tempfile
import pathlib
import argparse
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    pass


def download_data(dst_dir):
    """Downloads and uncompresses the Stanford Dogs dataset."""

    def update_download_progress(blk_n=1, blk_sz=1, total_sz=None):
        assert update_download_progress.pbar is not None
        pbar = update_download_progress.pbar
        if total_sz is not None:
            pbar.total = total_sz
        pbar.update(blk_n * blk_sz - pbar.n)

    url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, "compressed_data")
        with tqdm(unit="B", unit_scale=True, desc="Downloading Dataset") as pbar:
            update_download_progress.pbar = pbar
            urllib.request.urlretrieve(
                url, tmp_path, reporthook=update_download_progress
            )
        with tarfile.open(tmp_path, "r") as f:
            for member in tqdm(f.getmembers(), desc="Extracting Dataset"):
                f.extract(member, path=dst_dir)


def get_images(img_dir):
    """Gets the paths of all images contained in directory."""
    return list(pathlib.Path(img_dir).rglob("*.jpg"))


def main(args):
    pass


if __name__ == "__main__":
    path = os.path.abspath(os.path.dirname(__file__))
    data = os.path.join(path, "data")
    download_data(data)
    print(get_images(data))
