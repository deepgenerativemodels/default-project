import os
import tarfile
import tempfile
import argparse
import urllib.request

import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models
import trainer


def parse_args():
    """Parses command line arguments."""

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
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help=(
            "Name of the current experiment."
            "Checkpoints will be stored in '{out_dir}/{name}/ckpt/'. "
            "Logs will be stored in '{out_dir}/{name}/log/'. "
            "If there are existing checkpoints in '{out_dir}/{name}/ckpt/', "
            "training will resume from the last checkpoint."
        ),
    )
    parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
        help=(
            "Resumes training using the last checkpoint in '{out_dir}/{name}/ckpt/' if set. "
            "Throws error if '{out_dir}/{name}/' already exists by default."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Manual seed for reproducibility."
    )

    return parser.parse_args()


def download_data(
    dst_dir, url="http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
):
    """Downloads and uncompresses the specified dataset."""

    def update_download_progress(blk_n=1, blk_sz=1, total_sz=None):
        assert update_download_progress.pbar is not None
        pbar = update_download_progress.pbar
        if total_sz is not None:
            pbar.total = total_sz
        pbar.update(blk_n * blk_sz - pbar.n)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, "compressed_data")
        with tqdm.tqdm(unit="B", unit_scale=True, desc="Downloading Dataset") as pbar:
            update_download_progress.pbar = pbar
            urllib.request.urlretrieve(
                url, tmp_path, reporthook=update_download_progress
            )
        with tarfile.open(tmp_path, "r") as f:
            for member in tqdm.tqdm(f.getmembers(), desc="Extracting Dataset"):
                f.extract(member, path=dst_dir)


def prepare_data(data_dir, imsize, batch_size):
    """Creates a dataloader from a directory containing image data."""

    transform = transforms.Compose(
        [
            transforms.Resize(imsize),
            transforms.CenterCrop(imsize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    return torch.utils.data.DataLoader(
        datasets.ImageFolder(root=data_dir, transform=transform),
        batch_size=batch_size,
        shuffle=True,
    )


def train(args):
    """Sets up environment, configures and trains model."""

    # Setup dataset
    if not os.path.exists(args.data_dir):
        download_data(args.data_dir)

    # Check existing experiment
    exp_dir = os.path.join(args.out_dir, args.name)
    if os.path.exists(exp_dir) and not args.resume:
        raise FileExistsError(
            f"Directory '{exp_dir}' already exists. "
            "Set '--resume' if you wish to resume training or "
            "change '--name' if you wish to start a new experiment."
        )

    # Setup output directories
    log_dir = os.path.join(exp_dir, "log")
    ckpt_dir = os.path.join(exp_dir, "ckpt")
    for d in [args.out_dir, exp_dir, log_dir, ckpt_dir]:
        if not os.path.exists(d):
            os.mkdir(d)

    # Fixed seed
    torch.manual_seed(args.seed)

    # Set parameters
    nz, ngf, nc, ndf, imsize, batch_size = 100, 64, 3, 64, 64, 128
    num_epochs, lr, betas, ckpt_every, num_samples = 50, 0.0002, (0.5, 0.999), 50, 9
    criterion = nn.BCELoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Configure and initialize models
    net_g = models.Generator(nz, ngf, nc).to(device)
    net_g.apply(models.weights_init)
    net_d = models.Discriminator(nc, ndf).to(device)
    net_d.apply(models.weights_init)

    # Configure dataloader and trainer
    dataloader = prepare_data(args.data_dir, imsize, batch_size)
    trainer_ = trainer.Trainer(
        net_g,
        net_d,
        dataloader,
        num_epochs,
        nz,
        lr,
        betas,
        criterion,
        ckpt_every,
        ckpt_dir,
        log_dir,
        num_samples,
        device,
    )

    # Load last checkpoint if specified
    if args.resume:
        trainer_.load_checkpoint()

    # Train model
    trainer_.train()


if __name__ == "__main__":
    train(parse_args())
