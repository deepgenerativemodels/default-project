import os
import pprint
import tarfile
import tempfile
import argparse
import urllib.request

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import model
import trainer


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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Minibatch size used during training.",
    )
    parser.add_argument(
        "--max_steps", type=int, default=100000, help="Number of steps to train for."
    )
    parser.add_argument(
        "--repeat_d",
        type=int,
        default=5,
        help="Number of discriminator updates before a generator update.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=100,
        help="Number of steps between checkpointing.",
    )
    parser.add_argument(
        "--ckpt_every",
        type=int,
        default=1000,
        help="Number of steps between checkpointing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda:0" if torch.cuda.is_available() else "cpu"),
        help="Device to train on.",
    )

    return parser.parse_args()


def download_data(
    dst_dir, url="http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
):
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
        with tqdm(unit="B", unit_scale=True, desc="Downloading Dataset") as pbar:
            update_download_progress.pbar = pbar
            urllib.request.urlretrieve(
                url, tmp_path, reporthook=update_download_progress
            )
        with tarfile.open(tmp_path, "r") as f:
            for member in tqdm(f.getmembers(), desc="Extracting Dataset"):
                f.extract(member, path=dst_dir)


def prepare_data(data_dir, imsize, batch_size):
    r"""
    Creates a dataloader from a directory containing image data.
    """

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
    r"""
    Sets up environment, configures and trains model.
    """

    # Print command line arguments and architectures
    pprint.pprint(vars(args))

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
    nz, bw, ngf, ndf, nc, imsize, lr, betas = (
        128,
        4,
        256,
        128,
        3,
        32,
        2e-4,
        (0.0, 0.9),
    )

    # Configure models, optimizers and schedulers
    net_g = model.Generator(nz, ngf, bw, nc)
    net_d = model.Discriminator(nc, ndf)
    opt_g = optim.Adam(net_g.parameters(), lr, betas)
    opt_d = optim.Adam(net_d.parameters(), lr, betas)
    linear_sch = lambda s: 1.0 - (s / args.max_steps)
    sch_g = optim.lr_scheduler.LambdaLR(opt_g, lr_lambda=linear_sch)
    sch_d = optim.lr_scheduler.LambdaLR(opt_d, lr_lambda=linear_sch)

    # Configure dataloader and trainer
    dataloader = prepare_data(args.data_dir, imsize, args.batch_size)
    trainer_ = trainer.Trainer(
        net_g,
        net_d,
        opt_g,
        opt_d,
        sch_g,
        sch_d,
        dataloader,
        nz,
        log_dir,
        ckpt_dir,
        torch.device(args.device),
    )

    # Train model
    trainer_.train(args.max_steps, args.repeat_d, args.log_every, args.ckpt_every)


if __name__ == "__main__":
    train(parse_args())
