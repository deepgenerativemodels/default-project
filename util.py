import os
import tarfile
import tempfile
import argparse
import urllib.request

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
from torchmetrics import IS, FID, KID


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
        with tqdm(unit="B", unit_scale=True, desc="Downloading Dataset") as pbar:
            update_download_progress.pbar = pbar
            urllib.request.urlretrieve(
                url, tmp_path, reporthook=update_download_progress
            )
        with tarfile.open(tmp_path, "r") as f:
            for member in tqdm(f.getmembers(), desc="Extracting Dataset"):
                f.extract(member, path=dst_dir)


def get_dataloaders(data_dir, imsize, batch_size, eval_size, num_workers=1):
    r"""
    Creates a dataloader from a directory containing image data.
    """

    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transforms.Compose(
            [
                transforms.Resize(imsize),
                transforms.CenterCrop(imsize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    eval_dataset, train_dataset = torch.utils.data.random_split(
        dataset,
        [eval_size, len(dataset) - eval_size],
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=batch_size, num_workers=num_workers
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    return train_dataloader, eval_dataloader


def prepare_data_for_inception(x, device):
    r"""
    Preprocess data to be feed into the Inception model.
    """

    x = F.interpolate(x, 299, mode="bicubic", align_corners=False)
    minv, maxv = float(x.min()), float(x.max())
    x.clamp_(min=minv, max=maxv).add_(-minv).div_(maxv - minv + 1e-5)
    x.mul_(255).add_(0.5).clamp_(0, 255)

    return x.to(device).to(torch.uint8)


def prepare_data_for_gan(x, nz, device):
    r"""
    Helper function to prepare inputs for model.
    """

    return (
        x.to(device),
        torch.randn((x.size(0), nz)).to(device),
    )


def compute_prob(logits):
    r"""
    Computes probability from model output.
    """

    return torch.sigmoid(logits).mean()


def compute_hinge_loss_g(fake_preds):
    r"""
    Computes generator hinge loss.
    """

    return -fake_preds.mean()


def compute_hinge_loss_d(real_preds, fake_preds):
    r"""
    Computes discriminator hinge loss.
    """

    return F.relu(1.0 - real_preds).mean() + F.relu(1.0 + fake_preds).mean()


def compute_loss_g(net_g, net_d, z):
    r"""
    General implementation to compute generator loss.
    """

    fakes = net_g(z)
    fake_preds = net_d(fakes).view(-1)
    loss_g = compute_hinge_loss_g(fake_preds)

    return loss_g, fakes, fake_preds


def compute_loss_d(net_g, net_d, reals, z):
    r"""
    General implementation to compute discriminator loss.
    """

    real_preds = net_d(reals).view(-1)
    fakes = net_g(z).detach()
    fake_preds = net_d(fakes).view(-1)
    loss_d = compute_hinge_loss_d(real_preds, fake_preds)

    return loss_d, fakes, real_preds, fake_preds


def train_step(net, opt, sch, loss_func):
    r"""
    General implementation to perform a training step.
    """

    net.train()
    loss = loss_func()
    net.zero_grad()
    loss.backward()
    opt.step()
    sch.step()

    return loss


def load_checkpoint(state_dict, path):
    r"""
    Loads checkpoint from path into state_dict.
    """

    ckpt = torch.load(path)
    for k, v in state_dict.items():
        assert k in ckpt, f"Missing key '{k}' from checkpoint at '{path}'."
        if isinstance(v, nn.Module):
            v.load_state_dict(ckpt[k])
        else:
            state_dict[k] = ckpt[k]


def save_checkpoint(state_dict, path):
    r"""
    Saves state_dict containing nn.Modules and scalars to path.
    """

    torch.save(
        {
            k: v.state_dict() if isinstance(v, nn.Module) else v
            for k, v in state_dict.items()
        },
        path,
    )


def eval(net_g, net_d, dataloader, nz, device, samples_z=None):
    r"""
    Evaluates model and logs metrics.
    """

    net_g.to(device).eval()
    net_d.to(device).eval()

    with torch.no_grad():

        # Initialize metrics
        is_, fid, kid, loss_gs, loss_ds, real_preds, fake_preds = (
            IS().to(device),
            FID().to(device),
            KID().to(device),
            [],
            [],
            [],
            [],
        )

        for data, _ in tqdm(dataloader, desc="Evaluating Model"):

            # Compute losses and save intermediate outputs
            reals, z = prepare_data_for_gan(data, nz, device)
            loss_d, fakes, real_pred, fake_pred = compute_loss_d(net_g, net_d, reals, z)
            loss_g, _, _ = compute_loss_g(net_g, net_d, z)

            # Update metrics
            loss_gs.append(loss_g)
            loss_ds.append(loss_d)
            real_preds.append(compute_prob(real_pred))
            fake_preds.append(compute_prob(fake_pred))
            reals = prepare_data_for_inception(reals, device)
            fakes = prepare_data_for_inception(fakes, device)
            is_.update(fakes)
            fid.update(reals, real=True)
            fid.update(fakes, real=False)
            kid.update(reals, real=True)
            kid.update(fakes, real=False)

        # Process metrics
        metrics = {
            "L(G)": torch.stack(loss_gs).mean().item(),
            "L(D)": torch.stack(loss_ds).mean().item(),
            "D(x)": torch.stack(real_preds).mean().item(),
            "D(G(z))": torch.stack(fake_preds).mean().item(),
            "IS": is_.compute()[0].item(),
            "FID": fid.compute().item(),
            "KID": kid.compute()[0].item(),
        }

        # Create samples
        if samples_z is not None:
            samples = net_g(samples_z)
            samples = F.interpolate(samples, 256).cpu()
            samples = vutils.make_grid(samples, nrow=6, padding=4, normalize=True)
        else:
            samples = fakes

    return metrics, samples
