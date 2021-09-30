import os

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.tensorboard as tbx
import torchvision.utils as vutils
from torchmetrics import IS, FID, KID


def _prepare_data_for_inception(x, device):
    r"""
    Preprocess data to be feed into the Inception model.
    """

    x = F.interpolate(x, 299, mode="bicubic", align_corners=False)
    minv, maxv = float(x.min()), float(x.max())
    x.clamp_(min=minv, max=maxv).add_(-minv).div_(maxv - minv + 1e-5)
    x.mul_(255).add_(0.5).clamp_(0, 255)

    return x.to(device).to(torch.uint8)


def _prepare_data_for_gan(x, nz, device):
    r"""
    Helper function to prepare inputs for model.
    """

    return (
        x.to(device),
        torch.randn((x.size(0), nz)).to(device),
    )


def _compute_prob(logits):
    r"""
    Computes probability from model output.
    """

    return torch.sigmoid(logits).mean()


def _compute_hinge_loss_g(fake_preds):
    r"""
    Computes generator hinge loss.
    """

    return -fake_preds.mean()


def _compute_hinge_loss_d(real_preds, fake_preds):
    r"""
    Computes discriminator hinge loss.
    """

    return F.relu(1.0 - real_preds).mean() + F.relu(1.0 + fake_preds).mean()


def _compute_loss_g(net_g, net_d, z):
    r"""
    General implementation to compute generator loss.
    """

    fakes = net_g(z)
    fake_preds = net_d(fakes).view(-1)
    loss_g = _compute_hinge_loss_g(fake_preds)

    return loss_g, fakes, fake_preds


def _compute_loss_d(net_g, net_d, reals, z):
    r"""
    General implementation to compute discriminator loss.
    """

    real_preds = net_d(reals).view(-1)
    fakes = net_g(z).detach()
    fake_preds = net_d(fakes).view(-1)
    loss_d = _compute_hinge_loss_d(real_preds, fake_preds)

    return loss_d, fakes, real_preds, fake_preds


def _train_step(net, opt, sch, loss_func):
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


class Trainer:
    r"""
    Trainer performs GAN training, checkpointing and logging.
    Attributes:
        net_g (Module): Torch generator model.
        net_d (Module): Torch discriminator model.
        opt_g (Optimizer): Torch optimizer for generator.
        opt_d (Optimizer): Torch optimizer for discriminator.
        sch_g (Scheduler): Torch lr scheduler for generator.
        sch_d (Scheduler): Torch lr scheduler for discriminator.
        train_dataloader (Dataloader): Torch training set dataloader.
        eval_dataloader (Dataloader): Torch evaluation set dataloader.
        nz (int): Generator input / noise dimension.
        log_dir (str): Path to store log outputs.
        ckpt_dir (str): Path to store and load checkpoints.
        device (Device): Torch device to dispatch data to.
    """

    def __init__(
        self,
        net_g,
        net_d,
        opt_g,
        opt_d,
        sch_g,
        sch_d,
        train_dataloader,
        eval_dataloader,
        nz,
        log_dir,
        ckpt_dir,
        device,
    ):
        # Setup models, dataloader, optimizers
        self.net_g = net_g.to(device)
        self.net_d = net_d.to(device)
        self.opt_g = opt_g
        self.opt_d = opt_d
        self.sch_g = sch_g
        self.sch_d = sch_d
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Setup training parameters
        self.device = device
        self.nz = nz
        self.step = 0

        # Setup checkpointing, evaluation and logging
        self.fixed_z = torch.randn((36, nz), device=device)
        self.logger = tbx.SummaryWriter(log_dir)
        self.ckpt_dir = ckpt_dir

    def _load_checkpoint(self):
        r"""
        Finds the last checkpoint in ckpt_dir and load states.
        """

        ckpt_paths = [f for f in os.listdir(self.ckpt_dir) if f.endswith(".pth")]
        if ckpt_paths:  # Train from scratch if no checkpoints were found
            last_ckpt_path = sorted(ckpt_paths, key=lambda f: int(f[:-4]))[-1]
            last_ckpt_path = os.path.join(self.ckpt_dir, last_ckpt_path)
            last_ckpt = torch.load(last_ckpt_path)
            self.net_g.load_state_dict(last_ckpt["net_g"])
            self.net_d.load_state_dict(last_ckpt["net_d"])
            self.opt_g.load_state_dict(last_ckpt["opt_g"])
            self.opt_d.load_state_dict(last_ckpt["opt_d"])
            self.sch_g.load_state_dict(last_ckpt["sch_g"])
            self.sch_d.load_state_dict(last_ckpt["sch_d"])
            self.step = last_ckpt["step"]

    def _save_checkpoint(self):
        r"""
        Saves model, optimizer and trainer states.
        """

        ckpt_path = os.path.join(self.ckpt_dir, f"{self.step}.pth")
        torch.save(
            {
                "net_g": self.net_g.state_dict(),
                "net_d": self.net_d.state_dict(),
                "opt_g": self.opt_g.state_dict(),
                "opt_d": self.opt_d.state_dict(),
                "sch_g": self.sch_g.state_dict(),
                "sch_d": self.sch_d.state_dict(),
                "step": self.step,
            },
            ckpt_path,
        )

    def _eval(self):
        r"""
        Evaluates model and logs metrics.
        """

        self.net_g.eval()
        self.net_d.eval()

        with torch.no_grad():

            # Initialize metrics
            is_, fid, kid = (
                IS().to(self.device),
                FID().to(self.device),
                KID().to(self.device),
            )
            metrics = {
                "L(G)": [],
                "L(D)": [],
                "D(x)": [],
                "D(G(z))": [],
            }

            for data, _ in tqdm(self.eval_dataloader, desc="Evaluating Model"):

                # Compute losses and save intermediate outputs
                reals, z = _prepare_data_for_gan(data, self.nz, self.device)
                loss_d, fakes, real_pred, fake_pred = _compute_loss_d(
                    self.net_g, self.net_d, reals, z
                )
                loss_g, _, _ = _compute_loss_g(self.net_g, self.net_d, z)

                # Update metrics
                metrics["L(G)"].append(loss_g)
                metrics["L(D)"].append(loss_d)
                metrics["D(x)"].append(_compute_prob(real_pred))
                metrics["D(G(z))"].append(_compute_prob(fake_pred))
                reals = _prepare_data_for_inception(reals, self.device)
                fakes = _prepare_data_for_inception(fakes, self.device)
                is_.update(fakes)
                fid.update(reals, real=True)
                fid.update(fakes, real=False)
                kid.update(reals, real=True)
                kid.update(fakes, real=False)

            # Process and log metrics
            for k, v in metrics.items():
                v = torch.stack(v).mean().item()
                self.logger.add_scalar(k, v, self.step)
            self.logger.add_scalar("lr(G)", self.sch_g.get_last_lr()[0], self.step)
            self.logger.add_scalar("lr(D)", self.sch_d.get_last_lr()[0], self.step)
            self.logger.add_scalar("IS", is_.compute()[0].item(), self.step)
            self.logger.add_scalar("FID", fid.compute().item(), self.step)
            self.logger.add_scalar("KID", kid.compute()[0].item(), self.step)

            # Create samples using fixed noise
            samples = self.net_g(self.fixed_z)
            samples = F.interpolate(samples, 256).cpu()
            samples = vutils.make_grid(samples, nrow=6, padding=4, normalize=True)
            self.logger.add_image("Samples", samples, self.step)

        self.logger.flush()

    def _train_step_g(self, z):
        r"""
        Performs a generator training step.
        """

        return _train_step(
            self.net_g,
            self.opt_g,
            self.sch_g,
            lambda: _compute_loss_g(self.net_g, self.net_d, z)[0],
        )

    def _train_step_d(self, reals, z):
        r"""
        Performs a discriminator training step.
        """

        return _train_step(
            self.net_d,
            self.opt_d,
            self.sch_d,
            lambda: _compute_loss_d(self.net_g, self.net_d, reals, z)[0],
        )

    def train(self, max_steps, repeat_d, eval_every, ckpt_every):
        r"""
        Performs GAN training, checkpointing and logging.
        Attributes:
            max_steps (int): Number of steps before stopping.
            repeat_d (int): Number of discriminator updates before a generator update.
            eval_every (int): Number of steps before logging to Tensorboard.
            ckpt_every (int): Number of steps before checkpointing models.
        """

        self._load_checkpoint()

        while True:
            pbar = tqdm(self.train_dataloader)
            for data, _ in pbar:

                # Training step
                reals, z = _prepare_data_for_gan(data, self.nz, self.device)
                loss_d = self._train_step_d(reals, z)
                if self.step % repeat_d == 0:
                    loss_g = self._train_step_g(z)

                pbar.set_description(
                    f"L(G):{loss_g.item():.2f}|L(D):{loss_d.item():.2f}|{self.step}/{max_steps}"
                )

                if self.step != 0 and self.step % eval_every == 0:
                    self._eval()

                if self.step != 0 and self.step % ckpt_every == 0:
                    self._save_checkpoint()

                self.step += 1
                if self.step >= max_steps:
                    return
