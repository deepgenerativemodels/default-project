import os

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.utils.tensorboard as tbx
import torchvision.utils as vutils


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
        dataloader (Dataloader): Torch training set dataloader.
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
        dataloader,
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
        self.dataloader = dataloader

        # Setup training parameters
        self.device = device
        self.nz = nz
        self.step = 0

        # Setup checkpointing, evaluation and logging
        self.fixed_noise = torch.randn((36, nz), device=device)
        self.log_writer = tbx.SummaryWriter(log_dir)
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

    def _compute_loss_g(self, fake_preds):
        r"""
        Calculates generator hinge loss.
        """

        return -fake_preds.mean()

    def _compute_loss_d(self, real_preds, fake_preds):
        r"""
        Calculates discriminator hinge loss.
        """

        return F.relu(1.0 - real_preds).mean() + F.relu(1.0 + fake_preds).mean()

    def _train_step_g(self, noise):
        r"""
        Performs a generator training step.
        """

        fakes = self.net_g(noise)
        fake_preds = self.net_d(fakes).view(-1)
        loss_g = self._compute_loss_g(fake_preds)

        self.net_g.zero_grad()
        loss_g.backward()
        self.opt_g.step()

        return loss_g.item(), fake_preds.mean().item()

    def _train_step_d(self, reals, noise):
        r"""
        Performs a discriminator training step.
        """

        real_preds = self.net_d(reals).view(-1)
        fakes = self.net_g(noise).detach()
        fake_preds = self.net_d(fakes).view(-1)
        loss_d = self._compute_loss_d(real_preds, fake_preds)

        self.net_d.zero_grad()
        loss_d.backward()
        self.opt_d.step()

        return loss_d.item(), real_preds.mean().item()

    def _log(self, loss_g, loss_d, real_pred, fake_pred):
        r"""
        Records losses and samples.
        """

        with torch.no_grad():
            samples = self.net_g(self.fixed_noise)
            samples = F.interpolate(samples, 256).cpu()
            samples = vutils.make_grid(samples, nrow=6, padding=4, normalize=True)
        self.log_writer.add_image("Samples", samples, self.step)
        self.log_writer.add_scalar("L(G)", loss_g, self.step)
        self.log_writer.add_scalar("L(D)", loss_d, self.step)
        self.log_writer.add_scalar("D(x)", real_pred, self.step)
        self.log_writer.add_scalar("D(G(z))", fake_pred, self.step)
        self.log_writer.add_scalar("lr(G)", self.sch_g.get_last_lr()[0], self.step)
        self.log_writer.add_scalar("lr(D)", self.sch_d.get_last_lr()[0], self.step)
        self.log_writer.flush()

    def train(self, max_steps, repeat_d, log_every, ckpt_every):
        r"""
        Performs GAN training, checkpointing and logging.
        Attributes:
            max_steps (int): Number of steps before stopping.
            repeat_d (int): Number of discriminator updates before a generator update.
            log_every (int): Number of steps before logging to Tensorboard.
            ckpt_every (int): Number of steps before checkpointing models.
        """

        self._load_checkpoint()

        while True:
            pbar = tqdm(self.dataloader)
            for data, _ in pbar:

                # Prepare inputs
                reals = data.to(self.device)
                noise = torch.randn((data.size(0), self.nz)).to(self.device)

                # Training step
                loss_d, real_pred = self._train_step_d(reals, noise)
                if self.step % repeat_d == 0:
                    loss_g, fake_pred = self._train_step_g(noise)

                # Update learning rate
                self.sch_d.step()
                self.sch_g.step()

                pbar.set_description(
                    (
                        f"L(G):{loss_g:.2f}|L(D):{loss_d:.2f}|"
                        f"D(x):{real_pred:.2f}|D(G(z)):{fake_pred:.2f}|"
                        f"{self.step}/{max_steps}"
                    )
                )

                if self.step != 0 and self.step % log_every == 0:
                    self._log(loss_g, loss_d, real_pred, fake_pred)

                if self.step != 0 and self.step % ckpt_every == 0:
                    self._save_checkpoint()

                self.step += 1
                if self.step >= max_steps:
                    return
