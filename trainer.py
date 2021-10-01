import os

from tqdm import tqdm
import torch
import torch.utils.tensorboard as tbx

import util


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

    def _state_dict(self):
        return {
            "net_g": self.net_g,
            "net_d": self.net_d,
            "opt_g": self.opt_g,
            "opt_d": self.opt_d,
            "sch_g": self.sch_g,
            "sch_d": self.sch_d,
            "step": self.step,
        }

    def _load_state_dict(self, state_dict):
        (
            self.net_g,
            self.net_d,
            self.opt_g,
            self.opt_d,
            self.sch_g,
            self.sch_d,
            self.step,
        ) = (
            state_dict["net_g"],
            state_dict["net_d"],
            state_dict["opt_g"],
            state_dict["opt_d"],
            state_dict["sch_g"],
            state_dict["sch_d"],
            state_dict["step"],
        )

    def _load_checkpoint(self):
        r"""
        Finds the last checkpoint in ckpt_dir and load states.
        """

        ckpt_paths = [f for f in os.listdir(self.ckpt_dir) if f.endswith(".pth")]
        if ckpt_paths:  # Train from scratch if no checkpoints were found
            ckpt_path = sorted(ckpt_paths, key=lambda f: int(f[:-4]))[-1]
            ckpt_path = os.path.join(self.ckpt_dir, ckpt_path)
            state_dict = self._state_dict()
            util.load_checkpoint(state_dict, ckpt_path)
            self._load_state_dict(state_dict)

    def _save_checkpoint(self):
        r"""
        Saves model, optimizer and trainer states.
        """

        ckpt_path = os.path.join(self.ckpt_dir, f"{self.step}.pth")
        util.save_checkpoint(self._state_dict(), ckpt_path)

    def _log(self, metrics, samples):
        r"""
        Logs metrics and samples to Tensorboard.
        """

        for k, v in metrics.items():
            self.logger.add_scalar(k, v, self.step)
        self.logger.add_image("Samples", samples, self.step)
        self.logger.flush()

    def _train_step_g(self, z):
        r"""
        Performs a generator training step.
        """

        return util.train_step(
            self.net_g,
            self.opt_g,
            self.sch_g,
            lambda: util.compute_loss_g(self.net_g, self.net_d, z)[0],
        )

    def _train_step_d(self, reals, z):
        r"""
        Performs a discriminator training step.
        """

        return util.train_step(
            self.net_d,
            self.opt_d,
            self.sch_d,
            lambda: util.compute_loss_d(self.net_g, self.net_d, reals, z)[0],
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
                reals, z = util.prepare_data_for_gan(data, self.nz, self.device)
                loss_d = self._train_step_d(reals, z)
                if self.step % repeat_d == 0:
                    loss_g = self._train_step_g(z)

                pbar.set_description(
                    f"L(G):{loss_g.item():.2f}|L(D):{loss_d.item():.2f}|{self.step}/{max_steps}"
                )

                if self.step != 0 and self.step % eval_every == 0:
                    self._log(
                        *util.eval(
                            self.net_g,
                            self.net_d,
                            self.eval_dataloader,
                            self.nz,
                            self.device,
                            samples_z=self.fixed_z,
                        )
                    )

                if self.step != 0 and self.step % ckpt_every == 0:
                    self._save_checkpoint()

                self.step += 1
                if self.step >= max_steps:
                    return
