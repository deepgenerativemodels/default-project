import os

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.utils.tensorboard as tbx
import torchvision.utils as vutils


class Trainer:
    """Trainer performs GAN training, checkpointing and logging."""

    def __init__(
        self,
        net_g,
        net_d,
        opt_g,
        opt_d,
        criterion,
        dataloader,
        nz,
        log_dir,
        ckpt_dir,
        device,
    ):
        # Setup models, dataloader, optimizers and loss
        self.net_g = net_g.to(device)
        self.net_d = net_d.to(device)
        self.opt_g = opt_g
        self.opt_d = opt_d
        self.criterion = criterion
        self.dataloader = dataloader

        # Setup training parameters
        self.device = device
        self.nz = nz
        self.step = 0

        # Setup checkpointing, evaluation and logging
        self.fixed_noise = torch.randn(36, nz, 1, 1, device=device)
        self.log_writer = tbx.SummaryWriter(log_dir)
        self.ckpt_dir = ckpt_dir

    def load_checkpoint(self):
        """Finds the last checkpoint in ckpt_dir and load states."""

        ckpt_paths = [f for f in os.listdir(self.ckpt_dir) if f.endswith(".pth")]
        if ckpt_paths:  # Train from scratch if no checkpoints were found
            last_ckpt_path = sorted(ckpt_paths, key=lambda f: int(f[:-4]))[-1]
            last_ckpt_path = os.path.join(self.ckpt_dir, last_ckpt_path)
            last_ckpt = torch.load(last_ckpt_path)
            self.net_g.load_state_dict(last_ckpt["net_g"])
            self.net_d.load_state_dict(last_ckpt["net_d"])
            self.opt_g.load_state_dict(last_ckpt["opt_g"])
            self.opt_d.load_state_dict(last_ckpt["opt_d"])
            self.step = last_ckpt["step"]

    def save_checkpoint(self):
        """Saves trainer states."""

        ckpt_path = os.path.join(self.ckpt_dir, f"{self.step}.pth")
        torch.save(
            {
                "net_g": self.net_g.state_dict(),
                "net_d": self.net_d.state_dict(),
                "opt_g": self.opt_g.state_dict(),
                "opt_d": self.opt_d.state_dict(),
                "step": self.step,
            },
            ckpt_path,
        )

    def log(self, samples, statistics):
        """Saves generated samples and training statistics to tensorboard logs."""

        samples = F.interpolate(samples, 256)  # Resize samples to 256x256
        grid = vutils.make_grid(samples, nrow=6, padding=4, normalize=True)
        self.log_writer.add_image("Samples", grid, self.step)
        for k, v in statistics.items():
            self.log_writer.add_scalar(k, v, self.step)
        self.log_writer.flush()

    def eval(self):
        """Generates fake samples using fixed noise."""

        with torch.no_grad():
            fakes = self.net_g(self.fixed_noise).cpu()
            return fakes

    def train_step_g(self, noise, real_labels):
        """Performs a generator training step."""

        # Train generator ~ argmax log(D(G(z)))
        fakes = self.net_g(noise)
        fake_preds = self.net_d(fakes).view(-1)
        loss_g = self.criterion(fake_preds, real_labels)
        self.net_g.zero_grad()
        loss_g.backward()
        self.opt_g.step()

        return loss_g.item(), fake_preds.mean().item()

    def train_step_d(self, reals, noise, real_labels, fake_labels):
        """Performs a discriminator training step."""

        # Calculate discriminator loss on real data ~ log(D(x))
        real_preds = self.net_d(reals).view(-1)
        loss_d_real = self.criterion(real_preds, real_labels)

        # Calculate discriminator loss on fake data ~ log(1 - D(G(z)))
        fakes = self.net_g(noise)
        fake_preds = self.net_d(fakes.detach()).view(-1)
        loss_d_fake = self.criterion(fake_preds, fake_labels)

        # Train discriminator ~ argmax log(D(x)) + log(1 - D(G(z)))
        loss_d = loss_d_real + loss_d_fake
        self.net_d.zero_grad()
        loss_d.backward()
        self.opt_d.step()

        return loss_d.item(), real_preds.mean().item()

    def train(self, max_steps, ckpt_every):
        """Performs GAN training and logs progress."""

        while True:
            tqdm_dataloader = tqdm(self.dataloader)
            for data, _ in tqdm_dataloader:

                # Setup inputs and labels
                reals = data.to(self.device)
                noise = torch.randn(data.size(0), self.nz, 1, 1).to(self.device)
                real_labels = torch.ones((data.size(0),)).to(self.device)
                fake_labels = torch.zeros((data.size(0),)).to(self.device)

                # Training step
                loss_d, mean_real_pred = self.train_step_d(
                    reals,
                    noise,
                    real_labels,
                    fake_labels,
                )
                loss_g, mean_fake_pred = self.train_step_g(
                    noise,
                    real_labels,
                )

                # Print statistics
                statistics = {
                    "loss_g": loss_g,
                    "loss_d": loss_d,
                    "D(x)": mean_real_pred,
                    "D(G(z))": mean_fake_pred,
                }
                tqdm_dataloader.set_description(
                    "|".join(
                        [f"step={self.step}"]
                        + [f"{k}={v:.2f}" for k, v in statistics.items()]
                    )
                )

                # Log statistics and checkpoint
                if self.step % ckpt_every == 0:
                    self.log(self.eval(), statistics)
                    self.save_checkpoint()

                self.step += 1
                if self.step >= max_steps:
                    return
