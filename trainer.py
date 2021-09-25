import os

from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.tensorboard as tbx
import torchvision.utils as vutils


class Trainer:
    """Trainer performs GAN training, checkpointing and logging."""

    def __init__(
        self,
        net_g,
        net_d,
        dataloader,
        num_epochs,
        nz,
        lr,
        betas,
        criterion,
        train_ratio,
        ckpt_every,
        ckpt_dir,
        log_dir,
        num_samples,
        device,
    ):
        # Setup models, dataloader, optimizers and loss
        self.net_g = net_g
        self.net_d = net_d
        self.dataloader = dataloader
        self.opt_g = optim.Adam(net_g.parameters(), lr=lr, betas=betas)
        self.opt_d = optim.Adam(net_d.parameters(), lr=lr, betas=betas)
        self.criterion = criterion

        # Setup training parameters
        self.num_epochs = num_epochs
        self.train_ratio = train_ratio
        self.device = device
        self.nz = nz
        self.step = 0

        # Setup checkpointing, evaluation and logging
        self.ckpt_every = ckpt_every
        self.ckpt_dir = ckpt_dir
        self.fixed_noise = torch.randn(num_samples, nz, 1, 1, device=device)
        self.log_writer = tbx.SummaryWriter(log_dir)

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

    def log(self, samples, statistics, resize=256):
        """Saves generated samples and training statistics to tensorboard logs."""

        samples = F.interpolate(samples, resize)
        grid = vutils.make_grid(samples, nrow=3, padding=2, normalize=True)
        self.log_writer.add_image("Samples", grid, self.step)
        for k, v in statistics.items():
            self.log_writer.add_scalar(k, v, self.step)

    def eval(self):
        """Generates fake samples using fixed noise."""

        with torch.no_grad():
            fakes = self.net_g(self.fixed_noise).cpu()
            return fakes

    def train_step_g(self, real_labels):
        """Performs a generator training step."""

        # Train generator ~ argmax log(D(G(z)))
        self.net_g.zero_grad()
        noise = torch.randn(real_labels.size(0), self.nz, 1, 1, device=self.device)
        fakes = self.net_g(noise)
        fake_preds = self.net_d(fakes).view(-1)
        loss_g = self.criterion(fake_preds, real_labels)
        loss_g.backward()
        self.opt_g.step()

        return loss_g.item(), fake_preds.mean().item()

    def train_step_d(self, data, real_labels, fake_labels):
        """Performs a discriminator training step."""

        # Calculate discriminator loss on real data ~ log(D(x))
        self.net_d.zero_grad()
        reals = data.to(self.device)
        real_preds = self.net_d(reals).view(-1)
        loss_d_real = self.criterion(real_preds, real_labels)

        # Calculate discriminator loss on fake data ~ log(1 - D(G(z)))
        noise = torch.randn(data.size(0), self.nz, 1, 1, device=self.device)
        fakes = self.net_g(noise)
        fake_preds = self.net_d(fakes.detach()).view(-1)
        loss_d_fake = self.criterion(fake_preds, fake_labels)

        # Train discriminator ~ argmax log(D(x)) +  log(1 - D(G(z)))
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        self.opt_d.step()

        return loss_d.item(), real_preds.mean().item()

    def train(self):
        """Performs GAN training and logs progress."""

        for epoch in range(1, self.num_epochs + 1):
            tqdm_dataloader = tqdm(self.dataloader)
            for data, _ in tqdm_dataloader:

                # Training step
                real_labels = torch.ones((data.size(0),), device=self.device)
                fake_labels = torch.zeros((data.size(0),), device=self.device)
                loss_d, mean_real_pred = self.train_step_d(
                    data, real_labels, fake_labels
                )
                for _ in range(self.train_ratio):
                    loss_g, mean_fake_pred = self.train_step_g(real_labels)

                # Print statistics
                statistics = {
                    "L(D)": loss_d,
                    "L(G)": loss_g,
                    "<D(x)>": mean_real_pred,
                    "<D(G(z))>": mean_fake_pred,
                }
                tqdm_dataloader.set_description(
                    f"EPOCH:{epoch}|STEP:{self.step}|"
                    + "|".join(f"{k}:{v:.2f}" for k, v in statistics.items())
                )

                # Log statistics and checkpoint
                if self.step % self.ckpt_every == 0:
                    self.log(self.eval(), statistics)
                    self.save_checkpoint()

                self.step += 1
