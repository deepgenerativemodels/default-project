r"""Baseline GAN architecture

The architecture is based on the Residual SNGAN implementation from
Mimicry: Towards the Reproducibility of GAN Research (Lee 2020).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralNorm(object):
    r"""
    Spectral Normalization for GANs (Miyato 2018).
    Inheritable class for performing spectral normalization of weights,
    as approximated using power iteration.
    Attributes:
        n_dim (int): Number of dimensions.
        num_iters (int): Number of iterations for power iter.
        eps (float): Epsilon for zero division tolerance when normalizing.
    """

    def __init__(self, n_dim, num_iters=1, eps=1e-12):
        self.num_iters = num_iters
        self.eps = eps

        # Register a singular vector for each sigma
        self.register_buffer("sn_u", torch.randn(1, n_dim))
        self.register_buffer("sn_sigma", torch.ones(1))

    @property
    def u(self):
        return getattr(self, "sn_u")

    @property
    def sigma(self):
        return getattr(self, "sn_sigma")

    def _power_iteration(self, W, u, num_iters, eps=1e-12):
        with torch.no_grad():
            for _ in range(num_iters):
                v = F.normalize(torch.matmul(u, W), eps=eps)
                u = F.normalize(torch.matmul(v, W.t()), eps=eps)

        # Note: must have gradients, otherwise weights do not get updated!
        sigma = torch.mm(u, torch.mm(W, v.t()))

        return sigma, u, v

    def sn_weights(self):
        r"""
        Spectrally normalize current weights of the layer.
        """
        W = self.weight.view(self.weight.shape[0], -1)

        # Power iteration
        sigma, u, v = self._power_iteration(
            W=W, u=self.u, num_iters=self.num_iters, eps=self.eps
        )

        # Update only during training
        if self.training:
            with torch.no_grad():
                self.sigma[:] = sigma
                self.u[:] = u

        return self.weight / sigma


class SNConv2d(nn.Conv2d, SpectralNorm):
    r"""
    Spectrally normalized layer for Conv2d.
    Attributes:
        in_channels (int): Input channel dimension.
        out_channels (int): Output channel dimensions.
    """

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        nn.Conv2d.__init__(self, in_channels, out_channels, *args, **kwargs)
        SpectralNorm.__init__(
            self, n_dim=out_channels, num_iters=kwargs.get("num_iters", 1)
        )

    def forward(self, x):
        return F.conv2d(
            input=x,
            weight=self.sn_weights(),
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class SNLinear(nn.Linear, SpectralNorm):
    r"""
    Spectrally normalized layer for Linear.
    Attributes:
        in_features (int): Input feature dimensions.
        out_features (int): Output feature dimensions.
    """

    def __init__(self, in_features, out_features, *args, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, *args, **kwargs)
        SpectralNorm.__init__(
            self, n_dim=out_features, num_iters=kwargs.get("num_iters", 1)
        )

    def forward(self, x):
        return F.linear(input=x, weight=self.sn_weights(), bias=self.bias)


class GBlock(nn.Module):
    r"""
    Skip connection block for generator.
    Uses bilinear (rather than nearest) interpolation, and align_corners
    set to False. This is as per how torchvision does upsampling, as seen in:
    https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/_utils.py
    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.c1 = SNConv2d(in_channels, out_channels, 3, 1, 1)
        self.c2 = SNConv2d(out_channels, out_channels, 3, 1, 1)
        self.c3 = SNConv2d(in_channels, out_channels, 1, 1, 0)
        self.b1 = nn.BatchNorm2d(in_channels)
        self.b2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c3.weight.data, 1.0)

    def _upsample_conv(self, x, conv):
        return conv(
            F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        )

    def _residual(self, x):
        h = x
        h = self.b1(h)
        h = self.activation(h)
        h = self._upsample_conv(h, self.c1)
        h = self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def _shortcut(self, x):
        x = self._upsample_conv(x, self.c3)
        return x

    def forward(self, x):
        return self._residual(x) + self._shortcut(x)


class DBlock(nn.Module):
    """
    Residual block for discriminator.
    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        downsample (bool): Enable 2x downsampling using avgpool.
    """

    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()

        self.downsample = downsample

        self.c1 = SNConv2d(in_channels, in_channels, 3, 1, 1)
        self.c2 = SNConv2d(in_channels, out_channels, 3, 1, 1)
        if downsample:
            self.c3 = SNConv2d(in_channels, out_channels, 1, 1, 0)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2.0))
        if downsample:
            nn.init.xavier_uniform_(self.c3.weight.data, 1.0)

    def _residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        return h

    def _shortcut(self, x):
        if self.downsample:
            x = self.c3(x)
            x = F.avg_pool2d(x, 2)
        return x

    def forward(self, x):
        return self._residual(x) + self._shortcut(x)


class Generator(nn.Module):
    r"""
    Generator with skip connections.
    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bw (int): Starting width for upsampling generator output to an image.
        nc (int): Final output channel dimension.
    """

    def __init__(self, nz=128, ngf=256, bw=4, nc=3):
        super().__init__()

        self.bw = bw

        self.l1 = nn.Linear(nz, (bw ** 2) * ngf)
        self.block2 = GBlock(ngf, ngf)
        self.block3 = GBlock(ngf, ngf)
        self.block4 = GBlock(ngf, ngf)
        self.b5 = nn.BatchNorm2d(ngf)
        self.c6 = nn.Conv2d(ngf, nc, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c6.weight.data, 1.0)

    def forward(self, x):
        h = self.l1(x)
        h = h.view(h.size(0), -1, self.bw, self.bw)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.b5(h)
        h = self.activation(h)
        h = self.c6(h)
        y = torch.tanh(h)
        return y


class Discriminator(nn.Module):
    r"""
    Discriminator with residual blocks.
    Attributes:
        nc (int): Final output channel dimension.
        ndf (int): Variable controlling discriminator feature map sizes.
    """

    def __init__(self, nc=3, ndf=128):
        super().__init__()

        self.block1 = DBlock(nc, ndf)
        self.block2 = DBlock(ndf, ndf)
        self.block3 = DBlock(ndf, ndf, downsample=False)
        self.block4 = DBlock(ndf, ndf, downsample=False)
        self.l5 = SNLinear(ndf, 1)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.l5.weight.data, 1.0)

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.activation(h)
        h = self.block2(h)
        h = self.activation(h)
        h = self.block3(h)
        h = self.activation(h)
        h = self.block4(h)
        h = self.activation(h)
        h = torch.sum(h, dim=(2, 3))
        y = self.l5(h)
        return y
