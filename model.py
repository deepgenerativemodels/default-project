import math 

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralNorm(object):
    r"""
    Spectral Normalization for GANs (Miyato 2018).
    Inheritable class for performing spectral normalization of weights,
    as approximated using power iteration.
    Details: See Algorithm 1 of Appendix A (Miyato 2018).
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
    Residual block for generator.
    Uses bilinear (rather than nearest) interpolation, and align_corners
    set to False. This is as per how torchvision does upsampling, as seen in:
    https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/_utils.py
    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        upsample (bool): If True, upsamples the input feature map.
    """

    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.learnable_sc = in_channels != out_channels or upsample
        self.upsample = upsample

        self.c1 = SNConv2d(in_channels, out_channels, 3, 1, padding=1)
        self.c2 = SNConv2d(out_channels, out_channels, 3, 1, padding=1)

        self.b1 = nn.BatchNorm2d(in_channels)
        self.b2 = nn.BatchNorm2d(out_channels)

        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2.0))

        if self.learnable_sc:
            self.c_sc = SNConv2d(in_channels, out_channels, 1, 1, padding=0)

            nn.init.xavier_uniform_(self.c_sc.weight.data, 1.0)

    def _upsample_conv(self, x, conv):
        return conv(
            F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        )

    def _residual(self, x):
        h = x
        h = self.b1(h)
        h = self.activation(h)
        h = self._upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def _shortcut(self, x):
        if self.learnable_sc:
            x = self._upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def forward(self, x):
        return self._residual(x) + self._shortcut(x)


class DBlock(nn.Module):
    """
    Residual block for discriminator.
    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        downsample (bool): If True, downsamples the input feature map.
    """

    def __init__(
        self, in_channels, out_channels, downsample=False
    ):
        super().__init__()
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample

        self.c1 = SNConv2d(in_channels, in_channels, 3, 1, 1)
        self.c2 = SNConv2d(in_channels, out_channels, 3, 1, 1)

        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2.0))

        if self.learnable_sc:
            self.c_sc = SNConv2d(in_channels, out_channels, 1, 1, 0)

            nn.init.xavier_uniform_(self.c_sc.weight.data, 1.0)

    def _residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        return h

    def _shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            return F.avg_pool2d(x, 2) if self.downsample else x
        else:
            return x

    def forward(self, x):
        return self._residual(x) + self._shortcut(x)


class DBlockOptimized(nn.Module):
    """
    Optimized residual block for discriminator. This is used as the first residual block,
    where there is a definite downsampling involved. Follows the official SNGAN reference implementation
    in chainer.
    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.c1 = SNConv2d(in_channels, out_channels, 3, 1, 1)
        self.c2 = SNConv2d(out_channels, out_channels, 3, 1, 1)
        self.c_sc = SNConv2d(in_channels, out_channels, 1, 1, 0)

        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c_sc.weight.data, 1.0)

    def _residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = F.avg_pool2d(h, 2)
        return h

    def _shortcut(self, x):
        return self.c_sc(F.avg_pool2d(x, 2))

    def forward(self, x):
        return self._residual(x) + self._shortcut(x)


class Generator(nn.Module):
    r"""
    ResNet backbone generator for SNGAN.
    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
    """

    def __init__(self, nz=128, ngf=1024, bottom_width=4):
        super().__init__()

        self.l1 = nn.Linear(nz, (bottom_width ** 2) * ngf)
        self.unfatten = nn.Unflatten(1, (-1, bottom_width, bottom_width))
        self.block2 = GBlock(ngf, ngf >> 1, upsample=True)
        self.block3 = GBlock(ngf >> 1, ngf >> 2, upsample=True)
        self.block4 = GBlock(ngf >> 2, ngf >> 3, upsample=True)
        self.block5 = GBlock(ngf >> 3, ngf >> 4, upsample=True)
        self.b6 = nn.BatchNorm2d(ngf >> 4)
        self.c6 = nn.Conv2d(ngf >> 4, 3, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c6.weight.data, 1.0)

    def forward(self, x):
        h = self.l1(x)
        h = self.unfatten(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.b6(h)
        h = self.activation(h)
        h = torch.tanh(self.c6(h))
        return h


class Discriminator(nn.Module):
    r"""
    ResNet backbone discriminator for SNGAN.
    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
    """

    def __init__(self, ndf=1024):
        super().__init__()

        self.block1 = DBlockOptimized(3, ndf >> 4)
        self.block2 = DBlock(ndf >> 4, ndf >> 3, downsample=True)
        self.block3 = DBlock(ndf >> 3, ndf >> 2, downsample=True)
        self.block4 = DBlock(ndf >> 2, ndf >> 1, downsample=True)
        self.block5 = DBlock(ndf >> 1, ndf, downsample=True)
        self.l6 = SNLinear(ndf, 1)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.l6.weight.data, 1.0)

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        h = torch.sum(h, dim=(2, 3))
        output = self.l6(h)
        return output
