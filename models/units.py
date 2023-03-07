import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from scipy import signal


def coarse_center(img, img_ppi=500):
    # seg = sndi.gaussian_filter(img, sigma=19 * img_ppi / 500)
    # seg = sndi.grey_opening(seg, size=round(5 * img_ppi / 500))
    img = np.rint(img).astype(np.uint8)
    ksize1 = int(19 * img_ppi / 500)
    ksize2 = int(5 * img_ppi / 500)
    seg = cv2.GaussianBlur(img, ksize=(ksize1, ksize1), sigmaX=0, borderType=cv2.BORDER_REPLICATE)
    seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize2, ksize2)))
    seg = seg.astype(np.float32)

    grid = np.stack(np.meshgrid(*[np.arange(x) for x in img.shape[:2]], indexing="ij")).reshape(2, -1)
    img_c = (seg.reshape(1, -1) * grid).sum(1) / seg.sum().clip(1e-6, None)
    return img_c


def custom_linspace(num, bin_type=None, delta=False):
    x = torch.linspace(-1, 1, num + 1)
    if bin_type is not None:
        x = interval_location(x, bin_type)
    if delta:
        return x[1:] - x[:-1]
    else:
        return (x[:-1] + x[1:]) / 2


def interval_location(x, bin_type="x1"):
    x = x.clamp(-1, 1)

    if bin_type == "x1":
        return x
    elif bin_type == "x2":
        return x.abs() ** 2
    elif bin_type == "invprop":
        return x / (2 - x.abs())
    elif bin_type == "arcsin":
        return torch.arcsin(x) / (np.pi / 2)
    else:
        raise ValueError(f"Unsupported bin_type:{bin_type}")


def interval_location_inverse(x, bin_type="x1"):
    x = x.clamp(-1, 1)

    if bin_type == "x1":
        return x
    elif bin_type == "x2":
        return torch.sign(x) * torch.sqrt(x.abs())
    elif bin_type == "invprop":
        return 2 * x / (x.abs() + 1)
    elif bin_type == "arcsin":
        return torch.sin(np.pi / 2 * x)
    else:
        raise ValueError(f"Unsupported bin_type {bin_type}")


def theta2R(theta, is_deg=True):
    if is_deg:
        theta = torch.deg2rad(theta)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    R = torch.tensor([cos_theta, -sin_theta, sin_theta, cos_theta]).reshape(1, 2, 2)
    return R


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel, padding=padding), nn.ReLU(inplace=True))


def convbnrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ImageGradient(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)
        kernel_y = [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        grad_x = F.conv2d(x, self.weight_x, padding=1)
        grad_y = F.conv2d(x, self.weight_y, padding=1)
        return grad_x, grad_y


class FingerprintCompose(nn.Module):
    def __init__(self, win_size=8, do_norm=False, m0=0, var0=1.0, eps=1e-6):
        super().__init__()
        self.win_size = max(3, win_size // 2 * 2 + 1)

        self.norm = NormalizeModule(m0=m0, var0=var0, eps=eps) if do_norm else nn.Identity()
        self.conv_grad = ImageGradient()
        self.conv_gaussian = ImageGaussian(self.win_size, self.win_size / 3.0)
        mean_kernel = torch.ones([self.win_size, self.win_size], dtype=torch.float32)[None, None] / self.win_size ** 2
        self.weight_avg = nn.Parameter(data=mean_kernel, requires_grad=False)

    def forward(self, x):
        assert x.size(1) == 1

        Gx, Gy = self.conv_grad(x)
        Gxx = self.conv_gaussian(Gx ** 2)
        Gyy = self.conv_gaussian(Gy ** 2)
        Gxy = self.conv_gaussian(-Gx * Gy)
        sin2 = F.conv2d(2 * Gxy, self.weight_avg, padding=self.win_size // 2)
        cos2 = F.conv2d(Gxx - Gyy, self.weight_avg, padding=self.win_size // 2)

        x = torch.cat((x, sin2, cos2), dim=1)

        x = self.norm(x)

        return x


class FingerprintRepeat(nn.Module):
    def __init__(self, num_out=3) -> None:
        super().__init__()
        self.num_out = num_out

    def forward(self, x):
        assert x.size(1) == 1

        return x.repeat(1, self.num_out, 1, 1)


class DoubleConv(nn.Module):
    def __init__(self, in_chn, out_chn, do_bn=True, do_res=False):
        super().__init__()
        self.conv = (
            nn.Sequential(
                nn.Conv2d(in_chn, out_chn, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_chn),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_chn, out_chn, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_chn),
                nn.LeakyReLU(inplace=True),
            )
            if do_bn
            else nn.Sequential(
                nn.Conv2d(in_chn, out_chn, 3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_chn, out_chn, 3, padding=1),
                nn.LeakyReLU(inplace=True),
            )
        )

        self.do_res = do_res
        if self.do_res:
            if out_chn < in_chn:
                self.original = nn.Conv2d(in_chn, out_chn, 1, padding=0)
            elif out_chn == in_chn:
                self.original = nn.Identity()
            else:
                self.original = ChannelPad2d(out_chn - in_chn)

    def forward(self, x):
        out = self.conv(x)
        if self.do_res:
            res = self.original(x)
            out = out + res
        return out


class ChannelPad2d(nn.Module):
    def __init__(self, after_C, before_C=0, value=0) -> None:
        super().__init__()
        self.before_C = before_C
        self.after_C = after_C
        self.value = value

    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, self.before_C, self.after_C), value=self.value)
        return out


def gaussian_fn(win_size, std):
    n = torch.arange(0, win_size) - (win_size - 1) / 2.0
    sig2 = 2 * std ** 2
    w = torch.exp(-(n ** 2) / sig2) / (np.sqrt(2 * np.pi) * std)
    return w


def gkern(win_size, std):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = gaussian_fn(win_size, std)
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d


class ImageGaussian(nn.Module):
    def __init__(self, win_size, std):
        super().__init__()
        self.win_size = max(3, win_size // 2 * 2 + 1)

        n = np.arange(0, win_size) - (win_size - 1) / 2.0
        gkern1d = np.exp(-(n ** 2) / (2 * std ** 2)) / (np.sqrt(2 * np.pi) * std)
        gkern2d = np.outer(gkern1d, gkern1d)
        gkern2d = torch.FloatTensor(gkern2d).unsqueeze(0).unsqueeze(0)
        self.gkern2d = nn.Parameter(data=gkern2d, requires_grad=False)

    def forward(self, x):
        x_gaussian = F.conv2d(x, self.gkern2d, padding=self.win_size // 2)
        return x_gaussian


class NormalizeInput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)

    def forward(self, input):
        input = (input - input.mean(dim=(-1, -2), keepdim=True)) / input.std(dim=(-1, -2), keepdim=True)
        input = input * self.std.type_as(input) + self.mean.type_as(input)
        return input


class Gray2RGB(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, gray):
        assert gray.size(1) == 1
        return gray.repeat(1, 3, 1, 1)


class CrossAttention(nn.Module):
    def __init__(self, k_chn) -> None:
        super().__init__()
        self.C = k_chn
        self.W = nn.Linear(k_chn, k_chn, bias=False)

    def forward(self, a, b):
        a_shape = a.shape[2:]
        b_shape = b.shape[2:]

        a_flat = a.flatten(2)  # (B,C,WH)
        b_flat = b.flatten(2)
        S = torch.bmm(self.W(a_flat.transpose(1, 2)), b_flat)

        a_new = torch.bmm(a_flat, torch.softmax(S, 1)).reshape(-1, self.C, *a_shape)
        b_new = torch.bmm(b_flat, torch.softmax(S.transpose(1, 2), 1)).reshape(-1, self.C, *b_shape)

        return a_new, b_new


class DownSample(nn.Module):
    def __init__(self, scale_factor=2) -> None:
        super().__init__()
        self.scale_factor = 1.0 / scale_factor

    def forward(self, input, mode="nearest", align_corners=False):
        return F.interpolate(input, scale_factor=self.scale_factor, mode=mode, align_corners=align_corners)


class AlexNet(nn.Module):
    # ?????
    def __init__(self, num_out=3, num_layers=[64, 192, 384, 256, 256]):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )


def positionalencoding2d(channels, height, width):
    """
    :param channels: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: channels*height*width position matrix
    """
    if channels % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with " "odd dimension (got dim={:d})".format(channels))
    pe = torch.zeros(channels, height, width)
    # Each dimension use half of channels
    channels = int(channels / 2)
    div_term = 10000.0 ** (-torch.arange(0, channels, 2).float() / channels)
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe[0:channels:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:channels:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[channels::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[channels + 1 :: 2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


class Encoder(nn.Module):
    def __init__(self, num_layers, do_bn, in_chn=3, do_res=False, do_pos=False):
        super().__init__()
        self.n_layer = len(num_layers)
        self.do_pos = do_pos

        self.layer0 = DoubleConv(in_chn, num_layers[0], do_bn)
        if do_pos:
            in_channel = 2 * num_layers[0]
        else:
            in_channel = num_layers[0]
        for ii, out_channel in enumerate(num_layers[1:]):
            setattr(self, f"pool{ii}", nn.MaxPool2d(2, 2))
            setattr(self, f"layer{ii+1}", DoubleConv(in_channel, out_channel, do_bn=do_bn, do_res=do_res))
            in_channel = out_channel

    def forward(self, input):
        y = self.layer0(input)
        if self.do_pos:
            B, C, H, W = y.shape
            pos_enc = positionalencoding2d(C, H, W)
            y = torch.cat((y, pos_enc.type_as(y).repeat(B, 1, 1, 1)), dim=1)
        out = [y]
        for ii in range(self.n_layer - 1):
            y = getattr(self, f"pool{ii}")(y)
            y = getattr(self, f"layer{ii+1}")(y)
            out.append(y)

        return out


class Decoder(nn.Module):
    def __init__(self, in_channel, num_layers, out_channel, expansion=1, do_bn=True) -> None:
        super().__init__()
        self.n_layer = len(num_layers)

        for ii, cur_channel in enumerate(num_layers):
            setattr(self, f"layer{ii}", DoubleConv(in_channel * expansion, cur_channel * expansion, do_bn))
            setattr(self, f"upsample{ii}", nn.ConvTranspose2d(cur_channel * expansion, cur_channel * expansion, 2, 2))
            in_channel = cur_channel

        self.out = nn.Sequential(
            nn.Conv2d(in_channel * expansion, in_channel * expansion, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channel * expansion),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channel * expansion, out_channel, 1),
        )

    def forward(self, input):
        y = input
        out = []
        for ii in range(self.n_layer):
            y = getattr(self, f"layer{ii}")(y)
            out.append(y)
            y = getattr(self, f"upsample{ii}")(y)
        y = self.out(y)
        out.append(y)

        return out


class DecoderSkip(nn.Module):
    def __init__(self, in_channel, num_layers, out_channel, expansion=1, do_bn=True) -> None:
        super().__init__()
        self.n_layer = len(num_layers)

        for ii, cur_channel in enumerate(num_layers):
            setattr(self, f"upsample{ii}", nn.ConvTranspose2d(in_channel * expansion, cur_channel * expansion, 2, 2))
            setattr(self, f"layer{ii}", DoubleConv((cur_channel + cur_channel) * expansion, cur_channel * expansion, do_bn))
            in_channel = cur_channel

        self.out = nn.Sequential(nn.Conv2d(in_channel * expansion, out_channel, 1))

    def forward(self, inputs):
        y = inputs[0]
        for ii in range(self.n_layer):
            y = getattr(self, f"upsample{ii}")(y)
            y = getattr(self, f"layer{ii}")(torch.cat((inputs[ii + 1], y), dim=1))
        y = self.out(y)

        return y


class DecoderSkip2(nn.Module):
    def __init__(self, in_channel, num_layers, expansion=1, do_bn=True) -> None:
        super().__init__()
        self.n_layer = len(num_layers)

        for ii, cur_channel in enumerate(num_layers):
            setattr(self, f"upsample{ii}", nn.ConvTranspose2d(in_channel * expansion, cur_channel * expansion, 2, 2))
            setattr(self, f"layer{ii}", DoubleConv((cur_channel + cur_channel) * expansion, cur_channel * expansion, do_bn))
            in_channel = cur_channel

    def forward(self, inputs):
        y = inputs[0]
        for ii in range(self.n_layer):
            y = getattr(self, f"upsample{ii}")(y)
            y = getattr(self, f"layer{ii}")(torch.cat((inputs[ii + 1], y), dim=1))

        return y


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class BasicConv(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-6, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialGate, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return scale


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class FastCartoonTexture(nn.Module):
    def __init__(self, sigma=2.5, eps=1e-6) -> None:
        super().__init__()
        self.sigma = sigma
        self.eps = eps
        self.cmin = 0.3
        self.cmax = 0.7
        self.lim = 20

        self.img_grad = ImageGradient()

    def lowpass_filtering(self, img, L):
        img_fft = torch.fft.fftshift(torch.fft.fft2(img), dim=(-2, -1)) * L

        img_rec = torch.fft.ifft2(torch.fft.fftshift(img_fft, dim=(-2, -1)))
        img_rec = torch.real(img_rec)

        return img_rec

    def gradient_norm(self, img):
        Gx, Gy = self.img_grad(img)
        return torch.sqrt(Gx ** 2 + Gy ** 2) + self.eps

    def forward(self, input):
        H, W = input.size(-2), input.size(-1)
        grid_y, grid_x = torch.meshgrid(torch.linspace(-0.5, 0.5, H), torch.linspace(-0.5, 0.5, W), indexing="ij")
        grid_radius = torch.sqrt(grid_x ** 2 + grid_y ** 2) + self.eps

        L = (1.0 / (1 + (2 * np.pi * grid_radius * self.sigma) ** 4)).type_as(input)[None, None]

        grad_img1 = self.gradient_norm(input)
        grad_img1 = self.lowpass_filtering(grad_img1, L)

        img_low = self.lowpass_filtering(input, L)
        grad_img2 = self.gradient_norm(img_low)
        grad_img2 = self.lowpass_filtering(grad_img2, L)

        diff = grad_img1 - grad_img2
        flag = torch.abs(grad_img1)
        diff = torch.where(flag > 1, diff / flag.clamp_min(self.eps), torch.zeros_like(diff))

        weight = (diff - self.cmin) / (self.cmax - self.cmin)
        weight = torch.clamp(weight, 0, 1)

        cartoon = weight * img_low + (1 - weight) * input
        texture = (input - cartoon + self.lim) * 255 / (2 * self.lim)
        texture = torch.clamp(texture, 0, 255)
        return texture


class NormalizeModule(nn.Module):
    def __init__(self, m0=0.0, var0=1.0, eps=1e-6):
        super(NormalizeModule, self).__init__()
        self.m0 = m0
        self.var0 = var0
        self.eps = eps

    def forward(self, x):
        x_m = x.mean(dim=(1, 2, 3), keepdim=True)
        x_var = x.var(dim=(1, 2, 3), keepdim=True)
        y = (self.var0 * (x - x_m) ** 2 / x_var.clamp_min(self.eps)).sqrt()
        y = torch.where(x > x_m, self.m0 + y, self.m0 - y)
        return y


class ConvBnPRelu(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_chn, out_chn, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_chn, eps=0.001, momentum=0.99)
        self.relu = nn.PReLU(out_chn, init=0)

    def forward(self, input):
        y = self.conv(input)
        y = self.bn(y)
        y = self.relu(y)
        return y


def gabor_bank(enh_ksize=25, ori_stride=2, sigma=4.5, Lambda=8, psi=0, gamma=0.5):
    grid_theta, grid_y, grid_x = torch.meshgrid(
        torch.arange(-90, 90, ori_stride, dtype=torch.float32),
        torch.arange(-(enh_ksize // 2), enh_ksize // 2 + 1, dtype=torch.float32),
        torch.arange(-(enh_ksize // 2), enh_ksize // 2 + 1, dtype=torch.float32),
        indexing="ij",
    )
    cos_theta = torch.cos(torch.deg2rad(-grid_theta))
    sin_theta = torch.sin(torch.deg2rad(-grid_theta))

    x_theta = grid_y * sin_theta + grid_x * cos_theta
    y_theta = grid_y * cos_theta - grid_x * sin_theta
    # gabor filters
    sigma_x = sigma
    sigma_y = float(sigma) / gamma
    exp_fn = torch.exp(-0.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2))
    gb_cos = exp_fn * torch.cos(2 * np.pi * x_theta / Lambda + psi)
    gb_sin = exp_fn * torch.sin(2 * np.pi * x_theta / Lambda + psi)

    return gb_cos[:, None], gb_sin[:, None]


def cycle_gaussian_weight(ang_stride=2, to_tensor=True):
    gaussian_pdf = signal.windows.gaussian(181, 3)
    coord = np.arange(ang_stride / 2, 180, ang_stride)
    delta = np.abs(coord.reshape(1, -1, 1, 1) - coord.reshape(-1, 1, 1, 1))
    delta = np.minimum(delta, 180 - delta) + 90
    if to_tensor:
        return torch.tensor(gaussian_pdf[delta.astype(int)]).float()
    else:
        return gaussian_pdf[delta.astype(int)].astype(np.float32)


def orientation_highest_peak(x, ang_stride=2):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    filter_weight = cycle_gaussian_weight(ang_stride=ang_stride).type_as(x)
    return F.conv2d(x, filter_weight, stride=1, padding=0)


def select_max_orientation(x):
    x = x / torch.max(x, dim=1, keepdim=True).values.clamp_min(1e-6)
    x = torch.where(x > 0.999, x, torch.zeros_like(x))
    x = x / x.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return x
