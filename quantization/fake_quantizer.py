"""
this file is used to build FakeQuantize that is used in QAT
gradscale, roundpass and quantize_ draw on this paper:
LEARNED STEP SIZE QUANTIZATION, doi: 10.48550/arxiv.1902.08153
"""

import torch
from torch.ao.quantization import FakeQuantizeBase

def gradscale(x, scale):
    yout = x
    ygrad = x * scale
    y = (yout - ygrad).detach() + ygrad
    return y

def roundpass(x):
    yout = torch.round(x)
    ygrad = x
    y = (yout - ygrad).detach() + ygrad
    return y

def quantize_(x, s, z, qmin, qmax, is_per_channel):
    # weight'shape: (outc, inc ,ks1, ks2)
    # act's shape: (bs, c, h, w)
    # s and z: (1) or (outc,)
    N = torch.tensor(x.numel(), dtype=torch.float, device=x.device)
    if is_per_channel:
        N = N / x.shape[0]
        # reshape s and z to the shape of x
        new_shape = [1] * x.dim()
        new_shape[0] = -1
        s = s.reshape(new_shape)
        z = z.reshape(new_shape)

    if s.requires_grad:
        gradScaleFactor = torch.rsqrt((qmax - qmin) * N)
        s = gradscale(s, gradScaleFactor)
    x = x / s + z
    x = torch.clamp(x, qmin, qmax)
    x = roundpass(x) - z
    x = x * s
    return x


class VitFakeQuantizer(torch.nn.Module):
    def __init__(self, outc, quant_min=-128, quant_max=127, is_per_channel=False, requires_grad=True):
        # outc = !1 if is_per_channel else 1
        super().__init__()
        outc = outc if is_per_channel else 1
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.is_per_channel = is_per_channel

        self.scale = torch.nn.Parameter(torch.ones(outc), requires_grad=requires_grad)
        self.zero_point = torch.nn.Parameter(torch.zeros(outc).to(torch.int), requires_grad=False)

        self.fake_quant_enabled = True

    def set_scale(self, x=True):
        self.scale.requires_grad_(x)

    def forward(self, inputs):
        if self.fake_quant_enabled:
            inputs = quantize_(inputs, self.scale, self.zero_point, self.quant_min, self.quant_max, self.is_per_channel)

        return inputs

    def init_(self, input):
        input = input.detach()
        if self.is_per_channel:
            r, _ = torch.abs(input.view((input.shape[0], -1))).max(dim=1)
        else:
            r = torch.addcdivabs(input).max()
        self.scale.data = (r * 2) / (self.quant_max - self.quant_min)


class ResFakeQuantize(FakeQuantizeBase):
    sacle: torch.Tensor
    zero_point: torch.Tensor

    def __init__(self, quant_min=None, quant_max=None, **observer_kwargs):
        super().__init__()
        # Populate quant_min/quant_max to observer_kwargs if valid
        if quant_min is not None and quant_max is not None:
            assert quant_min <= quant_max, \
                'quant_min must be less than or equal to quant_max'
            dtype = observer_kwargs.get("dtype", torch.quint8)

            assert torch.iinfo(dtype).min <= quant_min, 'quant_min out of bound'
            assert quant_max <= torch.iinfo(dtype).max, 'quant_max out of bound'
            observer_kwargs.update({"quant_min": quant_min, "quant_max": quant_max})

        # TODO: keeping self.quant_min/max for BC; remove after a couple releases
        # Users should use self.activation_post_process.quant_min
        self.quant_min = quant_min
        self.quant_max = quant_max

        self.dtype = dtype
        self.qscheme = observer_kwargs.get("qscheme", torch.per_tensor_affine)
        self.ch_axis = observer_kwargs.get("ch_axis", 0)

        self.is_per_channel = self.qscheme in [torch.per_channel_symmetric,torch.per_channel_affine,
                                               torch.per_channel_affine_float_qparams]
        # 初始化sacle和zero_point
        zp = 128 if dtype == torch.quint8 else 0
        self.scale = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float), requires_grad=True)
        self.zero_point = torch.nn.Parameter(torch.tensor([zp], dtype=torch.int), requires_grad=False)

    @torch.jit.export
    def calculate_qparams(self):
        return self.scale, self.zero_point

    def forward(self, X):
        if self.observer_enabled[0] != 1:
            self.scale.requires_grad_(False)

        if self.fake_quant_enabled[0] == 1:
            X = quantize_(X, self.scale, self.zero_point, self.quant_min, self.quant_max, self.is_per_channel)
        return X

    @torch.jit.export
    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={}, ' \
               'quant_min={}, quant_max={}, dtype={}, qscheme={}, ch_axis={}, ' \
               'scale={}, zero_point={}'.format(
                   self.fake_quant_enabled, self.observer_enabled,
                   self.quant_min, self.quant_max,
                   self.dtype, self.qscheme, self.ch_axis, self.scale, self.zero_point)
