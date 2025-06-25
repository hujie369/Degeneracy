"""
该文件用于定义自定义QAT量化模块, 以控制量化训练时的推理流程
"""

import torch
from torch.nn.quantized import FloatFunctional
from torch.ao.nn.intrinsic.qat import ConvBnReLU2d, ConvBn2d
from torch.ao.nn.intrinsic.quantized import ConvReLU2d
from torch.ao.nn.quantized import Conv2d
from torch.nn.utils import fuse_conv_bn_weights
from quantization.fake_quantizer import MyFakeQuantize


class NewConvBn2d(ConvBn2d):
    """
    改进了_forward_approximate方法
    """
    def _forward_approximate(self, input):
        """Approximated method to fuse conv and bn. It requires only one forward pass.
        conv_orig = conv / scale_factor where scale_factor = bn.weight / running_std
        """
        if (
            isinstance(self.weight_fake_quant, MyFakeQuantize) and
            self.weight_fake_quant.is_per_channel
            ):
            scaled_weight = self.weight_fake_quant(self.weight)
            conv_orig = self._conv_forward(input, scaled_weight, self.bias)
        else:
            assert self.bn.running_var is not None
            running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
            scale_factor = self.bn.weight / running_std
            weight_shape = [1] * len(self.weight.shape)
            weight_shape[0] = -1
            bias_shape = [1] * len(self.weight.shape)
            bias_shape[1] = -1
            scaled_weight = self.weight_fake_quant(self.weight * scale_factor.reshape(weight_shape))
            # using zero bias here since the bias for original conv
            # will be added later
            if self.bias is not None:
                zero_bias = torch.zeros_like(self.bias, dtype=input.dtype)
            else:
                zero_bias = torch.zeros(self.out_channels, device=scaled_weight.device, dtype=input.dtype)
            conv = self._conv_forward(input, scaled_weight, zero_bias)
            conv_orig = conv / scale_factor.reshape(bias_shape)
            if self.bias is not None:
                conv_orig = conv_orig + self.bias.reshape(bias_shape)

        conv = self.bn(conv_orig)
        return conv


class NewConvBnReLU2d(ConvBnReLU2d):
    """
    修改forward方法，使用NewConvBn2d的推理方法
    """
    def forward(self, input):
        return torch.nn.functional.relu(NewConvBn2d._forward_approximate(self, input))


class NewConvReLU2d(ConvReLU2d):
    """
    修改了from_float方法
    """
    @classmethod
    def from_float(cls, mod):
        if type(mod) == NewConvBnReLU2d:
            # 将卷积层参数与BN层参数进行融合
            mod.weight, mod.bias = fuse_conv_bn_weights(
                mod.weight, mod.bias, mod.bn.running_mean, mod.bn.running_var,
                mod.bn.eps, mod.bn.weight, mod.bn.bias)
            # 对weight_fake_quantize中的量化参数scale进行修正
            if (
                isinstance(mod.weight_fake_quant, MyFakeQuantize) and
                mod.weight_fake_quant.is_per_channel
                ):
                # (outc,)
                bn_var_rsqrt = torch.rsqrt(mod.bn.running_var + mod.bn.eps)
                new_scale = mod.weight_fake_quant.scale * mod.bn.weight * bn_var_rsqrt
                mod.weight_fake_quant.scale.data = new_scale.detach()
        return super().from_float(mod)


class NewConv2d(Conv2d):
    """
    修改了from_float方法
    """
    @classmethod
    def from_float(cls, mod):
        if type(mod) == NewConvBn2d:
            # 将卷积层参数与BN层参数进行融合
            mod.weight, mod.bias = fuse_conv_bn_weights(
                mod.weight, mod.bias, mod.bn.running_mean, mod.bn.running_var,
                mod.bn.eps, mod.bn.weight, mod.bn.bias)
            # 对weight_fake_quantize中的量化参数scale进行修正
            if (
                isinstance(mod.weight_fake_quant, MyFakeQuantize) and
                mod.weight_fake_quant.is_per_channel
                ):
                # (outc,)
                bn_var_rsqrt = torch.rsqrt(mod.bn.running_var + mod.bn.eps)
                new_scale = mod.weight_fake_quant.scale * mod.bn.weight * bn_var_rsqrt
                mod.weight_fake_quant.scale.data = new_scale.detach()
        return super().from_float(mod)


# 修改prepare_qat的mapping和qat convert的mapping，使其对应自定义的qat模块
from torch.ao.quantization.quantization_mappings import get_default_qat_module_mappings, get_default_static_quant_module_mappings
from torch.ao.nn.intrinsic import ConvBnReLU2d as FusedConvBnReLU2d
from torch.ao.nn.intrinsic import ConvBn2d as FusedConvBn2d


def getMappings(default=True):
    quantized_mappings = get_default_static_quant_module_mappings()
    qat_mappings = get_default_qat_module_mappings()
    if not default:
        qat_mappings[FusedConvBnReLU2d] = NewConvBnReLU2d
        qat_mappings[FusedConvBn2d] = NewConvBn2d
        quantized_mappings[NewConvBnReLU2d] = NewConvReLU2d
        quantized_mappings[NewConvBn2d] = NewConv2d
    return qat_mappings, quantized_mappings


def my_freeze_bn_stats(mod):
    if type(mod) in {NewConvBnReLU2d, NewConvBn2d}:
        mod.freeze_bn_stats()