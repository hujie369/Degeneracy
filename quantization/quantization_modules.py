"""
该文件用于定义自定义QAT量化模块, 以控制量化训练时的推理流程
"""

import torch
from torch.nn.quantized import FloatFunctional
from torch.ao.nn.intrinsic.qat import ConvBnReLU2d, ConvBn2d
from torch.ao.nn.intrinsic.quantized import ConvReLU2d
from torch.ao.nn.quantized import Conv2d
from torch.nn.utils import fuse_conv_bn_weights
from quantization.fake_quantizer import ResFakeQuantize
from torch.ao.quantization import QuantStub
from torch.ao.nn.qat import Linear


class NewConvBn2d(ConvBn2d):
    """
    改进了_forward_approximate方法
    """
    def _forward_approximate(self, input):
        """Approximated method to fuse conv and bn. It requires only one forward pass.
        conv_orig = conv / scale_factor where scale_factor = bn.weight / running_std
        """
        if (
            isinstance(self.weight_fake_quant, ResFakeQuantize) and
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
    修改了from_float方法和forward方法
    """
    def _get_name(self):
        return 'NewQuantizedConvReLU2d'
    
    def forward(self, input):
        """
        正常推理完之后, 可能需要对输出进行截断
        """
        output = torch.ops.quantized.conv2d_relu(
            input, self._packed_params, self.scale, self.zero_point)
        if self.isLow8bit:
            output = output.int_repr().clamp(self.qmin, self.qmax)
            output = torch._make_per_tensor_quantized_tensor(output, self.scale, self.zero_point)
        return output

    @classmethod
    def from_float(cls, mod):
        new_cls = super().from_float(mod)
        assert hasattr(mod, "activation_post_process")
        new_cls.isLow8bit = False
        if mod.activation_post_process.quant_max < 255:
            new_cls.isLow8bit = True
            new_cls.qmin = mod.activation_post_process.quant_min
            new_cls.qmax = mod.activation_post_process.quant_max

        return new_cls


class NewConv2d(Conv2d):
    """
    修改了from_float方法和forward方法
    """
    def _get_name(self):
        return 'NewQuantizedConv2d'

    def forward(self, input):
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")

        output = torch.ops.quantized.conv2d(
            input, self._packed_params, self.scale, self.zero_point)
        if self.isLow8bit:
            output = output.int_repr().clamp(self.qmin, self.qmax)
            output = torch._make_per_tensor_quantized_tensor(output, self.scale, self.zero_point)
        return output

    @classmethod
    def from_float(cls, mod):
        new_cls = super().from_float(mod)
        assert hasattr(mod, "activation_post_process")
        new_cls.isLow8bit = False
        if mod.activation_post_process.quant_max < 255:
            new_cls.isLow8bit = True
            new_cls.qmin = mod.activation_post_process.quant_min
            new_cls.qmax = mod.activation_post_process.quant_max
        
        return new_cls


class NewQFunctional(torch.ao.nn.quantized.QFunctional):
    def add_relu(self, x, y):
        output = torch.ops.quantized.add_relu(x, y, scale=self.scale, zero_point=self.zero_point)
        if self.isLow8bit:
            output = output.int_repr().clamp(self.qmin, self.qmax)
            output = torch._make_per_tensor_quantized_tensor(output, self.scale, self.zero_point)
        return output

    @classmethod
    def from_float(cls, mod):
        assert type(mod) == FloatFunctional,\
            "QFunctional.from_float expects an instance of FloatFunctional"
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_cls = NewQFunctional()
        new_cls.scale = float(scale)
        new_cls.zero_point = int(zero_point)

        new_cls.isLow8bit = False
        if mod.activation_post_process.quant_max < 255:
            new_cls.isLow8bit = True
            new_cls.qmin = mod.activation_post_process.quant_min
            new_cls.qmax = mod.activation_post_process.quant_max

        return new_cls


class NewQuantize(torch.ao.nn.quantized.Quantize):
    def forward(self, X):
        output = torch.quantize_per_tensor(X, float(self.scale), int(self.zero_point), self.dtype)
        if self.isLow8bit:
            output = output.int_repr().clamp(self.qmin, self.qmax)
            output = torch._make_per_tensor_quantized_tensor(output, float(self.scale), int(self.zero_point))
        return output

    @staticmethod
    def from_float(mod):
        assert hasattr(mod, 'activation_post_process')
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_cls = NewQuantize(scale.float().item(), zero_point.long().item(), mod.activation_post_process.dtype)

        new_cls.isLow8bit = False
        if mod.activation_post_process.quant_max < 255:
            new_cls.isLow8bit = True
            new_cls.qmin = mod.activation_post_process.quant_min
            new_cls.qmax = mod.activation_post_process.quant_max

        return new_cls


class NewLinear(torch.ao.nn.quantized.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.ops.quantized.linear(
            x, self._packed_params._packed_params, self.scale, self.zero_point)
        if self.isLow8bit:
            output = output.int_repr().clamp(self.qmin, self.qmax)
            output = torch._make_per_tensor_quantized_tensor(output, self.scale, self.zero_point)
        return output

    @classmethod
    def from_float(cls, mod):
        # 绑定rmin, rmax参数, 以便对activation进行正确的低bit量化
        new_cls = super().from_float(mod)
        assert hasattr(mod, "activation_post_process")
        new_cls.isLow8bit = False
        if mod.activation_post_process.quant_max < 255:
            new_cls.isLow8bit = True
            new_cls.qmin = mod.activation_post_process.quant_min
            new_cls.qmax = mod.activation_post_process.quant_max
        
        return new_cls


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
        quantized_mappings[FloatFunctional] = NewQFunctional
        quantized_mappings[QuantStub] = NewQuantize
        quantized_mappings[Linear] = NewLinear
    return qat_mappings, quantized_mappings


# 用于冻结bn层并融合参数，注意修改了权重参数和量化参数
def my_freeze_bn_stats(mod):
    if type(mod) in {NewConvBnReLU2d, NewConvBn2d}:
        mod.freeze_bn_stats()
        # 融合参数
        mod.weight, mod.bias = fuse_conv_bn_weights(
            mod.weight, mod.bias, mod.bn.running_mean, mod.bn.running_var,
            mod.bn.eps, mod.bn.weight, mod.bn.bias)
        # 注意给bias设置梯度更新
        mod.weight.requires_grad = True
        mod.bias.requires_grad = True

        # 调整scale
        bn_var_rsqrt = torch.rsqrt(mod.bn.running_var + mod.bn.eps)
        new_scale = mod.weight_fake_quant.scale * mod.bn.weight * bn_var_rsqrt
        mod.weight_fake_quant.scale.data = new_scale.detach()
        mod.weight_fake_quant.scale.requires_grad = True
        # 取消bn层
        mod.bn = torch.nn.Identity()


# *----------------------------------------------------------------*
# for ViT
import torch
from torch import nn
from torch.nn import functional as F

class QatConv2d(nn.Conv2d):
    def __init__(
        self, in_channels, out_channels, kernel_size,
        stride=1, padding=0, dilation=1, groups=1, bias=True,
        padding_mode="zeros", device=None, dtype=None):
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups,
            bias, padding_mode, device, dtype)

        self.fake_quantizer = None

    def forward(self, input):
        scaled_weight = self.fake_quantizer(self.weight)
        out = self._conv_forward(input, scaled_weight, self.bias)
        return out

    @classmethod
    def from_float(cls, mod, Quantizer):
        new_mod = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                      mod.stride, mod.padding, bias=mod.bias is not None,
                      device=mod.weight.device)
        new_mod.weight.data = mod.weight.detach().clone()
        if mod.bias is not None:
            new_mod.bias.data = mod.bias.detach().clone()

        # 创建并初始化fake quantizer
        new_mod.fake_quantizer = Quantizer(mod.out_channels)
        new_mod.fake_quantizer.init_(new_mod.weight)
        return new_mod


class QatLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, 
                 device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.fake_quantizer = None

    def forward(self, input):
        scaled_weight = self.fake_quantizer(self.weight)
        return F.linear(input, scaled_weight, self.bias)

    @classmethod
    def from_float(cls, mod, Quantizer):
        new_mod = cls(mod.in_features, mod.out_features, mod.bias is not None,
                      mod.weight.device)
        new_mod.weight.data = mod.weight.detach().clone()
        if mod.bias is not None:
            new_mod.bias.data = mod.bias.detach().clone()

        # 创建并初始化fake quantizer
        new_mod.fake_quantizer = Quantizer(mod.out_features)
        new_mod.fake_quantizer.init_(new_mod.weight)
        return new_mod


def _is_make_fx_tracing():
    if not torch.jit.is_scripting():
        torch_dispatch_mode_stack = (
            torch.utils._python_dispatch._get_current_dispatch_mode_stack()
        )
        return any(
            type(x) == torch.fx.experimental.proxy_tensor.ProxyTorchDispatchMode
            for x in torch_dispatch_mode_stack
        )
    else:
        return False

def _check_arg_device(x) -> bool:
    if x is not None:
        return x.device.type in [
            "cpu",
            "cuda",
            torch.utils.backend_registration._privateuse1_backend_name,
        ]
    return True

def _arg_requires_grad(x) -> bool:
    if x is not None:
        return x.requires_grad
    return False

class QatMHA(nn.MultiheadAttention):
    def __init__(
        self, embed_dim, num_heads, dropout=0.0, bias=True,
        add_bias_kv=False, add_zero_attn=False, kdim=None,
        vdim=None, batch_first=False, device=None, dtype=None):
        super().__init__(
            embed_dim, num_heads, dropout, bias, add_bias_kv,
            add_zero_attn, kdim, vdim, batch_first, device, dtype)

        self.fake_quantizer_in = None
        self.fake_quantizer_out = None


    @classmethod
    def from_float(cls, mod, Quantizer):
        new_mod = cls(mod.embed_dim, mod.num_heads, mod.dropout,
                      mod.in_proj_bias is not None,
                      batch_first=mod.batch_first,
                      device=mod.in_proj_weight.device)
        # 克隆parameters
        new_mod.in_proj_weight.data = mod.in_proj_weight.detach().clone()
        new_mod.out_proj.weight.data = mod.out_proj.weight.detach().clone()
        if mod.in_proj_bias is not None:
            new_mod.in_proj_bias.data = mod.in_proj_bias.detach().clone()
            new_mod.out_proj.bias.data = mod.out_proj.bias.detach().clone()

        # 创建并初始化fake quantizer
        new_mod.fake_quantizer_in = Quantizer(mod.embed_dim * 3)
        new_mod.fake_quantizer_in.init_(new_mod.in_proj_weight)

        new_mod.fake_quantizer_out = Quantizer(mod.out_proj.out_features)
        new_mod.fake_quantizer_out.init_(new_mod.out_proj.weight)
        return new_mod

    def forward(
        self, query, key, value, key_padding_mask=None, need_weights=True,
        attn_mask=None, average_attn_weights=True, is_causal=False):

        # NOTE: changed
        quantized_in_proj_weight = self.fake_quantizer_in(self.in_proj_weight)
        quantized_out_proj_weight = self.fake_quantizer_out(self.out_proj.weight)
        
        why_not_fast_path = ""
        if (
            (attn_mask is not None and torch.is_floating_point(attn_mask))
            or (key_padding_mask is not None)
            and torch.is_floating_point(key_padding_mask)
        ):
            why_not_fast_path = "floating-point masks are not supported for fast path."

        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        if not is_fastpath_enabled:
            why_not_fast_path = "torch.backends.mha.get_fastpath_enabled() was not True"
        elif not is_batched:
            why_not_fast_path = (
                f"input not batched; expected query.dim() of 3 but got {query.dim()}"
            )
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is None:
            why_not_fast_path = "in_proj_weight was None"
        elif query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif (self.num_heads % 2) != 0:
            why_not_fast_path = "self.num_heads is not even"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif query.is_nested and (
            key_padding_mask is not None or attn_mask is not None
        ):
            why_not_fast_path = "supplying both src_key_padding_mask and src_mask at the same time \
                                 is not supported with NestedTensor input"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                quantized_in_proj_weight,
                self.in_proj_bias,
                quantized_out_proj_weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif _is_make_fx_tracing():
                why_not_fast_path = "we are running make_fx tracing"
            elif not all(_check_arg_device(x) for x in tensor_args):
                why_not_fast_path = (
                    "some Tensor argument's device is neither one of "
                    f"cpu, cuda or {torch.utils.backend_registration._privateuse1_backend_name}"
                )
            elif torch.is_grad_enabled() and any(
                _arg_requires_grad(x) for x in tensor_args
            ):
                why_not_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )
            if not why_not_fast_path:
                merged_mask, mask_type = self.merge_masks(
                    attn_mask, key_padding_mask, query
                )

                if self.in_proj_bias is not None and self.in_proj_weight is not None:
                    return torch._native_multi_head_attention(
                        query,
                        key,
                        value,
                        self.embed_dim,
                        self.num_heads,
                        quantized_in_proj_weight,
                        self.in_proj_bias,
                        quantized_out_proj_weight,
                        self.out_proj.bias,
                        merged_mask,
                        need_weights,
                        average_attn_weights,
                        mask_type,
                    )

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, (
            "MultiheadAttention does not support NestedTensor outside of its fast path. "
            + f"The fast path was not hit because {why_not_fast_path}"
        )

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                quantized_in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                quantized_out_proj_weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                quantized_in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                quantized_out_proj_weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


def replace_qat_modules(model, quantizer):
    for name, mod in model.named_children():
        if isinstance(mod, nn.Conv2d):
            new_mod = QatConv2d.from_float(mod, quantizer)
            setattr(model, name, new_mod)
        elif type(mod) is nn.Linear:
            new_mod = QatLinear.from_float(mod, quantizer)
            setattr(model, name, new_mod)
        elif isinstance(mod, nn.MultiheadAttention):
            new_mod = QatMHA.from_float(mod, quantizer)
            setattr(model, name, new_mod)
        else:
            replace_qat_modules(mod, quantizer)


def quantized_vit(model):
    for name, mod in model.named_children():
        if isinstance(mod, QatConv2d):
            fq = mod.fake_quantizer
            qw = fq(mod.weight) # / fq.scale.reshape((-1, 1, 1, 1))
            mod.qscale = fq.scale.reshape((-1, 1, 1, 1)).detach().clone()
            mod.weight.data = qw.detach().clone()
            mod.fake_quantizer = nn.Identity()
        elif isinstance(mod, QatLinear):
            fq = mod.fake_quantizer
            qw = fq(mod.weight) # / fq.scale.reshape((-1, 1))
            mod.qscale = fq.scale.reshape((-1, 1)).detach().clone()
            mod.weight.data = qw.detach().clone()
            mod.fake_quantizer = nn.Identity()
        elif isinstance(mod, QatMHA):
            fq_in = mod.fake_quantizer_in
            qw_in = fq_in(mod.in_proj_weight) # / fq_in.scale.reshape((-1, 1))
            mod.qscale = fq_in.scale.reshape((-1, 1)).detach().clone()
            mod.in_proj_weight.data = qw_in.detach().clone()

            fq_out = mod.fake_quantizer_out
            qw_out = fq_out(mod.out_proj.weight) # / fq_out.scale.reshape((-1, 1))
            mod.out_proj.qscale = fq_out.scale.reshape((-1, 1)).detach().clone()
            mod.out_proj.weight.data = qw_out.detach().clone()

            mod.fake_quantizer_in = nn.Identity()
            mod.fake_quantizer_out = nn.Identity()
        else:
            quantized_vit(mod)
