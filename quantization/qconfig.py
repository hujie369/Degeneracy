"""
该文件用于定义几个进行量化配置和初始化的函数
"""
import torch
from torch.ao.quantization import QConfig
from quantization.quantization_modules import NewConvBnReLU2d, NewConvBn2d


def configureModel(qat_model, w_quantize, w_bit, a_quantize, a_bit):
    qat_model.qconfig = QConfig(
                    activation=a_quantize.with_args(
                        dtype=torch.quint8,
                        qscheme=torch.per_tensor_affine,
                        reduce_range=False,
                        quant_min=0,
                        quant_max=2**a_bit - 1,
                    ),
                    weight=w_quantize.with_args(
                        dtype=torch.qint8,
                        qscheme=torch.per_channel_symmetric,
                        reduce_range=False,
                        quant_min=-2**(w_bit - 1),
                        quant_max=2**(w_bit - 1) - 1,
                    )
                )
    return qat_model


def initializeModel(qat_model, paras):
    # use the PTQ parameters to initialize QAT
    weight_list = torch.load(paras.weight_list_path, weights_only=True)
    activation_list = torch.load(paras.act_list_path, weights_only=True)
    # PTQ's parameters is w8a8, need to scale to the target bit
    # factor is the scaling value
    factor = 255 / (2**paras.w_bit - 1)

    for temp in weight_list:
        name, sc, zp = temp
        sub_module = qat_model
        for part in name.split('.'):
            sub_module = getattr(sub_module, part)
        # PTQ's parameters was fused on conv and bn, need to be restored on QAT
        if type(sub_module) in (NewConvBnReLU2d, NewConvBn2d):
            bn_var_rsqrt = torch.sqrt(sub_module.bn.running_var + sub_module.bn.eps)
            sc = sc / sub_module.bn.weight * bn_var_rsqrt

        sub_module.weight_fake_quant.scale.data = sc * factor
        sub_module.weight_fake_quant.zero_point.data = torch.round(zp / factor).to(torch.int)

    factor = 255 / (2**paras.a_bit - 1)
    for temp in activation_list:
        name, sc, zp = temp
        sub_module = qat_model
        for part in name.split('.'):
            sub_module = getattr(sub_module, part)

        sub_module.activation_post_process.scale.data = sc * factor
        sub_module.activation_post_process.zero_point.data = torch.round(zp / factor).to(torch.int)
    return qat_model
