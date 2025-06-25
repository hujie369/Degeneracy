import os
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

import torch
import torchvision as tv

from networks import load_model
from config import Config
from quantization.qconfig import configureModel, initializeModel
from quantization.fake_quantizer import ResFakeQuantize
from quantization.quantization_modules import (
    getMappings,
    replace_qat_modules,
    quantized_vit,
)
from quantization.fake_quantizer import VitFakeQuantizer


config = {
    "font.family":'Times New Roman',  # 设置字体类型
    "axes.unicode_minus": False #解决负号无法显示的问题
}
rcParams.update(config)


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p") / 1e6)
    os.remove("temp.p")


def plot_sparsity(axs, qweights, w_bit, layer_names):
    bins = 2**(w_bit)
    hb = bins // 2
    for i in range(4):
        # 绘制直方图
        n, bins, patches = axs[i].hist(qweights[i], bins=bins, range=(-hb - 0.5, hb - 0.5), density=True)
        # 0对应的bin设置为红色
        patches[hb].set_facecolor('#FF7F50')
        # 标上各层的名字
        if w_bit == 3:
            axs[i].set_title(layer_names[i], fontsize=15)
        # 设置x轴
        axs[i].set_xlim(-hb - 0.5, hb + 0.5)
        step = 1 if w_bit == 3 else 2
        axs[i].set_xticks(np.arange(-hb, hb + 1, step))

        # 标出0bin的频率值
        frequency = n[hb]
        x_center = bins[hb + 2] - 0.4 if w_bit == 3 else bins[hb + 2] + 0.1
        y_height = n[hb] * 0.975  # 适当抬高y坐标
        axs[i].text(x_center, y_height, f"{frequency:.3f}", ha='center')
        axs[i].plot([-hb - 0.5, bins[hb]], [n[hb], n[hb]], 'r--', linewidth=1)  # 刻画横线

    axs[0].set_ylabel(f'{w_bit}-bit frequency', fontsize=15)


def get_quantized_models(model_name, w_bit, pre_path="quantized_models"):
    if "vit" not in model_name:
        paras = Config(model_name, w_bit)
        qat_mappings, quantized_mappings = getMappings(paras.defaultMap)

        qat_model = load_model(paras.model)
        qat_model.fuse_model(is_qat=True)
        qat_model = configureModel(
            qat_model, ResFakeQuantize, paras.w_bit, ResFakeQuantize, paras.a_bit
        )
        torch.ao.quantization.prepare_qat(qat_model, mapping=qat_mappings, inplace=True)
        qat_model = initializeModel(qat_model, paras)
        adict = {2: "w2a8", 3: "w3a8", 4: "w4a8"}
    else:
        qat_model = tv.models.vit_b_16()
        quantizer = partial(
            VitFakeQuantizer,
            quant_min=-(2 ** (w_bit - 1)),
            quant_max=2 ** (w_bit - 1) - 1,
            is_per_channel=True,
            requires_grad=True,
        )
        replace_qat_modules(qat_model, quantizer)
        adict = {2: "2-bit", 3: "3-bit", 4: "4-bit"}

    file_path = os.path.join(pre_path, os.path.join(model_name, "well_quantized"))
    model_path = [
        os.path.join(file_path, file)
        for file in os.listdir(file_path)
        if file.endswith(".pth")
    ]

    state = None
    for m_path in model_path:
        if adict[w_bit] in m_path:
            state = torch.load(m_path, weights_only=True)
            break
    if state is None:
        return None
    qat_model.load_state_dict(state)
    qat_model.to("cpu")

    if "vit" not in model_name:
        quantized_model = torch.ao.quantization.convert(
            qat_model.eval(), mapping=quantized_mappings, inplace=False
        )
    else:
        quantized_vit(qat_model)
        quantized_model = qat_model
    return quantized_model
