import torch
import os

from utils.data import prepare_data_loaders
from utils.utils import print_size_of_model
from networks import load_model
from utils.train import evaluate
from config import Config
from torch.ao.quantization import (
    QConfig,
    default_per_channel_weight_observer,
    HistogramObserver,
)
import torch.backends.quantized
from torch.ao.nn.intrinsic.quantized import ConvReLU2d, ConvAddReLU2d
from torch.ao.nn.quantized import Linear, Conv2d, Quantize, QFunctional


torch.backends.quantized.engine= 'onednn'

# get parameters from config
paras = Config("resnet18", w_bit=8, a_bit=8, tbs=128)
# load ImageNet dataset
data_train, data_eval = prepare_data_loaders(paras.train_batch_size, paras.eval_batch_size)
# load model to be quantized
float_model = load_model(paras.model)
float_model.eval()
# fuse conv, bn and\or relu
float_model.fuse_model()

print("Size of baseline model")
print_size_of_model(float_model)

# evaluate the accuracy of pretrained model
top1, top5 = evaluate(float_model, data_eval, device="cuda")
print(f"Float model's evaluation accuracy:\ntop1: {top1.avg:2.3f}, top5: {top5.avg:2.3f}")


# torch.jit.save(torch.jit.script(float_model), paras.scripted_float_model_path)

# build quantization config
float_model.to("cpu")
per_channel_quantized_model = float_model
per_channel_quantized_model.qconfig = QConfig(
    activation=HistogramObserver.with_args(
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        ),
    weight=default_per_channel_weight_observer
    )
# PTQ
torch.ao.quantization.prepare(per_channel_quantized_model, inplace=True)
# calibrate
evaluate(per_channel_quantized_model, data_train, paras.num_calibration_batches)
torch.ao.quantization.convert(per_channel_quantized_model, inplace=True)
# evaluate the accuracy of 8-bit quantized model
top1, top5 = evaluate(per_channel_quantized_model, data_eval)
print(f"PTQ model's evaluation accuracy:\ntop1: {top1.avg:2.3f}, top5: {top5.avg:2.3f}")

# torch.jit.save(
#     torch.jit.script(per_channel_quantized_model),
#     paras.scripted_quantized_model_path
# )

# save the PTQ parameters (scale and zero_point) of each layers
model = per_channel_quantized_model

weight_list = []
for name, module in model.named_modules():
    if isinstance(module, (ConvReLU2d, Linear, Conv2d)):
        temp = [name]
        temp.append(module.weight().q_per_channel_scales().to(torch.float))
        temp.append(module.weight().q_per_channel_zero_points().to(torch.int))
        weight_list.append(temp)

activation_list = []
for name, module in model.named_modules():
    if isinstance(module, (ConvReLU2d, Linear, Conv2d, Quantize, QFunctional)):
        temp = [name]
        temp.append(torch.tensor(module.scale, dtype=torch.float))
        temp.append(torch.tensor(module.zero_point, dtype=torch.int))
        activation_list.append(temp)

torch.save(weight_list, paras.weight_list_path)
torch.save(activation_list, paras.act_list_path)
