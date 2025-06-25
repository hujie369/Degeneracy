from functools import partial

import torch
from torch import nn
import torchvision as tv

from quantization.fake_quantizer import VitFakeQuantizer
from quantization.quantization_modules import QatConv2d, QatLinear, QatMHA  # replace_qat_modules
from utils.train import evaluate, train_one_epoch
from utils.data import prepare_data_loaders


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


w_bit = 4
vit = tv.models.vit_b_16(weights=tv.models.ViT_B_16_Weights.IMAGENET1K_V1)
# vit = tv.models.vit_l_16(weights=tv.models.ViT_L_16_Weights.IMAGENET1K_V1)

quantizer = partial(VitFakeQuantizer, quant_min=-2**(w_bit - 1), quant_max=2**(w_bit - 1) - 1, is_per_channel=True, requires_grad=True)
replace_qat_modules(vit, quantizer)
qat_vit = vit

data_train, data_eval = prepare_data_loaders(256, 100)
top1, top5 = evaluate(qat_vit, data_eval, device='cuda')
print(f'float qat model acc, top1: {top1.avg.item():.3f}, top5: {top5.avg.item():.3f}')

optimizer = torch.optim.SGD(qat_vit.parameters(), lr=1e-4, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()
for nepoch in range(8):
    if nepoch == 4:
        optimizer.param_groups[0]['lr'] = 1e-5
    if nepoch == 7:
        optimizer.param_groups[0]['lr'] = 1e-6

    train_one_epoch(qat_vit, criterion, optimizer, data_train, 'cuda')

    top1, top5 = evaluate(qat_vit, data_eval, device='cuda')
    print(f'float qat model acc, top1: {top1.avg.item():.3f}, top5: {top5.avg.item():.3f}')
    # save state_dict
    torch.save(qat_vit.state_dict(), f'./quantized_models/vit_b_16/{w_bit}-bit_{nepoch + 1}.pth')
