import argparse

import torch
from torch.ao.quantization import QConfig

from networks import load_model
from utils.train import train_one_epoch, evaluate, eval_qat_model
from utils.data import prepare_data_loaders
from quantization.fake_quantizer import ResFakeQuantize
from quantization.quantization_modules import getMappings, my_freeze_bn_stats
from quantization.qconfig import configureModel, initializeModel
from config import Config

import torch.backends.quantized
torch.backends.quantized.engine= 'onednn'   # support for full range


def main(model_name="resnet18", w_bit=4, a_bit=8, tbs=256, lr=1e-3):
    paras = Config(model_name, w_bit, a_bit, tbs)

    data_train, data_eval = prepare_data_loaders(paras.train_batch_size, paras.eval_batch_size)
    qat_mappings, quantized_mappings = getMappings(paras.defaultMap)

    # fuse modules
    qat_model = load_model(paras.model)
    qat_model.fuse_model(is_qat=True)

    qat_model = configureModel(qat_model, ResFakeQuantize, paras.w_bit, ResFakeQuantize, paras.a_bit)
    # insert fake quantizer
    torch.ao.quantization.prepare_qat(qat_model, mapping=qat_mappings, inplace=True)

    # initial the quantization parameter (scale and zero_point)
    qat_model = initializeModel(qat_model, paras)

    # evaluate the accuracy of initialized model
    eval_qat_model(qat_model, data_eval, quantized_mappings)

    best_top1 = 0

    # QAT
    optimizer = torch.optim.SGD(qat_model.parameters(), lr=lr, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    for nepoch in range(8):
        if nepoch == 4:
            optimizer.param_groups[0]['lr'] *= 0.1
        if nepoch == 5:
            qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        if nepoch == 6:
            # Freeze quantizer parameters
            qat_model.apply(torch.ao.quantization.disable_observer)
        if nepoch == 7:
            optimizer.param_groups[0]["lr"] *= 0.1

        train_one_epoch(qat_model, criterion, optimizer, data_train, 'cuda', paras.qat_train_bs)

        top1, top5 = evaluate(qat_model, data_eval, device='cuda')
        print(f'float qat model acc, top1: {top1.avg.item():.3f}, top5: {top5.avg.item():.3f}')
        torch.save(qat_model.state_dict(), paras.qat_float_path + f"epoch{nepoch+1}.pth")

        # evaluate the accuracy on validation dataset
        qat_model.to('cpu')
        quantized_model = torch.ao.quantization.convert(qat_model.eval(), mapping=quantized_mappings, inplace=False)
        quantized_model.eval()
        top1, top5 = evaluate(quantized_model, data_eval)
        print(f'quantized model acc, top1: {top1.avg.item():.3f}, top5: {top5.avg.item():.3f}')

        # save the best state_dict
        if top1.avg.item() > best_top1:
            best_top1 = top1.avg.item()
            torch.save(qat_model.state_dict(), paras.model_path + f"best_w{paras.w_bit}a{paras.a_bit}.pth")

    print(f"the best top1 is: {best_top1:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="model name")
    parser.add_argument("--w-bit", '-w', type=int, default=4, help="weight bit width")
    parser.add_argument("--a-bit", '-a', type=int, default=8, help="activation bit width")
    parser.add_argument("--tbs", type=int, default=256, help="training batch size")
    parser.add_argument("--lr", "-l", type=float, default=1e-3, help="initial learning rate")
    args = parser.parse_args()

    main(model_name=args.model_name, w_bit=args.w_bit, a_bit=args.a_bit, tbs=args.tbs, lr=args.lr)
