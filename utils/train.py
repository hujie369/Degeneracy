import torch
from tqdm import tqdm
import torch.amp as amp

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, data_loader, neval_batches=100000, device='cpu'):
    model.to(device)
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in tqdm(data_loader):
            image = image.to(device)
            
            # bgr -> rgb
            # image = image.flip(1)
            
            target = target.to(device)
            output = model(image)
            # loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # print('.', end = '')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            cnt += 1
            if cnt >= neval_batches:
                 return top1, top5

    return top1, top5


scaler = amp.GradScaler('cuda')

def train_one_epoch(model, criterion, optimizer, data_loader, device='cuda', ntrain_batches=1000000):
    model.to(device)
    model.train()
    cnt = 0
    for image, target in tqdm(data_loader):
        image, target = image.to(device), target.to(device)
        # mixed precision training
        with amp.autocast('cuda'):
            output = model(image)
            loss = criterion(output, target)
        # 缩放梯度并进行反向传播
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        cnt += 1
        if cnt >= ntrain_batches:
            return


def eval_qat_model(qat_model, data_eval, quantized_mappings):
    top1, top5 = evaluate(qat_model, data_eval, device='cuda')
    print(f'PTQ initial test before quantization, {top1.avg.item():.3f}, top5: {top5.avg.item():.3f}')

    qat_model.to("cpu")
    quantized_model = torch.ao.quantization.convert(qat_model.eval(), mapping=quantized_mappings, inplace=False)
    quantized_model.eval()
    top1, top5 = evaluate(quantized_model, data_eval)
    print(f'PTQ initial test after quantization, {top1.avg.item():.3f}, top5: {top5.avg.item():.3f}')