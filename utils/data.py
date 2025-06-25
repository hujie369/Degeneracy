import torch
from torch.utils import data
import torchvision
import torchvision.transforms as transforms


def prepare_data_loaders(train_batch_size, eval_batch_size, data_path='~/datasets/imagenet/', num_workers=16):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = torchvision.datasets.ImageNet(
        data_path, split="train", transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    dataset_test = torchvision.datasets.ImageNet(
        data_path, split="val", transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler, num_workers=num_workers, pin_memory=True)

    data_loader_test = data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler, num_workers=num_workers)

    return data_loader, data_loader_test