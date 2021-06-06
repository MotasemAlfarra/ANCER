import os
import torch
from torchvision import transforms
from torchvision import datasets
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]


# CIFAR10
class CIF10_TRAINSET(Dataset):
    def __init__(self):
        t_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
        self.cifar10 = CIFAR10(root='datasets',
                               download=True,
                               train=True,
                               transform=t_train)

    def __getitem__(self, index: slice):
        data, target = self.cifar10[index]
        return data, target, index

    def __len__(self):
        return len(self.cifar10)


class CIF10_TESTSET(Dataset):
    def __init__(self):
        self.cifar10 = CIFAR10(root='datasets',
                               download=True,
                               train=False,
                               transform=transforms.ToTensor())

    def __getitem__(self, index: slice):
        data, target = self.cifar10[index]
        return data, target, index

    def __len__(self):
        return len(self.cifar10)


def cifar10(batch_sz: int):
    img_sz = [3, 32, 32]
    trainset, testset = CIF10_TRAINSET(), CIF10_TESTSET()
    train_loader = DataLoader(trainset,  batch_size=batch_sz, shuffle=True,
                              pin_memory=True, num_workers=4)
    test_loader = DataLoader(testset, batch_size=batch_sz, shuffle=False,
                             pin_memory=True, num_workers=4, drop_last=False)
    return train_loader, test_loader, img_sz, len(trainset), len(testset)


# ImageNet
def ImageNet(batch_sz: int, directory: str = None):
    if directory is None:
        raise ValueError(
            "to use ImageNet, please provide its correct directory on disk."
        )

    img_sz = [3, 224, 224]
    trainset, testset = ImageNet_Trainset(directory), ImageNet_Testset(directory)
    print('length of trainset and test set is {}, {}'.format(
        len(trainset), len(testset)))
    train_loader = DataLoader(trainset,  batch_size=batch_sz, shuffle=True,
                              pin_memory=True, num_workers=4)
    test_loader = DataLoader(testset, batch_size=batch_sz, shuffle=False,
                             pin_memory=True, num_workers=4)
    return train_loader, test_loader, img_sz, len(trainset), len(testset)


class ImageNet_Trainset(Dataset):
    def __init__(self, path: str):
        subdir = os.path.join(path, "train")
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        self.imgnet = datasets.ImageFolder(subdir, transform)

    def __getitem__(self, index: slice):
        data, target = self.imgnet[index]
        return data, target, index

    def __len__(self):
        return len(self.imgnet)


class ImageNet_Testset(Dataset):
    def __init__(self, path: str):
        subdir = os.path.join(path, "val")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        indices = list(range(0, 50000, 100))
        testset = datasets.ImageFolder(subdir, transform)
        self.imgnet = torch.utils.data.Subset(testset, indices)

    def __getitem__(self, index: slice):
        data, target = self.imgnet[index]
        return data, target, index

    def __len__(self):
        return len(self.imgnet)
