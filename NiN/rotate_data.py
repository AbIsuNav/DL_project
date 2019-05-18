from torchvision import transforms
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import numpy as np


def train_data(tbatch=128, vbatch=100, wcuda=True):
    train_set = torchvision.datasets.CIFAR10(
        root='./data/cifar'
        , train=True
        , download=True
        , transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    len(train_set)
    split = int(0.8 * 50000)
    index_list = list(range(50000))
    train_idx, valid_idx = index_list[:split], index_list[split:]

    tr_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(valid_idx)

    data = []
    targets = []

    for i in range(len(train_set)):
        a = train_set.data[i]
        b = np.rot90(a)
        c = np.rot90(b)
        d = np.rot90(c)
        data.append(a)
        data.append(b)
        data.append(c)
        data.append(d)
        targets.append(0)
        targets.append(1)
        targets.append(2)
        targets.append(3)

    train_set.data = data
    train_set.targets = targets
    if wcuda:
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=tbatch, sampler=tr_sampler, num_workers=2
        )

        val_loader = torch.utils.data.DataLoader(
            train_set, batch_size=vbatch, sampler=val_sampler, num_workers=2
        )
        return train_loader, val_loader
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=tbatch, sampler=tr_sampler)

    val_loader = torch.utils.data.DataLoader(
        train_set, batch_size=vbatch, sampler=val_sampler)
    return train_loader, val_loader


def test_data(batch=100, wcuda=True):
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transforms.ToTensor())

    test_idx = list(range(len(testset)))

    data = []
    targets = []

    for i in range(len(test_idx)):
        a = testset.data[i]
        b = np.rot90(a)
        c = np.rot90(b)
        d = np.rot90(c)
        data.append(a)
        data.append(b)
        data.append(c)
        data.append(d)
        targets.append(0)
        targets.append(1)
        targets.append(2)
        targets.append(3)

    testset.data = data
    testset.targets = targets
    if wcuda:
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch, shuffle=False, num_workers=2
        )
        return test_loader
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch, shuffle=False)
    return test_loader

