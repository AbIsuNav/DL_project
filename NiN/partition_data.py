from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


def partition_Cifar10(elemets_per_class=20,t_batch_size=10):
    """
    Returns a Data loader (train and validation) of Cifar10 with a specific Elements_perClass and Batch_size
    """
    cifar = CIFAR10('data', train=True, download=True, transform=transforms.ToTensor())
    split = int(0.8 * len(cifar))
    index_list = list(range(len(cifar)))
    train_idx, valid_idx = index_list[:split], index_list[split:]
    data, target = [], []
    data_check, check_class = [False]*10, [0]*10
    # -------get training data---------------
    for i in range(len(train_idx)):
        t = cifar.targets[i]
        if data_check[t] < elemets_per_class:
            data.append(cifar.data[i])
            target.append(t)
            data_check[t] += 1
        else:
            check_class[t] = True
        flag = True
        for j in check_class:
            if not j:
                flag = j
                break
        if flag:
            break
    cifar.data = data
    cifar.targets = target
#    train_idx = list(range(len(cifar)))
#    tr_sampler = SubsetRandomSampler(train_idx)
    train_loader = DataLoader(cifar, shuffle=True, num_workers=2)
    #train_loader = DataLoader(cifar, shuffle=True)
    return train_loader


if __name__ == "__main__":
    print("...partitioning........")
    trainloader = partition_Cifar10(20)
    print("...data partitioned")
