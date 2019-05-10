import torch
import numpy as np
from torchvision import transforms
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
# import torch.optim.lr_scheduler.StepLR as StepLR

_tasks = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])


NUMBER_OF_CLASSES = 10
from torchvision.datasets import CIFAR10
cifar = CIFAR10('data', train=True, download=True, transform=transforms.ToTensor())
split = int(0.8 * len(cifar))
index_list = list(range(len(cifar)))
train_idx, valid_idx = index_list[:split], index_list[split:]

tr_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(valid_idx)

trainloader = DataLoader(cifar, batch_size=128, sampler=tr_sampler)
validloader = DataLoader(cifar, batch_size=25, sampler=val_sampler)

class NIN(nn.Module):
    def __init__(self, num_classes):
        super(NIN, self).__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, 192, 5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 160, 1),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 96, 1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            nn.Dropout(inplace=True),

            nn.Conv2d(96, 192, 5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(3, stride=2, ceil_mode=True),
            nn.Dropout(inplace=True),

            nn.Conv2d(192, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, self.num_classes, 1),
            nn.BatchNorm2d(self.num_classes),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(8, stride=1)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), self.num_classes)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                if m.bias is not None:
                    m.bias.data.zero_()
model = NIN(NUMBER_OF_CLASSES)


#optimizer and loss function
# cross entropy for classes
# loss_function =
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay= 5e-4, momentum = 0.9, nesterov = True)


# running on number of epochs

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
for epoch in range(1, 101):
    print("epoch = ", epoch)
    train_loss, valid_loss = [], []

    # training the model
    model.train()
    cont = 0
    for data, target in trainloader:
        # data = 255 -data #/255
        optimizer.zero_grad()
        output = model(data)

        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        cont+=1
        if cont>= 100:
            break

        ## evaluation part
    model.eval()
    for data, target in validloader:
        # data = 255 - data #/ 255
        output = model(data)
        loss = loss_function(output, target)
        valid_loss.append(loss.item())
        _, preds_tensor = torch.max(output, 1)
        preds = np.squeeze(preds_tensor.numpy())
        print("loss = ",loss.item())
        break
    scheduler.step()


## dataloader for validation dataset
dataiter = iter(validloader)
data, labels = dataiter.next()
# data = 255-data #/255
output = model(data)

_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy())
#
# print ("Actual:", labels[:10])
# print ("Predicted:", preds[1:25])
# print ("size = ",len(preds))

# def accuracy(pre):
#     acc = 0
#     for i in range(pre):
#         if pre[i] == labels [i]:
#             acc = acc + 1
#     a = acc / len(pre)
#     return a
