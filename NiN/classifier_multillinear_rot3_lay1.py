# Building fully connected layers

import torch
import numpy as np
from torchvision import transforms
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# ---- import model ----------

# NUMBER_OF_CLASSES = 10
from torchvision.datasets import CIFAR10
cifar = CIFAR10('data', train=True, download=True, transform=transforms.ToTensor())
cifar_test = CIFAR10('data', train=False, download=True, transform=transforms.ToTensor())
split = int(0.8 * len(cifar))
index_list = list(range(len(cifar)))
train_idx, valid_idx = index_list[:split], index_list[split:]

tr_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(valid_idx)
# BATCH_SIZE = 4
train_loader = DataLoader(cifar, batch_size=128, sampler=tr_sampler, num_workers=2)
validloader = DataLoader(cifar, batch_size=1, sampler=val_sampler, num_workers=2)
learning_rate = 0.1
epochs = 100
num_classes = 10

class FC_classifier(nn.Module):
    def __init__(self):
        super(FC_classifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 96, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(inplace=True)
        )
        self.fc1 = nn.Linear(96*16*16, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,96*16*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x) #soft_max dimension

net = FC_classifier()
# print(net)


# -----loading model --------

# for CPU:
#nin_trained = torch.load('NiN_model.pt', map_location='cpu')
# for GPU only
nin_trained = torch.load('NiN_model_5blocks.pt')
model_dict = net.state_dict()
# 1. filter out unnecessary keys
pretrained_dict = {k[9:]: v for k, v in nin_trained.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict)
# 3. load the new state dict
net.features.load_state_dict(pretrained_dict)


# create a stochastic gradient descent optimizer
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
# create a loss function
criterion = nn.NLLLoss()

net.cuda()

net.train()
# run the main training loop
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,45,50], gamma=0.2)
for epoch in range(epochs):
    train_loss, valid_loss = [], []
    losst, lossv = 0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        net_out = net(data)
        losst = criterion(net_out, target)
        losst.backward()
        optimizer.step()
    scheduler.step()
#    net.eval()
    for data, target in validloader:
        test_loss = 0
        correct = 0
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output = net(data)
        lossv = criterion(output, target)
        valid_loss.append(lossv.item())
        _, preds_tensor = torch.max(output, 1)
        preds = np.squeeze(torch.Tensor.cpu(preds_tensor).numpy)

        test_loss /= len(validloader.dataset)
    print("epoch = ",epoch , "loss = ",losst.item(), "validation loss = ", lossv.item())

# -- ------ testing ------------------
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

net.eval()
test_loss = 0
correct = 0
for data, target in testloader:
    data, target = Variable(data.cuda()), Variable(target.cuda())
    output = net(data)
    test_loss += criterion(output, target).item()
    pred = output.data.max(1, keepdim=True)[1]
    correct += pred.eq(target.data.view_as(pred)).cpu().sum()

test_loss /= len(testloader.dataset)
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    test_loss * 128., correct, len(testloader.dataset),
    100. * correct / len(testloader.dataset)))

