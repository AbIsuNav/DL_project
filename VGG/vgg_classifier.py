# %%
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

TOTAL_EPOCHS = 50
# %%
train_set = torchvision.datasets.CIFAR10(
    root='./data/cifar'
    , train=True
    , download=True
    , transform=transforms.Compose([
        transforms.ToTensor()
    ])
)
# %%
test_set = torchvision.datasets.CIFAR10(
    root='./data/cifar'
    , train=False
    , download=True
    , transform=transforms.Compose([
        transforms.ToTensor()
    ])
)
# %%
data = []
targ = []
for j in range(10):
    val = 0
    for i in range(len(train_set.targets)):
        if train_set.targets[i] == j:
            data.append(train_set.data[i])
            targ.append(train_set.targets[i])
            val += 1
            if val == 5000:
                i = len(train_set.targets)
                break
print(len(targ))
#print(targ)
train_set.data = data
train_set.targets = targ
# %%
split = int(0.8 * len(train_set))

index_list = list(range(len(train_set)))
train_idx, valid_idx = index_list[:split], index_list[split:]

tr_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=10, sampler=tr_sampler, num_workers=2
)
val_loader = torch.utils.data.DataLoader(
    train_set, batch_size=10, sampler=val_sampler, num_workers=2
)

test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=10, num_workers=2
)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        torch.nn.init.xavier_uniform_(self.conv2.weight)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        torch.nn.init.xavier_uniform_(self.conv4.weight)

        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Sequential(
            nn.Linear(8192, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 10)
        )

    def forward(self, t):
        t = self.conv1(t).cuda()
        y = nn.BatchNorm2d(64).cuda()
        t = y(t).cuda()
        t = nn.functional.relu(t).cuda()
        # t = nn.functional.max_pool2d(t, kernel_size = 2, stride = 1)
        # print("First")
        # print(t)
        t = self.conv2(t).cuda()
        y = nn.BatchNorm2d(64).cuda()
        t = y(t).cuda()
        t = nn.functional.relu(t).cuda()
        t = nn.functional.max_pool2d(t, kernel_size=2, stride=2).cuda()
        # print("Second")
        # print(t)

        t = self.conv3(t).cuda()
        y = nn.BatchNorm2d(128).cuda()
        t = y(t).cuda()
        t = nn.functional.relu(t).cuda()
        t = self.conv4(t).cuda()
        y = nn.BatchNorm2d(128).cuda()
        t = y(t).cuda()
        t = nn.functional.relu(t).cuda()
        t = nn.functional.max_pool2d(t, kernel_size=2, stride=2).cuda()

        t = t.view(-1, 8192).cuda()

        # print("before fc")
        # print(t)
        t = self.classifier(t)

        return F.log_softmax(t)


# %%
#print(torch.cuda.current_device())
#print(torch.cuda.device(0))
#print(torch.cuda.get_device_name(0))
device = "cuda:0"
network = Network()
network.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, milestones=[15, 30, 40], gamma=0.2)

#network.load_state_dict(torch.load('./vggtrainedNew', map_location='cpu'), strict=False)
network.load_state_dict(torch.load('./vggtrainedNew'), strict=False)
network.train()

loss_training = []
loss_val = []
for epoch in range(TOTAL_EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to("cuda:0"), labels.to("cuda:0")

        optimizer.zero_grad()

        outputs = network(inputs)
        # print(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        """if (i % 100 == 0 and i != 0):
            for param_group in optimizer.param_groups:
                print(param_group['lr'])
            print("no minibatches " + str(i))
            a = str((running_loss) / i)
            print("training loss = " + a)"""
    scheduler.step()
    loss_training.append((running_loss) / i)
    a = str((running_loss) / i)
    print("training loss = " + a)
    valloss = 0.0
    for i, data in enumerate(val_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = network(inputs)
        valloss += criterion(outputs, labels).item()
    loss_val.append(valloss / i)
    b = str(valloss / i)
    print("Validation loss = " + b)

    print("Epochs completed : " + str(epoch + 1))
network.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to("cuda:0"), labels.to("cuda:0")
        outputs = network(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

