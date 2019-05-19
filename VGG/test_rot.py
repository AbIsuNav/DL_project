# %%
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.sampler import SubsetRandomSampler
from rotate_data import test_data
import numpy as np
import matplotlib.pyplot as plt


test_loader = test_data()


# %%
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

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,padding = 1)
        torch.nn.init.xavier_uniform_(self.conv5.weight)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,padding = 1)
        torch.nn.init.xavier_uniform_(self.conv6.weight)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,padding = 1)
        torch.nn.init.xavier_uniform_(self.conv7.weight)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,padding = 1)
        torch.nn.init.xavier_uniform_(self.conv8.weight)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,padding = 1)
        torch.nn.init.xavier_uniform_(self.conv9.weight)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,padding = 1)
        torch.nn.init.xavier_uniform_(self.conv10.weight)

        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,padding = 1)
        torch.nn.init.xavier_uniform_(self.conv11.weight)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,padding = 1)
        torch.nn.init.xavier_uniform_(self.conv12.weight)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,padding = 1)
        torch.nn.init.xavier_uniform_(self.conv13.weight)


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

        #print("third")
        #print(t)
        t = self.conv5(t).cuda()
        y = nn.BatchNorm2d(256).cuda()
        t = y(t).cuda()
        t = nn.functional.relu(t).cuda()
        t = self.conv6(t).cuda()
        y = nn.BatchNorm2d(256).cuda()
        t = y(t).cuda()
        t = nn.functional.relu(t).cuda()
        t = self.conv7(t).cuda()
        y = nn.BatchNorm2d(256).cuda()
        t = y(t).cuda()
        t = nn.functional.relu(t).cuda()
        t = nn.functional.max_pool2d(t, kernel_size = 2, stride = 2).cuda()

        #print("fourth")
        #print(t)
        t = self.conv8(t).cuda()
        y = nn.BatchNorm2d(512).cuda()
        t = y(t).cuda()
        t = nn.functional.relu(t).cuda()
        t = self.conv9(t).cuda()
        y = nn.BatchNorm2d(512).cuda()
        t = y(t).cuda()
        t = nn.functional.relu(t).cuda()
        t = self.conv10(t).cuda()
        y = nn.BatchNorm2d(512).cuda()
        t = y(t).cuda()
        t = nn.functional.relu(t).cuda()
        t = nn.functional.max_pool2d(t, kernel_size = 2, stride = 2).cuda()

        #print("fifth")
        #print(t)
        t = self.conv11(t).cuda()
        y = nn.BatchNorm2d(512).cuda()
        t = y(t).cuda()
        t = nn.functional.relu(t).cuda()
        t = self.conv12(t).cuda()
        y = nn.BatchNorm2d(512).cuda()
        t = y(t).cuda()
        t = nn.functional.relu(t).cuda()
        t = self.conv13(t).cuda()
        y = nn.BatchNorm2d(512).cuda()
        t = y(t).cuda()
        t = nn.functional.relu(t).cuda()
        t = nn.functional.max_pool2d(t, kernel_size = 2, stride = 2).cuda()
        # t = nn.functional.softmax(t, dim=1)

        return t


# %%
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))
device = "cuda:0"
network = Network()
network.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, milestones=[15, 30, 40], gamma=0.2)

# %%
network.load_state_dict(torch.load('./vggtrainedNew'), strict=False)

network.eval()
# %%
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
# %%

# %%
