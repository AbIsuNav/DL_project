import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from rotate_data import train_data

# .... loading data ......


NUMBER_OF_CLASSES = 4
train_loader, val_loader = train_data()
r_loss = list()
r_epoch = list()
v_loss = list()
# ......Parameters.....
class NIN(nn.Module):
    def __init__(self, num_classes):
        super(NIN, self).__init__()
        self.num_classes = num_classes

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
            nn.Dropout(inplace=True),

            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(inplace=True),

            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, num_classes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.num_classes),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
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

loss_function = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-4, momentum = 0.9, nesterov = True)
model.cuda()

# ..........running on number of epochs............

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
for epoch in range(1, 101):
    train_loss, valid_loss = [], []

    # ......... Training the model.................
    model.train()
    cont = 0
    loss = 0.0
    for data, target in train_loader:
        # data = 255 -data #/255
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data)

        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    # .........Printing values every 3 epochs.........
    if epoch % 3 == 0:
        r_epoch.append(epoch)
        r_loss.append(loss.item())

    # ..........Validation .............
    model.eval()
    for data, target in val_loader:
        # data = 255 - data #/ 255
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output = model(data)
        loss = loss_function(output, target)
        valid_loss.append(loss.item())
        _, preds_tensor = torch.max(output, 1)

        preds = np.squeeze(torch.Tensor.cpu(preds_tensor).numpy)
    if epoch%3== 0:
        v_loss.append(loss.item())
    scheduler.step()


dataiter = iter(val_loader)
data, labels = dataiter.next()

output = model(Variable(data.cuda()))

_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(torch.Tensor.cpu(preds_tensor).numpy)

# ...........save model...........
# torch.save(model.state_dict(), "./NiN_model.pt")
# ............writing loss.....
lossfile = open('loss.txt', 'w')
for i in range(len(r_loss)):
    lossfile.write('{} {:.4f} {:.4f}\n'.format(r_epoch[i],r_loss[i],v_loss[i]))