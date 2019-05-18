import partition_data
import torch
import numpy as np
from torchvision import transforms
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
from torch import optim
from torch.autograd import Variable


#nin_trained = torch.load('NiN_model.pt',map_location='cpu')
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
            nn.Dropout(inplace=True)
        )
        self.CNN3layer = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, self.num_classes, kernel_size=1, stride=1, padding=0),
            #nn.BatchNorm2d(self.num_classes),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
        )

        self.lin = nn.Linear(10, self.num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.CNN3layer(x)
        x = x.view(-1,self.num_classes)
        x = self.lin(x)
        x = x.view(x.size(0), self.num_classes)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                if m.bias is not None:
                    m.bias.data.zero_()


def nin_network(elements, save=False):
    NUMBER_OF_CLASSES = 10

    # ---- loading data -----
    train_loader = partition_data.partition_Cifar10(elements)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transforms.ToTensor())
    #testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)
    # -----loading model --------
    model = NIN(NUMBER_OF_CLASSES)
    # for CPU:
    #nin_trained = torch.load('NiN_model.pt', map_location='cpu')
    # for GPU only
    nin_trained = torch.load('NiN_model.pt')
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k[9:]: v for k, v in nin_trained.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.features.load_state_dict(pretrained_dict)

    loss_function = nn.CrossEntropyLoss()

    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9, nesterov = True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 45, 50], gamma=0.2)
    train_model(model, scheduler, optimizer, loss_function, train_loader, testloader, save)


def train_model(model, scheduler, optimizer, loss_function, train_loader, testloader, save):
    # ..........running on number of epochs............
    r_loss = list()
    r_epoch = list()
    for epoch in range(1, 2):
        train_loss, valid_loss = [], []

        # ......... Training the model.................
        model.train()
        cont = 0
        loss = 0.0
        for data, target in train_loader:
            data, target = Variable(data.cuda()), Variable(target.cuda())
            # print("data", data.shape)
            # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
            #data = data.view(-1, 32 * 32 * 3)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            print("epoch = ", epoch, "loss = ", loss.item())
        # .........Printing values every 3 epochs.........
        if epoch % 5 == 0:
            r_epoch.append(epoch)
            r_loss.append(loss.item())
            for param_group in optimizer.param_groups:
                print("lr  ",param_group['lr'])

        scheduler.step()

    output = model(data)

    _, preds_tensor = torch.max(output, 1)

    model.eval()
    test_loss = 0
    correct = 0
    for data, target in testloader:
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output = model(data)
        test_loss += loss_function(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss * 128., correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))

    #preds = np.squeeze(torch.Tensor.cpu(preds_tensor).numpy)
    # -------- save model ---------------------
    if save:
        save_model_params(model)
    write_loss(r_loss, r_epoch)


def save_model_params(model):
    # ...........save model...........
    torch.save(model.state_dict(), "./NiN_model.pt")


def write_loss( r_loss, r_epoch):
    # ............writing loss.....
    lossfile = open('loss_linearClass.txt', 'w')
    for i in range(len(r_loss)):
        lossfile.write('{} {:.4f}\n'.format(r_epoch[i],r_loss[i]))


if __name__ == "__main__":
    data_elements = [20,100,400,1000,5000]
    nin_network(20)