import partition_data
import torch
import numpy as np
from torchvision import transforms
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

EPOCHS = 101
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
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
        )

        self.lin = nn.Linear(192, self.num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.CNN3layer(x)
        x = x.view(-1,192)
        x = self.lin(x)
        x = x.view(x.size(0), self.num_classes)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                if m.bias is not None:
                    m.bias.data.zero_()


def nin_network( Num_per_class, save=False):
    NUMBER_OF_CLASSES = 10
    # ---- loading data -----
    train_set = torchvision.datasets.CIFAR10(
        root='./data/cifar'
        , train=True
        , download=True
        , transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    data = []
    targ = []
    for j in range(10):
        val = 0
        for i in range(len(train_set.targets)):
            if train_set.targets[i] == j:
                data.append(train_set.data[i])
                targ.append(train_set.targets[i])
                val += 1
                if val == Num_per_class:
                    i = len(train_set.targets)
                    break
    print("data size ",len(targ))
    # print(targ)
    train_set.data = data
    train_set.targets = targ

    index_list = list(range(len(train_set)))

    tr_sampler = SubsetRandomSampler(index_list)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, sampler=tr_sampler, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
    # -----loading model --------
    model = NIN(NUMBER_OF_CLASSES)
    # for CPU:
    #nin_trained = torch.load('NiN_model.pt', map_location='cpu')
    # for GPU only
    nin_trained = torch.load('NiN_model_5blocks.pt')
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

    train_model(model, optimizer, loss_function, train_loader, testloader, save)


def train_model(model, optimizer, loss_function, train_loader, testloader, save):
    # ..........running on number of epochs............
    r_loss = list()
    r_epoch = list()
    model.train()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 45, 50], gamma=0.2)
    for epoch in range(1, EPOCHS):
        train_loss, valid_loss = [], []

        # ......... Training the model.................

        for data, target in train_loader:
            data, target = Variable(data.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            #print("epoch = ", epoch, "loss = ", loss.item())
        scheduler.step()

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
   # write_loss(r_loss, r_epoch)


def save_model_params(model):
    # ...........save model...........
    torch.save(model.state_dict(), "./NiN_model.pt")


def write_loss( r_loss, r_epoch):
    # ............writing loss.....
    lossfile = open('loss_linearClass.txt', 'w')
    for i in range(len(r_loss)):
        lossfile.write('{} {:.4f}\n'.format(r_epoch[i],r_loss[i]))


if __name__ == "__main__":
    # [20,100,400,1000,5000]
    print("1000 for 101 for rot5")
    nin_network(1000)