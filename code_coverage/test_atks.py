import sys
import os
# Importing the parent directory
# This line must be preceded by
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # nopep8

from robustbench.utils import load_model  # nopep8
from robustbench.utils import clean_accuracy  # nopep8
from robustbench.data import load_cifar10  # nopep8
import torchattacks  # nopep8
import torch  # nopep8
import pytest  # nopep8
import time  # nopep8
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

CACHE = {}


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model():
    batch_size = 4
    net = Net()

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    # testset = torchvision.datasets.CIFAR10(
    #     root='./data', train=False, download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(
    #     testset, batch_size=batch_size, shuffle=False, num_workers=2)
    # classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #            'dog', 'frog', 'horse', 'ship', 'truck')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(3):  # loop over the dataset multiple times
        running_loss = 0.0
        for data in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()


def get_model(model_name='Standard', device='cpu', model_dir='./models'):
    model = load_model(model_name, model_dir=model_dir, norm='Linf')
    # fsize = os.path.getsize(filePath)
    return model.to(device)


def get_data(data_name='CIFAR10', device='cpu', n_examples=5, data_dir='./data'):
    images, labels = load_cifar10(n_examples=n_examples, data_dir=data_dir)
    return images.to(device), labels.to(device)


@torch.no_grad()
def test(atk_class, device='cpu', n_examples=5, model_dir='./models', data_dir='./data'):
    global CACHE
    model = get_model(device=device, model_dir=model_dir)

    if CACHE.get('images') is None:
        images, labels = get_data(
            device=device, n_examples=n_examples, data_dir=data_dir)
        CACHE['images'] = images
        CACHE['labels'] = labels
    else:
        images = CACHE['images']
        labels = CACHE['labels']

    if CACHE.get('clean_acc') is None:
        clean_acc = clean_accuracy(model, images, labels)
        CACHE['clean_acc'] = clean_acc
    else:
        clean_acc = CACHE['clean_acc']

    try:
        kargs = {}
        if atk_class in ['SPSA']:
            kargs['max_batch_size'] = 5
        atk = eval("torchattacks."+atk_class)(model, device, **kargs)
        start = time.time()
        with torch.enable_grad():
            adv_images = atk(images, labels)
        end = time.time()
        robust_acc = clean_accuracy(model, adv_images, labels)
        sec = float(end - start)
        print('{0:<12}: clean_acc={1:2.2f} robust_acc={2:2.2f} sec={3:2.2f}'.format(
            atk_class, clean_acc, robust_acc, sec))

        if 'targeted' in atk.supported_mode:
            atk.set_mode_targeted_random(quiet=True)
            with torch.enable_grad():
                adv_images = atk(images, labels)
            robust_acc = clean_accuracy(model, adv_images, labels)
            sec = float(end - start)
            print('{0:<12}: clean_acc={1:2.2f} robust_acc={2:2.2f} sec={3:2.2f}'.format(
                "- targeted", clean_acc, robust_acc, sec))

    except Exception as e:
        robust_acc = clean_acc + 1  # It will cuase assertion.
        print('{0:<12} test acc Error'.format(atk_class))
        print(e)

    assert clean_acc >= robust_acc


@pytest.mark.parametrize('atk_class', [atk_class for atk_class in torchattacks.__all__ if atk_class not in torchattacks.__wrapper__])
def test_atks_on_cifar10(atk_class, device='cpu', n_examples=5, model_dir='./models', data_dir='./data'):
    net = train_model()
    net.eval()
    global CACHE
    CACHE['model'] = net
    test(atk_class, device, n_examples, model_dir, data_dir)
