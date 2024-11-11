import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from tqdm import tqdm
import time
import os

from resnet import ResNet18

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
testset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=32)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=32)


def train(net, loss, optimizer, trainloader, epoch):
    # Training
    print('\nEpoch: {}'.format(epoch))
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    with tqdm(total=len(trainloader), desc='Train') as tbar:
        for batch_idx, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)
            total += y.shape[0]

            optimizer.zero_grad()
            outputs = net(x)
            _loss = loss(outputs, y)
            _loss.backward()
            optimizer.step()

            train_loss += _loss.item()
            predicted = torch.argmax(outputs, 1)
            correct += torch.sum((predicted == y)).item()

            lr = optimizer.param_groups[0].get('lr')
            tbar.set_postfix(loss=train_loss/(batch_idx+1),
                             acc=(correct/total)*100., lr=lr)
            tbar.update()

    return (correct / total) * 100.


def test(net, loss, testloader, epoch, best_acc):
    # Test
    net.eval()
    correct = 0
    total = 0
    test_loss = 0

    with tqdm(total=len(testloader), desc='Test') as tbar:
        for batch_idx, (x, y) in enumerate(testloader):
            x, y = x.to(device), y.to(device)
            total += y.shape[0]

            outputs = net(x)
            _loss = loss(outputs, y)
            test_loss += _loss.item()
            predicted = torch.argmax(outputs, 1)
            correct += torch.sum((predicted == y)).item()

            # tbar.set_description('loss: {:.3f} acc: {:.3f} aacc: {:.3f}'.format(
            #     test_loss/(batch_idx+1, (correct/total)*100, (adv_correct/total)*100)))
            tbar.set_postfix(loss=test_loss/(batch_idx+1),
                             acc=(correct/total)*100.)
            tbar.update()

    acc = (correct / total) * 100.
    # Save checkpoint.
    if acc > best_acc:
        best_acc = acc
        p = os.path.join('./', f'resnet18_eval.pth')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, p)

    return best_acc


def main():
    lr = 0.01
    momentum = 0.9
    weight_decay = 3.5e-3
    best_acc = 0
    epochs = 100

    net = ResNet18().to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr,
                          momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100)
    for epoch in range(1, epochs + 1):
        _ = train(net, loss, optimizer, train_loader, epoch)  # nopep8
        best_acc = test(net, loss, test_loader, epoch, best_acc)  # nopep8
        scheduler.step()

    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print(f'END: {time_str}')


if __name__ == '__main__':
    main()
