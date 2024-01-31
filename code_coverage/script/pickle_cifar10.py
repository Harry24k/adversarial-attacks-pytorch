import torch
from torchvision import datasets, transforms

transform_test = transforms.Compose([
    transforms.ToTensor(),
])
testset = datasets.CIFAR10(
    root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=10, shuffle=False)


def split(testloader):
    for (x, y) in testloader:
        torch.save(x, 'images.pth')
        torch.save(y, 'labels.pth')
        break


def main():
    split(test_loader)


if __name__ == '__main__':
    main()
