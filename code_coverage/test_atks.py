import sys
import os
# Importing the parent directory
# This line must be preceded by
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # nopep8

import torchattacks
import torch
import pytest
import time
import torch
from torchvision import datasets, transforms

from resnet import ResNet18


CACHE = {}


def get_model(device='cpu'):
    # load checkpoint.
    print(os.getcwd())
    checkpoint = torch.load(
        './code_coverage/resnet18_eval.pth', map_location=torch.device(device))
    net = ResNet18().to(device)
    net.load_state_dict(checkpoint['net'])
    net.eval()
    return net.to(device)


def get_data():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False)

    return test_loader


def clean_accuracy(model, images, labels):
    model.eval()
    total = 0
    correct = 0
    pred = torch.argmax(model(images), dim=1)
    correct += torch.sum(labels == pred)
    total += images.shape[0]
    return correct / total


@torch.no_grad()
@pytest.mark.parametrize('atk_class', [atk_class for atk_class in torchattacks.__all__ if atk_class not in torchattacks.__wrapper__])
def test_atks_on_cifar10(atk_class, device='cpu'):
    global CACHE
    if CACHE.get('model') is None:
        model = get_model(device=device)
        CACHE['model'] = model
    else:
        model = CACHE['model']

    if CACHE.get('testset') is None:
        test_loader = get_data()
        CACHE['testset'] = test_loader
    else:
        test_loader = CACHE['testset']

    if CACHE.get('clean_acc') is None:
        for (images, labels) in test_loader:
            clean_acc = clean_accuracy(model, test_loader)
            CACHE['clean_acc'] = clean_acc
            break
    else:
        clean_acc = CACHE['clean_acc']

    try:
        kargs = {}
        if atk_class in ['SPSA']:
            kargs['max_batch_size'] = 5
        atk = eval("torchattacks."+atk_class)(model, **kargs)
        start = time.time()
        with torch.enable_grad():
            for (images, labels) in test_loader:
                adv_images = atk(images, labels)
                break

        end = time.time()
        robust_acc = clean_accuracy(model, adv_images, labels)
        sec = float(end - start)
        print('{0:<12}: clean_acc={1:2.2f} robust_acc={2:2.2f} sec={3:2.2f}'.format(
            atk_class, clean_acc, robust_acc, sec))

        if 'targeted' in atk.supported_mode:
            atk.set_mode_targeted_random(quiet=True)
            with torch.enable_grad():
                for (images, labels) in test_loader:
                    adv_images = atk(images, labels)
                    break

            robust_acc = clean_accuracy(model, adv_images, labels)
            sec = float(end - start)
            print('{0:<12}: clean_acc={1:2.2f} robust_acc={2:2.2f} sec={3:2.2f}'.format(
                "- targeted", clean_acc, robust_acc, sec))

    except Exception as e:
        robust_acc = clean_acc + 1  # It will cuase assertion.
        print('{0:<12} test acc Error'.format(atk_class))
        print(e)

    assert clean_acc >= robust_acc
