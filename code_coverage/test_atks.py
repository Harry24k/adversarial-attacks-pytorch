# Importing the parent directory
# This line must be preceded by
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import torchattacks
import pytest
import time
import torch

from script.resnet import ResNet18


CACHE = {}


def get_model(device='cpu'):
    # load checkpoint.
    print(os.getcwd())
    checkpoint = torch.load('./code_coverage/resnet18_eval.pth',
                            map_location=torch.device(device))
    net = ResNet18().to(device)
    net.load_state_dict(checkpoint['net'])
    return net.to(device)


def get_data(device='cpu'):
    images = torch.load('./code_coverage/images.pth')  # 10 images
    labels = torch.load('./code_coverage/labels.pth')  # 10 images
    return images.to(device), labels.to(device)


def clean_accuracy(model, images, labels):
    model.eval()
    pred = torch.argmax(model(images), dim=1)
    correct = torch.sum(labels == pred)
    total = images.shape[0]
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

    if CACHE.get('images') is None or CACHE.get('labels') is None:
        images, labels = get_data()
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

    kargs = {}
    if atk_class in ['SPSA']:
        kargs['max_batch_size'] = 5

    atk = eval("torchattacks."+atk_class)(model, **kargs)
    start = time.time()
    with torch.enable_grad():
        adv_images = atk(images, labels)

    # non-targeted attack test
    robust_acc_1 = clean_accuracy(model, adv_images, labels)
    assert clean_acc >= robust_acc_1
    end = time.time()

    sec = float(end - start)
    print('{0:<12}: clean_acc={1:2.2f} robust_acc={2:2.2f} sec={3:2.2f}'.format(
        atk_class, clean_acc, robust_acc_1, sec))

    # targeted attack test
    start = time.time()
    if 'targeted' in atk.supported_mode:
        atk.set_mode_targeted_random(quiet=True)
        start = time.time()
        with torch.enable_grad():
            adv_images = atk(images, labels)
        end = time.time()
        robust_acc_2 = clean_accuracy(model, adv_images, labels)
    else:
        robust_acc_2 = 0
    assert clean_acc >= robust_acc_2
    end = time.time()

    sec = float(end - start)
    print('{0:<12}: clean_acc={1:2.2f} robust_acc={2:2.2f} sec={3:2.2f}'.format(
        "- targeted", clean_acc, robust_acc_2, sec))
