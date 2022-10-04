import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time

import torch
import torchattacks
from robustbench.data import load_cifar10
from robustbench.utils import load_model, clean_accuracy


def get_model(model_name='Standard', device="cpu", model_dir='./models'):
    model = load_model(model_name, model_dir=model_dir, norm='Linf')
    return model.to(device)


def get_data(data_name='CIFAR10', device="cpu", n_examples=5, data_dir='./data'):
    images, labels = load_cifar10(n_examples=n_examples, data_dir=data_dir)
    return images.to(device), labels.to(device)


def test_atks_on_cifar10(device="cpu", n_examples=5, model_dir='./models', data_dir='./data'):
    model = get_model(device=device, model_dir=model_dir)
    images, labels = get_data(device=device, n_examples=n_examples, data_dir=data_dir)
    clean_acc = clean_accuracy(model, images, labels)

    for atk_class in torchattacks.__all__:
        if atk_class in torchattacks.__wrapper__:
            continue
        try:
            atk = eval("torchattacks."+atk_class)(model)
            start = time.time()
            adv_images = atk(images, labels)
            end = time.time()
            robust_acc = clean_accuracy(model, adv_images, labels)
            sec = float(end - start)
            print('{0:<12}: clean_acc={1:2.2f} robust_acc={2:2.2f} sec={3:2.2f}'.format(atk_class, clean_acc, robust_acc, sec))
        except Exception as e:
            print('{0:<12} ocurrs Error'.format(atk_class))
            print(e)

    assert clean_acc >= robust_acc
