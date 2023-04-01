import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import pytest

import torch
import torchattacks
from robustbench.data import load_cifar10
from robustbench.utils import load_model, clean_accuracy

CACHE = {}

def get_model(model_name='Standard', device="cpu", model_dir='./models'):
    model = load_model(model_name, model_dir=model_dir, norm='Linf')
    return model.to(device)


def get_data(data_name='CIFAR10', device="cpu", n_examples=5, data_dir='./data'):
    images, labels = load_cifar10(n_examples=n_examples, data_dir=data_dir)
    return images.to(device), labels.to(device)

@torch.no_grad()
@pytest.mark.parametrize("atk_class", [atk_class for atk_class in torchattacks.__all__ if atk_class not in torchattacks.__wrapper__])
def test_atks_on_cifar10(atk_class, device="cpu", n_examples=5, model_dir='./models', data_dir='./data'):
    if CACHE.get('model') is None:
        model = get_model(device=device, model_dir=model_dir)
        CACHE['model'] = model
    else:
        model = CACHE['model']

    if CACHE.get('images') is None:
        images, labels = get_data(device=device, n_examples=n_examples, data_dir=data_dir)
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
        atk = eval("torchattacks."+atk_class)(model, **kargs)
        start = time.time()
        with torch.enable_grad():
            adv_images = atk(images, labels)
        end = time.time()
        robust_acc = clean_accuracy(model, adv_images, labels)
        sec = float(end - start)
        print('{0:<12}: clean_acc={1:2.2f} robust_acc={2:2.2f} sec={3:2.2f}'.format(atk_class, clean_acc, robust_acc, sec))
        
        if 'targeted' in atk.supported_mode:
            atk.set_mode_targeted_random(quiet=True)
            with torch.enable_grad():
                adv_images = atk(images, labels)
            robust_acc = clean_accuracy(model, adv_images, labels)
            sec = float(end - start)
            print('{0:<12}: clean_acc={1:2.2f} robust_acc={2:2.2f} sec={3:2.2f}'.format("- targeted", clean_acc, robust_acc, sec))
        
    except Exception as e:
        robust_acc = clean_acc + 1  # It will cuase assertion.
        print('{0:<12} occurs Error'.format(atk_class))
        print(e)

    assert clean_acc >= robust_acc
