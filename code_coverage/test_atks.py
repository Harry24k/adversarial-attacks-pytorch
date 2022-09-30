import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torchattacks

import robustbench_extracted as robustbench
from robustbench.data import load_cifar10
from robustbench.utils import load_model, clean_accuracy

def get_model(model_name='Standard', device="cpu"):
    model = load_model(model_name, norm='Linf')
    return model.to(device)
    
def get_data(data_name='CIFAR10', device="cpu"):
    images, labels = load_cifar10(n_examples=50)
    return images.to(device), labels.to(device)

def test_atks_on_cifar10(device="cpu"):
    model = get_model()
    images, labels = get_data()
    clean_acc = clean_accuracy(model, images, labels)

    for atk_class in torchattacks.__all__:
        atk = eval("torchattacks."+atk_class)(model)
        adv_images = atk(images, labels)
        robust_acc = clean_accuracy(model, adv_images, labels)

    assert clean_acc <= robust_acc