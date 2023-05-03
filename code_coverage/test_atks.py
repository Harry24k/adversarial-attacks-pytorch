import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import pytest

from tqdm.autonotebook import tqdm
import torch
import torchattacks
from robustbench.data import load_cifar10, load_cifar100, load_imagenet
from robustbench.utils import load_model, clean_accuracy

#import detectors
import timm

CACHE_img1k = {}
CACHE_c10 = {}
CACHE_c100 = {}

def get_model(model_name='Standard', device="cpu", model_dir='./models'):
    model = load_model(model_name, model_dir=model_dir, norm='Linf')
    return model.to(device)


def get_data_cifar10(data_name='CIFAR10', device="cpu", n_examples=5, data_dir='./data'):
    #images, labels = load_cifar10(n_examples=n_examples, data_dir=data_dir)
    dataset = load_cifar10(data_dir=data_dir, shashank=True)
    return dataset

def get_data_cifar100(data_name='CIFAR100', device="cpu", n_examples=5, data_dir='./data'):
    #images, labels = load_cifar10(n_examples=n_examples, data_dir=data_dir)
    dataset = load_cifar100(data_dir=data_dir, shashank=True)
    return dataset

@torch.no_grad()
@pytest.mark.parametrize("atk_class", [atk_class for atk_class in torchattacks.__all__ if atk_class not in torchattacks.__wrapper__])
def test_atks_on_cifar10(atk_class, device="cpu", n_examples=128, model_dir='./models', data_dir='./data'):
    import detectors
    if CACHE_c10.get('model') is None:
        #model = get_model(device=device, model_dir=model_dir)
        model = timm.create_model("resnet50_cifar10", pretrained=True).to(device)
        CACHE_c10['model'] = model
    else:
        model = CACHE_c10['model']

    dataset = get_data_cifar10(device=device, n_examples=n_examples, data_dir=data_dir)    

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=24,
        pin_memory=True,
        drop_last=False
    )

    
    clean_acc, robust_acc = 0, 0
    if CACHE_c10.get('clean_acc') is None:
        with tqdm(test_loader, unit="batch") as tepoch:
            for images, labels in tepoch:
                images, labels = images.to(device), labels.to(device)
                clean_acc += clean_accuracy(model, images, labels)
        clean_acc /= len(tepoch)
        CACHE_c10['clean_acc'] = clean_acc
    else:
        clean_acc = CACHE_c10['clean_acc']

    try:
        kargs = {}
        if atk_class in ['SPSA']:
            kargs['max_batch_size'] = 5
        atk = eval("torchattacks."+atk_class)(model, **kargs)
        start = time.time()
        with torch.enable_grad():
            with tqdm(test_loader, unit="batch", desc=atk_class) as tepoch:
                for images, labels in tepoch:
                    image, labels = images.to(device), labels.to(device)
                    adv_images = atk(images, labels)
                    robust_acc += clean_accuracy(model, adv_images, labels)
        end = time.time()
        robust_acc /= len(tepoch)
        
        sec = float(end - start)
        print('{0:<12}: clean_acc={1:2.4f} robust_acc={2:2.4f} sec={3:2.4f}'.format(atk_class, clean_acc, robust_acc, sec))
        
        robust_acc = 0
        if 'targeted' in atk.supported_mode:
            atk.set_mode_targeted_random(quiet=True)
            with torch.enable_grad():
                with tqdm(test_loader, unit="batch", desc=atk_class) as tepoch:
                    for images, labels in tepoch:
                        image, labels = images.to(device), labels.to(device)
                        adv_images = atk(images, labels)
                        robust_acc += clean_accuracy(model, adv_images, labels)
            end = time.time()
            sec = float(end - start)
            robust_acc /= len(tepoch)
            print('{0:<12}: clean_acc={1:2.4f} robust_acc={2:2.4f} sec={3:2.4f}'.format("- targeted", clean_acc, robust_acc, sec))
        
    except Exception as e:
        robust_acc = clean_acc + 1  # It will cuase assertion.
        print('{0:<12} occurs Error'.format(atk_class))
        print(e)

    assert clean_acc >= robust_acc


@torch.no_grad()
@pytest.mark.parametrize("atk_class", [atk_class for atk_class in torchattacks.__all__ if atk_class not in torchattacks.__wrapper__])
def test_atks_on_cifar100(atk_class, device="cpu", n_examples=128, model_dir='./models', data_dir='./data'):
    import detectors
    if CACHE_c100.get('model') is None:
        #model = get_model(device=device, model_dir=model_dir)
        model = timm.create_model("resnet50_cifar100", pretrained=True).to(device)
        CACHE_c100['model'] = model
    else:
        model = CACHE_c100['model']

    dataset = get_data_cifar100(device=device, n_examples=n_examples, data_dir=data_dir)    

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=24,
        pin_memory=True,
        drop_last=False
    )

    
    clean_acc, robust_acc = 0, 0
    if CACHE_c100.get('clean_acc') is None:
        with tqdm(test_loader, unit="batch") as tepoch:
            for images, labels in tepoch:
                images, labels = images.to(device), labels.to(device)
                clean_acc += clean_accuracy(model, images, labels)
        clean_acc /= len(tepoch)
        CACHE_c100['clean_acc'] = clean_acc
    else:
        clean_acc = CACHE_c100['clean_acc']

    try:
        kargs = {}
        if atk_class in ['SPSA']:
            kargs['max_batch_size'] = 5
        atk = eval("torchattacks."+atk_class)(model, **kargs)
        start = time.time()
        with torch.enable_grad():
            with tqdm(test_loader, unit="batch", desc=atk_class) as tepoch:
                for images, labels in tepoch:
                    image, labels = images.to(device), labels.to(device)
                    adv_images = atk(images, labels)
                    robust_acc += clean_accuracy(model, adv_images, labels)
        end = time.time()
        robust_acc /= len(tepoch)
        
        sec = float(end - start)
        print('{0:<12}: clean_acc={1:2.4f} robust_acc={2:2.4f} sec={3:2.4f}'.format(atk_class, clean_acc, robust_acc, sec))
        
        robust_acc = 0
        if 'targeted' in atk.supported_mode:
            atk.set_mode_targeted_random(quiet=True)
            with torch.enable_grad():
                with tqdm(test_loader, unit="batch", desc=atk_class) as tepoch:
                    for images, labels in tepoch:
                        image, labels = images.to(device), labels.to(device)
                        adv_images = atk(images, labels)
                        robust_acc += clean_accuracy(model, adv_images, labels)
            end = time.time()
            sec = float(end - start)
            robust_acc /= len(tepoch)
            print('{0:<12}: clean_acc={1:2.4f} robust_acc={2:2.4f} sec={3:2.4f}'.format("- targeted", clean_acc, robust_acc, sec))
        
    except Exception as e:
        robust_acc = clean_acc + 1  # It will cuase assertion.
        print('{0:<12} occurs Error'.format(atk_class))
        print(e)

    assert clean_acc >= robust_acc


@torch.no_grad()
@pytest.mark.parametrize("atk_class", [atk_class for atk_class in torchattacks.__all__ if atk_class not in torchattacks.__wrapper__])
def test_atks_on_imagenet1k(atk_class, device="cpu", n_examples=128, model_dir='./models', data_dir='./data', model='resnet50'):
    model_name = model
    if CACHE_img1k.get('model') is None:
        #model = get_model(device=device, model_dir=model_dir)
        model = timm.create_model(model, pretrained=True).to(device)
        CACHE_img1k['model'] = model
    else:
        model = CACHE_img1k['model']

    dataset = load_imagenet(data_dir=data_dir, shashank=True)

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=24,
        pin_memory=True,
        drop_last=False
    )

    #lent=len(test_loader)
    #for i, data in enumerate(test_loader):
    #    import ipdb;ipdb.set_trace() 

    
    clean_acc, robust_acc = 0, 0
    if CACHE_img1k.get('clean_acc') is None:
        with tqdm(test_loader, unit="batch") as tepoch:
            for images, labels, _ in tepoch:
                images, labels = images.to(device), labels.to(device)
                clean_acc += clean_accuracy(model, images, labels)
        clean_acc /= len(tepoch)
        CACHE_img1k['clean_acc'] = clean_acc
    else:
        clean_acc = CACHE_img1k['clean_acc']

    try:
        kargs = {}
        if atk_class in ['SPSA']:
            kargs['max_batch_size'] = 5
        atk = eval("torchattacks."+atk_class)(model, **kargs)
        start = time.time()
        with torch.enable_grad():
            with tqdm(test_loader, unit="batch", desc=model_name+'_'+atk_class) as tepoch:
                for images, labels, _ in tepoch:
                    image, labels = images.to(device), labels.to(device)
                    adv_images = atk(images, labels)
                    robust_acc += clean_accuracy(model, adv_images, labels)
        end = time.time()
        robust_acc /= len(tepoch)
        
        sec = float(end - start)
        print('{0:<12}: clean_acc={1:2.4f} robust_acc={2:2.4f} sec={3:2.4f}'.format(atk_class, clean_acc, robust_acc, sec))
        
        robust_acc = 0
        if 'targeted' in atk.supported_mode:
            atk.set_mode_targeted_random(quiet=True)
            with torch.enable_grad():
                with tqdm(test_loader, unit="batch", desc=model_name+'_'+atk_class) as tepoch:
                    for images, labels, _ in tepoch:
                        image, labels = images.to(device), labels.to(device)
                        adv_images = atk(images, labels)
                        robust_acc += clean_accuracy(model, adv_images, labels)
            end = time.time()
            sec = float(end - start)
            robust_acc /= len(tepoch)
            print('{0:<12}: clean_acc={1:2.4f} robust_acc={2:2.4f} sec={3:2.4f}'.format("- targeted", clean_acc, robust_acc, sec))
        
    except Exception as e:
        robust_acc = clean_acc + 1  # It will cuase assertion.
        print('{0:<12} occurs Error'.format(atk_class))
        print(e)

    #assert clean_acc >= robust_acc