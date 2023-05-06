import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import pytest

from tqdm.autonotebook import tqdm
import torch
import torchattacks
from robustbench.data import load_cifar10, load_cifar100, load_imagenet
from robustbench.utils import load_model, clean_accuracy, get_grad_cam, save_clean_image, reset_iter, top_k_accuracy

#import detectors
import timm

import json

CACHE_img1k = {}
CACHE_c10 = {}
CACHE_c100 = {}
CACHE = {}

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


@torch.enable_grad()
@pytest.mark.parametrize("atk_class", [atk_class for atk_class in torchattacks.__testing__ if atk_class not in torchattacks.__wrapper__])
def test_atks(dataset, atk_class, device="cpu", n_examples=128, model_dir='./models', data_dir='./data', model='resnet50', steps=1):
    model_name = model
    dataset_name = dataset
    num_classes=10

    final_results = {}
    results = {}
    detailed_results = {}

    cifar_10_models_paths = {'resnet50':None, 'wide_resnet50_2':'', 'convnext_tiny':'', 'resnext50_32x4d':'', 'vit_small_patch16_224':'' }
    cifar_100_models_paths = {'resnet50':None, 'wide_resnet50_2':'', 'convnext_tiny':'', 'resnext50_32x4d':'', 'vit_small_patch16_224':'' }

    if CACHE.get('model_name') is None:
        CACHE['model_name'] = model_name
    elif CACHE.get('model_name') is not None and CACHE.get('model') is not None:
        if CACHE['model_name'] == model_name:
            model = CACHE['model']
        elif CACHE['model_name'] != model_name:
            CACHE['model'] = None
            CACHE['clean_acc'] = None
            CACHE['clean_acc_top5'] = None
    
    if dataset_name == 'imagenet1k':
        num_classes = 1000
        if CACHE.get('model') is None:
            #model = get_model(device=device, model_dir=model_dir)
            model = timm.create_model(model, pretrained=True).to(device)
            CACHE_img1k['model'] = model
        else:
            model = CACHE_img1k['model']
        dataset = load_imagenet(data_dir=data_dir, shashank=True)

    elif dataset_name == 'cifar100':
        num_classes=100
        if CACHE.get('model') is None and model is None:
            import detectors
            #model = get_model(device=device, model_dir=model_dir)
            model = timm.create_model("resnet50_cifar100", pretrained=True).to(device)
            CACHE['model'] = model
        elif CACHE.get('model') is None:
            model_path = cifar_100_models_paths[model_name]
            if model_path == None:
                import detectors            
                model = timm.create_model("resnet50_cifar100", pretrained=True).to(device)
            else:
                import tejaswini_models
                model = tejaswini_models.cifar100_models[model_name]
                model.load_state_dict(torch.load(model_path))

        else:
            model = CACHE['model']

        dataset = get_data_cifar100(device=device, n_examples=n_examples, data_dir=data_dir)

    elif dataset_name == 'cifar10':
        if CACHE.get('model') is None and model is None:
            #model = get_model(device=device, model_dir=model_dir)
            import detectors
            model = timm.create_model("resnet50_cifar10", pretrained=True).to(device)
            CACHE['model'] = model
        elif CACHE.get('model') is None:
            model_path = cifar_10_models_paths[model_name]
            if model_path == None:
                import detectors            
                model = timm.create_model("resnet50_cifar10", pretrained=True).to(device)
            else:
                import tejaswini_models
                model = tejaswini_models.cifar10_models[model_name]
                model.load_state_dict(torch.load(model_path))
        else:
            model = CACHE['model']

        dataset = get_data_cifar10(device=device, n_examples=n_examples, data_dir=data_dir) 

        

    

    save_path = 'results/' + dataset_name + '/' + model_name + '/' + str(atk_class) + '/' + str(steps) + '/eps_2_255/'

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

    
    clean_acc, robust_acc, testing = 0, 0, 0
    clean_acc_top5, robust_acc_top5 = 0, 0
    iterator = 0
    reset_iter()
    if CACHE.get('clean_acc') is None:
        with tqdm(test_loader, unit="batch") as tepoch:
            for images, labels, _ in tepoch:
                images, labels = images.to(device), labels.to(device)
                #save_clean_image(images.clone().detach().cpu(), save_path)
                #clean_acc += clean_accuracy(model, images, labels)
                #import ipdb;ipdb.set_trace()
                top_1, top_5, confidences, preds, targets  = top_k_accuracy(model, images, labels, device=device)
                for conf, pred, tars in zip(confidences, preds, targets):
                    #import ipdb;ipdb.set_trace()
                    results['confidence'] = conf.tolist()
                    results['predictions'] = pred.tolist()
                    results['targets'] = tars.tolist()
                    name = 'image_00' + str(iterator)
                    detailed_results[name] = results
                    results={}
                    iterator +=1
                clean_acc += top_1
                clean_acc_top5 += top_5
        clean_acc /= len(tepoch)
        clean_acc_top5 /= len(tepoch)
        #print(clean_acc, testing, clean_acc_top5)
        CACHE['clean_acc'] = clean_acc
        CACHE['clean_acc_top5'] = clean_acc_top5
    else:
        clean_acc = CACHE['clean_acc']
        clean_acc_top5 = CACHE['clean_acc_top5']
    
    json_paths = os.path.join(save_path, 'json')
    os.makedirs(json_paths, exist_ok=True)
    clean_json_loc = json_paths + '/clean_perf.json'
    with open(clean_json_loc, 'w') as f:
        json.dump(detailed_results, f)
    detailed_results = {}
    iterator = 0

    try:
        kargs = {}
        reset_iter()
        if atk_class in ['SPSA']:
            kargs['max_batch_size'] = 5
        if atk_class in ['PGD', 'APGD', 'CosPGD', 'CosPGD_softmax', 'DIFGSM', 'UPGD', 'MIFGSM', 'APGDT', 'APGD_DLR']:
            try:
                atk = eval("torchattacks."+atk_class)(model, eps=2/255, steps=steps, n_classes=num_classes, **kargs)    
            except Exception:
                atk = eval("torchattacks."+atk_class)(model, eps=2/255, steps=steps, **kargs)    
        else:
            try:
                atk = eval("torchattacks."+atk_class)(model, eps=2/255, n_classes=num_classes, **kargs)
            except Exception:
                atk = eval("torchattacks."+atk_class)(model, eps=2/255, **kargs)
        saving_time = 0
        start = time.time()
        with torch.enable_grad():
            with tqdm(test_loader, unit="batch", desc=model_name+'_'+atk_class) as tepoch:                
                for images, labels, _ in tepoch:
                    image, labels = images.to(device), labels.to(device)
                    adv_images = atk(images, labels)
                    save_start = time.time()
                    get_grad_cam(model=model, input=adv_images, save_path=save_path + '/non-targeted')
                    #robust_acc += clean_accuracy(model, adv_images, labels)
                    top_1, top_5, confidences, preds, targets = top_k_accuracy(model, adv_images, labels, device=device)
                    for conf, pred, tars in zip(confidences, preds, targets):
                        results['confidence'] = conf.tolist()
                        results['predictions'] = pred.tolist()
                        results['targets'] = tars.tolist()
                        name = 'image_00' + str(iterator)
                        iterator +=1
                        detailed_results[name] = results
                        results={}
                    save_end = time.time()
                    saving_time += float(save_end - save_start)
                    robust_acc +=top_1
                    robust_acc_top5 += top_5
        end = time.time()
        robust_acc /= len(tepoch)
        robust_acc_top5 /= len(tepoch)
        
        sec = float(end - start) - saving_time
        #print(clean_acc, clean_acc_top5, robust_acc, robust_acc_top5)
        print('{0:<12}: clean_acc={1:2.4f} clean_acc top5={2:2.4f} robust_acc={3:2.4f} robust_acc top5={4:2.4f} sec={5:2.4f}'.format(atk_class, clean_acc, clean_acc_top5, robust_acc, robust_acc_top5, sec))
        
        final_results['Model'] = model_name
        final_results['dataset'] = dataset_name
        final_results['Attack'] = atk_class
        final_results['Attack_type'] = 'NON-targeted'
        final_results['iterations'] = steps
        final_results['eps'] = 2/255
        final_results['Clean_Acc'] = clean_acc
        final_results['Clean_Top5']=clean_acc_top5
        final_results['Robust_Acc']=robust_acc
        final_results['Robust_Top5']=robust_acc_top5
        final_results['Time'] = sec


        nontargeted_detail_json_loc = json_paths + '/nontargeted_detailed_perf.json'
        with open(nontargeted_detail_json_loc, 'w') as f:
            json.dump(detailed_results, f)
        detailed_results = {}
        iterator = 0

        nontargeted_json_loc = json_paths + '/nontargeted_perf.json'
        with open(nontargeted_json_loc, 'w') as f:
            json.dump(final_results, f)
        final_results = {}
        iterator = 0
        
        robust_acc = 0
        robust_acc_top5 = 0
        saving_time = 0
        reset_iter()
        if 'targeted' in atk.supported_mode:
            start = time.time()
            atk.set_mode_targeted_random(quiet=True)
            with torch.enable_grad():
                with tqdm(test_loader, unit="batch", desc=model_name+'_'+atk_class) as tepoch:
                    for images, labels, _ in tepoch:
                        image, labels = images.to(device), labels.to(device)
                        adv_images = atk(images, labels)
                        save_start = time.time()
                        get_grad_cam(model=model, input=adv_images, save_path=save_path + '/targeted')
                        #robust_acc += clean_accuracy(model, adv_images, labels)
                        top_1, top_5, confidences, preds, targets = top_k_accuracy(model, adv_images, labels, device=device)
                        for conf, pred, tars in zip(confidences, preds, targets):
                            results['confidence'] = conf.tolist()
                            results['predictions'] = pred.tolist()
                            results['targets'] = tars.tolist()
                            name = 'image_00' + str(iterator)
                            iterator +=1
                            detailed_results[name] = results
                            results={}
                        save_end = time.time()
                        saving_time += float(save_end - save_start)
                        robust_acc +=top_1
                        robust_acc_top5 += top_5
            end = time.time()
            sec = float(end - start) - saving_time
            robust_acc /= len(tepoch)
            robust_acc_top5 /= len(tepoch)
            print('{0:<12}: clean_acc={1:2.4f} clean_acc top5={2:2.4f} robust_acc={3:2.4f} robust_acc top5={4:2.4f} sec={5:2.4f}'.format("- targeted", clean_acc, clean_acc_top5, robust_acc, robust_acc_top5, sec))

            final_results['Model'] = model_name
            final_results['dataset'] = dataset_name
            final_results['Attack'] = atk_class
            final_results['Attack_type'] = 'Targeted'
            final_results['iterations'] = steps
            final_results['eps'] = 2/255
            final_results['Clean_Acc'] = clean_acc
            final_results['Clean_Top5']=clean_acc_top5
            final_results['Robust_Acc']=robust_acc
            final_results['Robust_Top5']=robust_acc_top5
            final_results['Time'] = sec


            targeted_detail_json_loc = json_paths + '/targeted_detailed_perf.json'
            with open(targeted_detail_json_loc, 'w') as f:
                json.dump(detailed_results, f)
            detailed_results = {}
            iterator = 0

            targeted_json_loc = json_paths + '/targeted_perf.json'
            with open(targeted_json_loc, 'w') as f:
                json.dump(final_results, f)
            final_results = {}
            iterator = 0
        
    except Exception as e:
        robust_acc = clean_acc + 1  # It will cuase assertion.
        print('{0:<12} occurs Error'.format(atk_class))
        print(e)

















#@torch.no_grad()
@torch.enable_grad()
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
                    get_grad_cam(model=model, input=adv_images, save_path='runs/testing/')
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
    reset_iter()
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
        reset_iter()
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
        reset_iter()
        if 'targeted' in atk.supported_mode:
            atk.set_mode_targeted_random(quiet=True)
            with torch.enable_grad():
                with tqdm(test_loader, unit="batch", desc=atk_class) as tepoch:
                    for images, labels in tepoch:
                        image, labels = images.to(device), labels.to(device)
                        adv_images = atk(images, labels)
                        get_grad_cam(model=model, input=adv_images, save_path='runs/testing/')
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


@torch.enable_grad()
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

    save_path = 'results/imagenet1k/' + str(atk_class) + '/' + model_name

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

    
    clean_acc, robust_acc, testing = 0, 0, 0
    clean_acc_top5, robust_acc_top5 = 0, 0
    reset_iter()
    if CACHE_img1k.get('clean_acc') is None:
        with tqdm(test_loader, unit="batch") as tepoch:
            for images, labels, _ in tepoch:
                images, labels = images.to(device), labels.to(device)
                #save_clean_image(images.clone().detach().cpu(), save_path)
                #clean_acc += clean_accuracy(model, images, labels)
                #import ipdb;ipdb.set_trace()
                top_1, top_5, confidences, preds, targets  = top_k_accuracy(model, images, labels, device=device)
                clean_acc += top_1
                clean_acc_top5 += top_5
        clean_acc /= len(tepoch)
        clean_acc_top5 /= len(tepoch)
        #print(clean_acc, testing, clean_acc_top5)
        CACHE_img1k['clean_acc'] = clean_acc
        CACHE_img1k['clean_acc_top5'] = clean_acc_top5
    else:
        clean_acc = CACHE_img1k['clean_acc']
        clean_acc_top5 = CACHE_img1k['clean_acc_top5']

    try:
        kargs = {}
        reset_iter()
        if atk_class in ['SPSA']:
            kargs['max_batch_size'] = 5
        atk = eval("torchattacks."+atk_class)(model, **kargs)
        saving_time = 0
        start = time.time()
        with torch.enable_grad():
            with tqdm(test_loader, unit="batch", desc=model_name+'_'+atk_class) as tepoch:                
                for images, labels, _ in tepoch:
                    image, labels = images.to(device), labels.to(device)
                    adv_images = atk(images, labels)
                    save_start = time.time()
                    get_grad_cam(model=model, input=adv_images, save_path=save_path + '/non-targeted')
                    save_end = time.time()
                    saving_time += float(save_end - save_start)
                    #robust_acc += clean_accuracy(model, adv_images, labels)
                    top_1, top_5, confidences, preds, targets = top_k_accuracy(model, adv_images, labels, device=device)
                    robust_acc +=top_1
                    robust_acc_top5 += top_5
        end = time.time()
        robust_acc /= len(tepoch)
        robust_acc_top5 /= len(tepoch)
        
        sec = float(end - start) - saving_time
        #print(clean_acc, clean_acc_top5, robust_acc, robust_acc_top5)
        print('{0:<12}: clean_acc={1:2.4f} clean_acc top5={2:2.4f} robust_acc={3:2.4f} robust_acc top5={4:2.4f} sec={5:2.4f}'.format(atk_class, clean_acc, clean_acc_top5, robust_acc, robust_acc_top5, sec))
        
        robust_acc = 0
        robust_acc_top5 = 0
        saving_time = 0
        reset_iter()
        if 'targeted' in atk.supported_mode:
            start = time.time()
            atk.set_mode_targeted_random(quiet=True)
            with torch.enable_grad():
                with tqdm(test_loader, unit="batch", desc=model_name+'_'+atk_class) as tepoch:
                    for images, labels, _ in tepoch:
                        image, labels = images.to(device), labels.to(device)
                        adv_images = atk(images, labels)
                        save_start = time.time()
                        get_grad_cam(model=model, input=adv_images, save_path=save_path + '/targeted')
                        save_end = time.time()
                        saving_time += float(save_end - save_start)
                        #robust_acc += clean_accuracy(model, adv_images, labels)
                        top_1, top_5, confidences, preds, targets = top_k_accuracy(model, adv_images, labels, device=device)
                        robust_acc +=top_1
                        robust_acc_top5 += top_5
            end = time.time()
            sec = float(end - start) - saving_time
            robust_acc /= len(tepoch)
            robust_acc_top5 /= len(tepoch)
            print('{0:<12}: clean_acc={1:2.4f} clean_acc top5={2:2.4f} robust_acc={3:2.4f} robust_acc top5={4:2.4f} sec={5:2.4f}'.format("- targeted", clean_acc, clean_acc_top5, robust_acc, robust_acc_top5, sec))
        
    except Exception as e:
        robust_acc = clean_acc + 1  # It will cuase assertion.
        print('{0:<12} occurs Error'.format(atk_class))
        print(e)

    #assert clean_acc >= robust_acc