# %%
from test_atks import test_atks
from test_import import test_import_version

# %%
test_import_version()

# %%
import torchattacks

# %%
print("*************************************              TESTING GRAD CAM              *************************************")
datasets = ['cifar10', 'cifar100']
models = ["convnext_tiny", "resnet50", "vit_small_patch16_224", "wide_resnet50_2"]
iter_attacks=['PGD', 'APGD', 'CosPGD', 'CosPGD_softmax', 'DIFGSM', 'UPGD', 'MIFGSM', 'APGDT', 'APGD_DLR']
for dataset in datasets:
    for model in models:
        for atk_class in [atk_class for atk_class in torchattacks.__testing__ if atk_class not in torchattacks.__wrapper__]:
            if atk_class in iter_attacks:
                steps = [3,5,10,20,40]
                for step in steps:
                    test_atks(dataset=dataset,
                                atk_class=atk_class,
                                device='cuda',
                                model_dir='../demo/models/',
                                data_dir='/home/prasse/Shashank_Projects/adversarial-attacks-pytorch/data',
                                model=model,
                                steps=step)
            else:
                test_atks(dataset=dataset,
                                atk_class=atk_class,
                                device='cuda',
                                model_dir='../demo/models/',
                                data_dir='/home/prasse/Shashank_Projects/adversarial-attacks-pytorch/data',
                                model=model,
                                steps=1)