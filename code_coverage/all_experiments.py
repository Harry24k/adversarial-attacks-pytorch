# %%
from test_atks import test_atks_on_cifar10, test_atks_on_cifar100, test_atks_on_imagenet1k
from test_import import test_import_version

# %%
test_import_version()

# %%
import torchattacks

# %%
print("*************************************              TESTING GRAD CAM              *************************************")
models = ["convnext_tiny", "resnet50", "vit_small_patch16_224", "wide_resnet50_2"]
for model in models:
    for atk_class in [atk_class for atk_class in torchattacks.__testing__ if atk_class not in torchattacks.__wrapper__]:
        test_atks_on_imagenet1k(atk_class=atk_class,
                                device='cuda',
                                model_dir='../demo/models/',
                                data_dir='/home/prasse/Shashank_Projects/adversarial-attacks-pytorch/data',
                                model=model)

"""
# %%
print("*************************************              IMAGENET-1k              *************************************")
models = ["convnext_tiny", "resnet50", "vit_small_patch16_224", "wide_resnet50_2"]
for model in models:
    for atk_class in [atk_class for atk_class in torchattacks.__all__ if atk_class not in torchattacks.__wrapper__]:
        test_atks_on_imagenet1k(atk_class=atk_class,
                                device='cuda',
                                model_dir='../demo/models/',
                                data_dir='/home/prasse/Shashank_Projects/adversarial-attacks-pytorch/data',
                                model=model)

# %%
print("*************************************              CIFAR-10              *************************************")
for atk_class in [atk_class for atk_class in torchattacks.__all__ if atk_class not in torchattacks.__wrapper__]:
    test_atks_on_cifar10(atk_class=atk_class,
                         device='cuda',
                         model_dir='/home/prasse/Shashank_Projects/adversarial-attacks-pytorch/data',
                         data_dir='../demo/data/')





# %%
print("*************************************              CIFAR-100              *************************************")
for atk_class in [atk_class for atk_class in torchattacks.__all__ if atk_class not in torchattacks.__wrapper__]:
    test_atks_on_cifar100(atk_class=atk_class,
                         device='cuda',
                         model_dir='../demo/models/',
                         data_dir='../demo/data/')

# %%
"""
"""
VANILA      : clean_acc=0.8148 robust_acc=0.8148 sec=5.7115
convnext_tiny_GN: 100%
40/40 [00:05<00:00, 11.27batch/s]
GN          : clean_acc=0.8148 robust_acc=0.7414 sec=5.6742
convnext_tiny_FGSM: 100%
40/40 [00:10<00:00, 4.72batch/s]
FGSM        : clean_acc=0.8148 robust_acc=0.3105 sec=10.5411
convnext_tiny_FGSM: 100%
40/40 [00:41<00:00, 1.04batch/s]
- targeted  : clean_acc=0.8148 robust_acc=0.4688 sec=51.8504
convnext_tiny_BIM: 100%
40/40 [00:55<00:00, 1.38s/batch]
BIM         : clean_acc=0.8148 robust_acc=0.0076 sec=55.8343
convnext_tiny_BIM: 100%
40/40 [01:25<00:00, 1.53s/batch]
- targeted  : clean_acc=0.8148 robust_acc=0.2619 sec=141.7860
convnext_tiny_RFGSM: 100%
40/40 [00:54<00:00, 1.35s/batch]
RFGSM       : clean_acc=0.8148 robust_acc=0.0084 sec=54.9663
convnext_tiny_RFGSM: 100%
40/40 [01:23<00:00, 1.51s/batch]
- targeted  : clean_acc=0.8148 robust_acc=0.2361 sec=138.6787
convnext_tiny_PGD: 100%
40/40 [00:54<00:00, 1.35s/batch]
PGD         : clean_acc=0.8148 robust_acc=0.0092 sec=54.7968
convnext_tiny_PGD: 100%
40/40 [01:24<00:00, 1.51s/batch]
- targeted  : clean_acc=0.8148 robust_acc=0.2605 sec=138.8668
convnext_tiny_CosPGD: 100%
40/40 [00:55<00:00, 1.35s/batch]
CosPGD      : clean_acc=0.8148 robust_acc=0.0000 sec=55.0754
convnext_tiny_CosPGD: 100%
40/40 [01:24<00:00, 1.51s/batch]
- targeted  : clean_acc=0.8148 robust_acc=0.0002 sec=139.9976
convnext_tiny_CosPGD_softmax: 100%
40/40 [00:55<00:00, 1.35s/batch]
CosPGD_softmax: clean_acc=0.8148 robust_acc=0.1258 sec=55.0348
convnext_tiny_CosPGD_softmax: 100%
40/40 [01:24<00:00, 1.52s/batch]
- targeted  : clean_acc=0.8148 robust_acc=0.6654 sec=140.0198
convnext_tiny_EOTPGD: 100%
40/40 [01:43<00:00, 1.87s/batch]
EOTPGD      : clean_acc=0.8148 robust_acc=0.0098 sec=103.9321
convnext_tiny_EOTPGD: 100%
40/40 [02:13<00:00, 2.41s/batch]
- targeted  : clean_acc=0.8148 robust_acc=0.2598 sec=237.5333
convnext_tiny_FFGSM: 100%
40/40 [00:10<00:00, 4.66batch/s]
FFGSM       : clean_acc=0.8148 robust_acc=0.3174 sec=10.7467
convnext_tiny_FFGSM: 100%
40/40 [00:39<00:00, 1.03batch/s]
- targeted  : clean_acc=0.8148 robust_acc=0.4801 sec=50.7458
convnext_tiny_TPGD: 100%
40/40 [00:57<00:00, 1.42s/batch]
TPGD        : clean_acc=0.8148 robust_acc=0.3322 sec=57.6089
convnext_tiny_MIFGSM: 100%
40/40 [00:55<00:00, 1.36s/batch]
MIFGSM      : clean_acc=0.8148 robust_acc=0.0000 sec=55.5994
convnext_tiny_MIFGSM: 100%
40/40 [01:24<00:00, 1.51s/batch]
- targeted  : clean_acc=0.8148 robust_acc=0.0023 sec=140.4215
convnext_tiny_UPGD: 100%
40/40 [00:55<00:00, 1.36s/batch]
UPGD        : clean_acc=0.8148 robust_acc=0.0000 sec=55.3579
convnext_tiny_UPGD: 100%
40/40 [01:25<00:00, 1.52s/batch]
- targeted  : clean_acc=0.8148 robust_acc=0.0043 sec=140.6881
convnext_tiny_APGD: 100%
40/40 [00:56<00:00, 1.02s/batch]
APGD        : clean_acc=0.8148 robust_acc=0.0002 sec=56.7230
convnext_tiny_APGDT: 0%
0/40 [00:02<?, ?batch/s]
APGDT        occurs Error
CUDA out of memory. Tried to allocate 130.00 MiB (GPU 0; 23.64 GiB total capacity; 22.19 GiB already allocated; 79.06 MiB free; 23.01 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
convnext_tiny_DIFGSM: 100%
40/40 [00:55<00:00, 1.36s/batch]
DIFGSM      : clean_acc=0.8148 robust_acc=0.0633 sec=55.4727
convnext_tiny_DIFGSM: 100%
40/40 [01:25<00:00, 1.52s/batch]
- targeted  : clean_acc=0.8148 robust_acc=0.4109 sec=140.7293
convnext_tiny_TIFGSM: 100%
40/40 [00:56<00:00, 1.39s/batch]
TIFGSM      : clean_acc=0.8148 robust_acc=0.2102 sec=56.3835
convnext_tiny_TIFGSM: 100%
40/40 [01:26<00:00, 1.54s/batch]
- targeted  : clean_acc=0.8148 robust_acc=0.6459 sec=142.5710
convnext_tiny_Jitter: 100%
40/40 [00:55<00:00, 1.01batch/s]
Jitter      : clean_acc=0.8148 robust_acc=0.0256 sec=55.8918
convnext_tiny_Jitter: 100%
40/40 [01:25<00:00, 1.52s/batch]
- targeted  : clean_acc=0.8148 robust_acc=0.3260 sec=141.2696
convnext_tiny_NIFGSM: 100%
40/40 [00:55<00:00, 1.36s/batch]
NIFGSM      : clean_acc=0.8148 robust_acc=0.3152 sec=55.3828
convnext_tiny_NIFGSM: 100%
40/40 [01:24<00:00, 1.52s/batch]
- targeted  : clean_acc=0.8148 robust_acc=0.6609 sec=139.8601
convnext_tiny_PGDRS: 0%
0/40 [00:02<?, ?batch/s]
PGDRS        occurs Error
CUDA out of memory. Tried to allocate 2.87 GiB (GPU 0; 23.64 GiB total capacity; 22.38 GiB already allocated; 82.81 MiB free; 23.00 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
convnext_tiny_SINIFGSM: 100%
40/40 [04:11<00:00, 4.59s/batch]
SINIFGSM    : clean_acc=0.8148 robust_acc=0.3480 sec=251.4323
convnext_tiny_SINIFGSM: 100%
40/40 [04:40<00:00, 5.13s/batch]
- targeted  : clean_acc=0.8148 robust_acc=0.6621 sec=532.1589
convnext_tiny_VMIFGSM: 100%
40/40 [05:00<00:00, 5.49s/batch]
VMIFGSM     : clean_acc=0.8148 robust_acc=0.0000 sec=300.6647
convnext_tiny_VMIFGSM: 100%
40/40 [05:29<00:00, 6.04s/batch]
- targeted  : clean_acc=0.8148 robust_acc=0.0043 sec=630.5667
convnext_tiny_VNIFGSM: 100%
40/40 [05:00<00:00, 5.49s/batch]
VNIFGSM     : clean_acc=0.8148 robust_acc=0.0008 sec=300.8512
convnext_tiny_VNIFGSM: 100%
40/40 [05:30<00:00, 6.03s/batch]
- targeted  : clean_acc=0.8148 robust_acc=0.0615 sec=631.2215
convnext_tiny_SPSA: 100%
40/40 [12:11<00:00, 13.38s/batch]
SPSA        : clean_acc=0.8148 robust_acc=0.7227 sec=731.2949
convnext_tiny_SPSA: 100%
40/40 [12:10<00:00, 13.35s/batch]
- targeted  : clean_acc=0.8148 robust_acc=0.8807 sec=1461.3532
convnext_tiny_JSMA: 0%
0/40 [00:10<?, ?batch/s]
JSMA         occurs Error
CUDA out of memory. Tried to allocate 84.41 GiB (GPU 0; 23.64 GiB total capacity; 1.04 GiB already allocated; 21.33 GiB free; 1.75 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
convnext_tiny_EADL1: 0%
0/40 [00:02<?, ?batch/s]
EADL1        occurs Error
CUDA out of memory. Tried to allocate 148.00 MiB (GPU 0; 23.64 GiB total capacity; 22.86 GiB already allocated; 116.81 MiB free; 22.97 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
convnext_tiny_EADEN: 0%
0/40 [00:02<?, ?batch/s]
EADEN        occurs Error
CUDA out of memory. Tried to allocate 148.00 MiB (GPU 0; 23.64 GiB total capacity; 22.86 GiB already allocated; 78.81 MiB free; 23.00 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
convnext_tiny_CW: 100%
40/40 [07:39<00:00, 8.81s/batch]
CW          : clean_acc=0.8148 robust_acc=0.0000 sec=459.0034
convnext_tiny_CW: 100%
40/40 [09:53<00:00, 11.00s/batch]
- targeted  : clean_acc=0.8148 robust_acc=0.1539 sec=1052.5946
convnext_tiny_PGDL2: 100%
40/40 [00:55<00:00, 1.35s/batch]
PGDL2       : clean_acc=0.8148 robust_acc=0.0070 sec=55.3150
convnext_tiny_PGDL2: 100%
40/40 [01:25<00:00, 1.53s/batch]
- targeted  : clean_acc=0.8148 robust_acc=0.1631 sec=140.6575
"""

