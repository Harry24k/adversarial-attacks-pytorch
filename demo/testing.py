import sys
from tqdm.autonotebook import tqdm

import torch
import torch.nn as nn

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24"


sys.path.insert(0, '..')
import torchattacks

sys.path.insert(0, '..')
import robustbench
from robustbench.data import load_cifar10
from robustbench.utils import load_model, clean_accuracy

from torchattacks import PGD, CosPGD
from utils import imshow, get_pred

#images, labels = load_cifar10(n_examples=5)
images, labels = load_cifar10()
print('[Data loaded]')

device = "cuda"
model = load_model('Standard', norm='Linf').to(device)
acc = clean_accuracy(model, images.to(device), labels.to(device))
print('[Model loaded]')
print('Acc: %2.2f %%'%(acc*100))

test_loader = torch.utils.data.DataLoader(
    (images, labels),
    batch_size=128,
    shuffle=False,
    num_workers=128,
    pin_memory=True,
    drop_last=True
)

cospgd_atk = CosPGD(model, eps=8/255, alpha=2/225, steps=10, random_start=True)
print(cospgd_atk)

pgd_atk = PGD(model, eps=8/255, alpha=2/225, steps=10, random_start=True)
print(pgd_atk)

#for i, (images, labels) in tqdm(enumerate(test_loader)):
cospgd_adv_images=None
with tqdm(test_loader, unit="batch") as tepoch:
    for images, labels in tepoch:
        cospgd_adv_images = cospgd_atk(images, labels)
        print(cospgd_adv_images.shape)

idx = 0
pre = get_pred(model, cospgd_adv_images[idx:idx+1], device)
imshow(cospgd_adv_images[idx:idx+1], title="True:%d, Pre:%d"%(labels[idx], pre))
plt.savefig('something.png')