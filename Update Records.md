
### ~~v0.3~~
* ~~**New Attacks** : FGSM, IFGSM, IterLL, RFGSM, CW(L2), PGD are added.~~
* ~~**Demos** are uploaded.~~

### ~~v0.4~~
* ~~**DO NOT USE** : 'init.py' is omitted.~~

### ~~v0.5~~
* ~~**Package name changed** : 'attacks' is changed to 'torchattacks'.~~
* ~~**New Attack** : APGD is added.~~
* ~~**attack.py** : 'update_model' method is added.~~

### ~~v0.6~~
* ~~**Error Solved** :~~ 
    * ~~Before this version, even after getting an adversarial image, the model remains evaluation mode.~~
    * ~~To solve this, below methods are modified.~~
        * ~~'_switch_model' method is added into **attack.py**. It will automatically change model mode to the previous mode after getting adversarial images. When getting adversarial images, model is switched to evaluation mode.~~
        * ~~'__call__' methods in all attack changed to forward. Instead of this, '__call__' method is added into 'attack.py'~~
* ~~**attack.py** : To provide ease of changing images to uint8 from float, 'set_mode' and '_to_uint' is added.~~
    * ~~'set_mode' determines returning all outputs as 'int' OR 'flaot' through '_to_uint'.~~
    * ~~'_to_uint' changes all outputs into uint8.~~

### ~~v0.7~~
* ~~**All attacks are modified**~~
    * ~~clone().detach() is used instead of .data~~
    * ~~torch.autograd.grad is used instead of .backward() and .grad :~~
        * ~~It showed 2% reduction of computation time.~~
    
### ~~v0.8~~
* ~~**New Attack** : RPGD is added.~~
* ~~**attack.py** : 'update_model' method is depreciated. Because torch models are passed by call-by-reference, we don't need to update models.~~
    * ~~**cw.py** : In the process of cw attack, now masked_select uses a mask with dtype torch.bool instead of a mask with dtype torch.uint8.~~

### ~~v0.9~~
* ~~**New Attack** : DeepFool is added.~~
* ~~**Some attacks are renamed** :~~
    * ~~I-FGSM -> BIM~~
    * ~~IterLL -> StepLL~~

### ~~v1.0~~
* ~~**attack.py** :~~
    * ~~**load** : Load is depreciated. Instead, use TensorDataset and DataLoader.~~
    * ~~**save** : The problem of calculating invalid accuracy when the mode of the attack set to 'int' is solved.~~

### ~~v1.1~~
* ~~**DeepFool** :~~
    * ~~[**Error solved**](https://github.com/Harry24k/adversairal-attacks-pytorch/issues/2).~~
    
### ~~v1.2~~
* ~~**Description has been added for each module.**~~
* ~~**Sphinx Document uploaded**~~ 
* ~~**attack.py** : 'device' will be decided by [next(model.parameters()).device](https://github.com/Harry24k/adversarial-attacks-pytorch/issues/3#issue-602571865).~~
* ~~**Two attacks are merged** :~~
    * ~~RPGD, PGD -> PGD~~

### ~v1.2 (DON'T USE)

  * **Pip packages were corrupted by accumulating previous versions.**



### v1.3

  * **Pip Package Re-uploaded.**



### v1.4

  * `PGD`:
    * Now PGD supports targeted mode.



### v1.5

  * `MultiAttack`:
    * MultiAttack is added.
    * With it, you can use PGD with N-random-restarts or stronger attacks with different methods.



### v2.4

  * **`steps` instead of `iters`**:
    * For compatibility reasons, all `iters` are changed to `steps`.
  * `TPGD`:
    * PGD (Linf) based on KL-Divergence loss proposed by [Hongyang Zhang et al.](https://arxiv.org/abs/1901.08573) is added.
  * `FFGSM`:
    * New FGSM proposed by [Eric Wong et al.](https://arxiv.org/abs/2001.03994) is added.



### v2.5

  * **Common methods are added**:
    * `set_attack_mode`: To set attack mode to `targeted` (Use input labels as targeted labels) or `least likely` (Use least likely labels as targeted labels), `set_attack_mode` is added.
      * `StepLL` is merged to `BIM`. Please use `set_attack_mode(mode='least_likely')`.
      * However, there are several methods that can not be changed by `set_attack_mode` such as `Deepfool`
    * `set_return_type`: Instead of `set_mode`, now `set_return_type` will be the method to change the return type of adversarial images.



### v2.6

  * ``MIFGSM``:
    * https://github.com/Harry24k/adversarial-attacks-pytorch/pull/10




### v2.7

  * ``VANILA``:
    * Vanila version of torch.Attack
  * **set_attack_mode**:
    * For the targeted mode, target_map_function is required.
  * **MutliAttack**:
    * MutliAttack does not need a model as an input. It automatically get models from given attacks.
    * Demo added.