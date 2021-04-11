
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

  * **Methods for ``Attack`` are added**:
    * `set_attack_mode`: To set attack mode to `targeted` (Use input labels as targeted labels) or `least likely` (Use least likely labels as targeted labels), `set_attack_mode` is added.
      * `StepLL` is merged to `BIM`. Please use `set_attack_mode(mode='least_likely')`.
      * However, there are several methods that can not be changed by `set_attack_mode` such as `Deepfool`
    * `set_return_type`: Instead of `set_mode`, now `set_return_type` will be the method to change the return type of adversarial images.



### v2.6

  * ``MIFGSM``:
    * https://github.com/Harry24k/adversarial-attacks-pytorch/pull/10




### v2.9

  * ``VANILA``:
    * Vanila version of _torch.Attack_.
  * ``MultiAttack``:
    * MultiAttack does not need a model as an input. It automatically get models from given attacks.
    * Demo added.
  * ``Attack.set_attack_mode``:
    * For the targeted mode, target_map_function is required.




### v2.10

  * ``GN``:
    * Add guassian noise with given sigma.




### v2.10.3

  * ``TPGD``: Faster computation




### v2.10.4

  * ``attacks`` : To preserve the original gradient status of images, all attacks uses ``images.clone().detach()`` instead of `images`.




### v2.11.0

  * ``CW``
      * Now it outputs the best L2 adversarial images.
      * Faster computation.
  * `DeepFool`
      * Make the codes cleaner.
  * `BIM`
      * Bug fixed: Wrong cliping.
  * `MIFGSM`
      * Bug fixed: Wrong cliping.
      * Bug fixed: [Gradient Norm](https://github.com/Harry24k/adversarial-attacks-pytorch/issues/12).
  * Demo Added
      * Performance Comparison (CIFAR10)



### v2.12.1

  * `DeepFool`
      * Deprecated.
  * `Attack._targeted`
      * ._targeted is set to 1 when targeted mode is activated. [Issue](https://github.com/Harry24k/adversarial-attacks-pytorch/issues/14).
          * All attacks supporting targeted mode is change.
  * `Attack.set_attack_mode`
      * To provide various attack mode, it uses following methods.
          * `set_default_mode`: default mode.
          * `set_targeted_mode`: targeted mode. Now supporting `target_map_function=None` for pre-generated targeted labels.
          * `set_least_likely_mode`: least likely targeted mode. Now supporting k-th smallest probability targeted mode by `kth_min`.
  * `Attack.save`
      * Bug fixed: When `verbose=True`, it now use model.eval() and torch.no_grad().



### v2.12.1

  * `DeepFool`
      * Deprecated.
  * `Attack._targeted`
      * ._targeted is set to 1 when targeted mode is activated. [Issue](https://github.com/Harry24k/adversarial-attacks-pytorch/issues/14).
          * All attacks supporting targeted mode is change.
  * `Attack.set_attack_mode`
      * To provide various attack mode, it uses following methods.
          * `set_default_mode`: default mode.
          * `set_targeted_mode`: targeted mode. Now supporting `target_map_function=None` for pre-generated targeted labels.
          * `set_least_likely_mode`: least likely targeted mode. Now supporting k-th smallest probability targeted mode by `kth_min`.
  * `Attack.save`
      * Bug fixed: When `verbose=True`, it now use model.eval() and torch.no_grad().



### v2.12.2

  * `PGDL2`
      * PGD with L2 distance measure.
  * `Attack.save`
      * Print L2 distance between adversarial examples and the original examples.



### v2.12.3

  * `PGDL2`
    * Initialization perturbation is changed.
    
    
    
### v2.13.1

  * `Attack.set_attack_mode`
      * Deprecated. Use following built-in functions.
          * `set_mode_default`: default mode.
          * `set_mode_targeted`: targeted mode. Now supporting `target_map_function=None` for pre-generated targeted labels.
          * `set_mode_least_likely`: least likely targeted mode. Now supporting k-th smallest probability targeted mode by `kth_min`.
  * `APGD` is changed to `EOTPGD`.
  * `PGDDLR` is added.
  * `APGD`, `APGDT`, `Square`, `FAB`
      * Modified from https://github.com/fra31/auto-attack.
          * `n_iters` is changed to `steps`.
          * `n_target_classes` is calculated based on `n_claases`.
          * `reduce=False` is erased because it is enough with `reduction='none'`.
  * `AutoAttack`
      * Created based on `APGD`, `APGDT`, `Square`, `FAB`.



### v2.13.2

  * `Attack.save`
      * Don't use an additional memory if `save_path=None`



### v2.14.0

  * `DeepFool`, `OnePixel`, `SparseFool` are added.