import warnings

import torch

from ..attack import Attack


class MultiAttack(Attack):
    r"""
    MultiAttack is a class to attack a model with various attacks agains same images and labels.

    Arguments:
        model (nn.Module): model to attack.
        attacks (list): list of attacks.
          
    Examples::
        >>> attack1 = torchattacks.PGD(model, eps = 4/255, alpha = 8/255, iters=40, random_start=False)
        >>> attack2 = torchattacks.PGD(model, eps = 4/255, alpha = 8/255, iters=40, random_start=False)
        >>> attack = torchattacks.MultiAttack(model, [attack1, attack2])
        >>> adv_images = attack(images, labels)
        
    """
    def __init__(self, model, attacks):
        super(MultiAttack, self).__init__("MultiAttack", model)
        self.attacks = attacks

        # Check validity
        ids = []
        for attack in attacks:
            ids.append(id(attack.model))

        if len(set(ids)) != 1:
            warnings.warn("At least one of attacks have different model.")

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        fails = torch.arange(images.shape[0]).to(self.device)
        final_images = images.clone().to(self.device)
        labels = labels.to(self.device)

        for i, attack in enumerate(self.attacks):
            # print('- Multi Attack Progress [%d / %d]             ' % ((i+1), len(self.attacks)), end='\r')
            adv_images = attack(images[fails], labels[fails])

            outputs = self.model(adv_images)
            _, pre = torch.max(outputs.data, 1)

            succeeds = torch.masked_select(fails, pre != labels[fails])
            succeeds_of_fails = torch.masked_select(torch.arange(fails.shape[0]).to(self.device), pre != labels[fails])

            final_images[succeeds] = adv_images[succeeds_of_fails]

            fails = torch.masked_select(fails, pre == labels[fails])

            if len(fails) == 0:
                warnings.warn("Ealry Stopped cause all images are successfully perturbed.")
                break

        return final_images
