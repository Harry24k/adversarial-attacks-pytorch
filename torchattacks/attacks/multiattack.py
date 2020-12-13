import torch

from ..attack import Attack


class MultiAttack(Attack):
    r"""
    MultiAttack is a class to attack a model with various attacks agains same images and labels.

    Arguments:
        model (nn.Module): model to attack.
        attacks (list): list of attacks.
          
    Examples::
        >>> attack1 = torchattacks.PGD(model, eps=4/255, alpha=8/255, iters=40, random_start=False)
        >>> attack2 = torchattacks.PGD(model, eps=4/255, alpha=8/255, iters=40, random_start=False)
        >>> attack = torchattacks.MultiAttack(model, [attack1, attack2])
        >>> adv_images = attack(images, labels)
        
    """
    def __init__(self, attacks):

        # Check validity
        ids = []
        for attack in attacks:
            ids.append(id(attack.model))

        if len(set(ids)) != 1:
            raise ValueError("At least one of attacks is referencing a different model.")
            
        super(MultiAttack, self).__init__("MultiAttack", attack.model)
        self.attacks = attacks
        self._attack_mode = 'only_default'

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        fails = torch.arange(images.shape[0]).to(self.device)
        final_images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        for i, attack in enumerate(self.attacks):
            adv_images = attack(images[fails], labels[fails])

            outputs = self.model(adv_images)
            _, pre = torch.max(outputs.data, 1)
            
            corrects = (pre == labels[fails])
            wrongs = ~corrects

            succeeds = torch.masked_select(fails, wrongs)
            succeeds_of_fails = torch.masked_select(torch.arange(fails.shape[0]).to(self.device), wrongs)

            final_images[succeeds] = adv_images[succeeds_of_fails]

            fails = torch.masked_select(fails, corrects)

            if len(fails) == 0:
                break

        return final_images
