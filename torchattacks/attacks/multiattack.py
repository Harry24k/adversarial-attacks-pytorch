import time

import torch

from ..attack import Attack


class MultiAttack(Attack):
    r"""
    MultiAttack is a class to attack a model with various attacks agains same images and labels.

    Arguments:
        model (nn.Module): model to attack.
        attacks (list): list of attacks.

    Examples::
        >>> atk1 = torchattacks.PGD(model, eps=8/255, alpha=2/255, iters=40, random_start=True)
        >>> atk2 = torchattacks.PGD(model, eps=8/255, alpha=2/255, iters=40, random_start=True)
        >>> atk = torchattacks.MultiAttack([atk1, atk2])
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, attacks, verbose=False):

        # Check validity
        ids = []
        for attack in attacks:
            ids.append(id(attack.model))

        if len(set(ids)) != 1:
            raise ValueError("At least one of attacks is referencing a different model.")

        super().__init__("MultiAttack", attack.model)
        self.attacks = attacks
        self.verbose = verbose
        self._success_rates = None
        self._supported_mode = ['default']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        fails = torch.arange(images.shape[0]).to(self.device)
        final_images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        success_rates = []

        for _, attack in enumerate(self.attacks):
            adv_images = attack(images[fails], labels[fails])

            outputs = self.model(adv_images)
            _, pre = torch.max(outputs.data, 1)

            corrects = (pre == labels[fails])
            wrongs = ~corrects

            succeeds = torch.masked_select(fails, wrongs)
            succeeds_of_fails = torch.masked_select(torch.arange(fails.shape[0]).to(self.device), wrongs)

            final_images[succeeds] = adv_images[succeeds_of_fails]

            fails = torch.masked_select(fails, corrects)
            success_rates.append(len(fails))

            if len(fails) == 0:
                break

        if self.verbose:
            print("Attack success rate: "+" | ".join(["%2.2f %%"%(sr*100/images.shape[0]) for sr in success_rates]))

        self._update(success_rates)

        return final_images

    def _update(self, success_rates):
        if self._success_rates:
            for i, sr in enumerate(success_rates):
                self._success_rates[i] += sr

    def save(self, data_loader, save_path=None, verbose=True):
        r"""
        Overridden.
        """
        if save_path is not None:
            image_list = []
            label_list = []

        correct = 0
        total = 0
        l2_distance = []
        self._success_rates = []

        for i, attack in enumerate(self.attacks):
            self._success_rates.append(0.0)

        total_batch = len(data_loader)

        training_mode = self.model.training
        for step, (images, labels) in enumerate(data_loader):
            start = time.time()
            adv_images = self.__call__(images, labels)

            batch_size = len(images)

            if save_path is not None:
                image_list.append(adv_images.cpu())
                label_list.append(labels.cpu())

            if self._return_type == 'int':
                adv_images = adv_images.float()/255

            if verbose:
                with torch.no_grad():
                    if training_mode:
                        self.model.eval()
                    outputs = self.model(adv_images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    right_idx = (predicted == labels.to(self.device))
                    correct += right_idx.sum()

                    end = time.time()
                    delta = (adv_images - images.to(self.device)).view(batch_size, -1)
                    l2_distance.append(torch.norm(delta[~right_idx], p=2, dim=1))
                    acc = 100 * float(correct) / total
                    print("- Save progress: %2.2f %% / Accuracy: %2.2f %%"%((step+1)/total_batch*100, acc)+\
                          " / Attack success rate: "+" | ".join(["%2.2f %%"%(sr*100/total) for sr in self._success_rates])+\
                          ' / L2: %1.5f (%2.3f it/s) \t'%(torch.cat(l2_distance).mean(), end-start), end='\r')

        if save_path is not None:
            x = torch.cat(image_list, 0)
            y = torch.cat(label_list, 0)
            torch.save((x, y), save_path)
            print('\n- Save complete!')

        if training_mode:
            self.model.train()

        self._success_rates = None
