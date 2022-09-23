import time
import torch
from torch.utils.data import DataLoader, TensorDataset


class Attack(object):
    r"""
    Base class for all attacks.

    .. note::
        It automatically set device to the device where given model is.
        It basically changes training mode to eval during attack process.
        To change this, please see `set_training_mode`.
    """
    def __init__(self, name, model):
        r"""
        Initializes internal attack state.

        Arguments:
            name (str): name of attack.
            model (torch.nn.Module): model to attack.
        """

        self.attack = name
        self.model = model
        self.model_name = str(model).split("(")[0]
        self.device = next(model.parameters()).device

        self._attack_mode = 'default'
        self._targeted = False
        self._target_map_function = None
        self._return_type = 'float'
        self._supported_mode = ['default']

        self._model_training = False
        self._batchnorm_training = False
        self._dropout_training = False

    def forward(self, *input, **kwargs):
        r"""
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def get_mode(self):
        r"""
        Get attack mode.

        """
        return self._attack_mode

    def set_mode_default(self):
        r"""
        Set attack mode as default mode.

        """
        self._attack_mode = 'default'
        self._targeted = False
        print("Attack mode is changed to 'default.'")

    def _set_mode_targeted(self, mode):
        if "targeted" not in self._supported_mode:
            raise ValueError("Targeted mode is not supported.")
        self._targeted = True
        self._attack_mode = mode
        print("Attack mode is changed to '%s'."%mode)

    def set_mode_targeted_by_function(self, target_map_function):
        r"""
        Set attack mode as targeted.

        Arguments:
            target_map_function (function): Label mapping function.
                e.g. lambda images, labels:(labels+1)%10.
                None for using input labels as targeted labels. (Default)

        """
        self._set_mode_targeted('targeted(custom)')
        self._target_map_function = target_map_function

    def set_mode_targeted_random(self):
        r"""
        Set attack mode as targeted with random labels.
        Arguments:
            num_classses (str): number of classes.

        """
        self._set_mode_targeted('targeted(random)')
        self._target_map_function = self.get_random_target_label

    def set_mode_targeted_least_likely(self, kth_min=1):
        r"""
        Set attack mode as targeted with least likely labels.
        Arguments:
            kth_min (str): label with the k-th smallest probability used as target labels. (Default: 1)

        """
        self._set_mode_targeted('targeted(least-likely)')
        assert (kth_min > 0)
        self._kth_min = kth_min
        self._target_map_function = self.get_least_likely_label

    def set_return_type(self, type):
        r"""
        Set the return type of adversarial images: `int` or `float`.

        Arguments:
            type (str): 'float' or 'int'. (Default: 'float')

        .. note::
            If 'int' is used for the return type, the file size of 
            adversarial images can be reduced (about 1/4 for CIFAR10).
            However, if the attack originally outputs float adversarial images
            (e.g. using small step-size than 1/255), it might reduce the attack
            success rate of the attack.

        """
        if type == 'float':
            self._return_type = 'float'
        elif type == 'int':
            self._return_type = 'int'
        else:
            raise ValueError(type + " is not a valid type. [Options: float, int]")

    def get_return_type(self):
        r"""
        Get the return type of adversarial images: `int` or `float`.
        """
        return self._return_type

    def set_training_mode(self, model_training=False, batchnorm_training=False, dropout_training=False):
        r"""
        Set training mode during attack process.

        Arguments:
            model_training (bool): True for using training mode for the entire model during attack process.
            batchnorm_training (bool): True for using training mode for batchnorms during attack process.
            dropout_training (bool): True for using training mode for dropouts during attack process.

        .. note::
            For RNN-based models, we cannot calculate gradients with eval mode.
            Thus, it should be changed to the training mode during the attack.
        """
        self._model_training = model_training
        self._batchnorm_training = batchnorm_training
        self._dropout_training = dropout_training

    def save(self, data_loader, save_path=None, verbose=True, return_verbose=False,
             save_predictions=False, save_clean_images=False, save_type='float'):
        r"""
        Save adversarial images as torch.tensor from given torch.utils.data.DataLoader.

        Arguments:
            save_path (str): save_path.
            data_loader (torch.utils.data.DataLoader): data loader.
            verbose (bool): True for displaying detailed information. (Default: True)
            return_verbose (bool): True for returning detailed information. (Default: False)
            save_predictions (bool): True for saving predicted labels (Default: False)
            save_clean_images (bool): True for saving clean images (Default: False)

        """
        if save_path is not None:
            adv_image_list = []
            label_list = []
            if save_predictions:
                pred_list = []
            if save_clean_images:
                image_list = []

        correct = 0
        total = 0
        l2_distance = []

        total_batch = len(data_loader)
        given_training = self.model.training

        for step, (images, labels) in enumerate(data_loader):
            start = time.time()
            adv_images = self.__call__(images, labels)
            batch_size = len(images)

            if verbose or return_verbose:
                with torch.no_grad():
                    images_normalized = self.to_type(adv_images, 'float')
                    outputs = self.get_output_with_eval_nograd(images_normalized)

                    # Calculate robust accuracy
                    _, pred = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    right_idx = (pred == labels.to(self.device))
                    correct += right_idx.sum()
                    rob_acc = 100 * float(correct) / total

                    # Calculate l2 distance
                    delta = (images_normalized - images.to(self.device)).view(batch_size, -1)
                    l2_distance.append(torch.norm(delta[~right_idx], p=2, dim=1))
                    l2 = torch.cat(l2_distance).mean().item()

                    # Calculate time computation
                    progress = (step+1)/total_batch*100
                    end = time.time()
                    elapsed_time = end-start

                    if verbose:
                        self._save_print(progress, rob_acc, l2, elapsed_time, end='\r')

            if save_path is not None:
                adv_image_list.append(self.to_type(adv_images.detach().cpu(), save_type))
                label_list.append(labels.detach().cpu())

                adv_image_list_cat = torch.cat(adv_image_list, 0)
                label_list_cat = torch.cat(label_list, 0)
                save_dict = {'adv_images':adv_image_list_cat, 'labels':label_list_cat}

                if save_predictions:
                    pred_list.append(pred.detach().cpu())
                    pred_list_cat = torch.cat(pred_list, 0)
                    save_dict['preds'] = pred_list_cat

                if save_clean_images:
                    image_list.append(self.to_type(images.detach().cpu(), save_type))
                    image_list_cat = torch.cat(image_list, 0)
                    save_dict['clean_images'] = image_list_cat

                save_dict['save_type'] = save_type
                torch.save(save_dict, save_path)

        # To avoid erasing the printed information.
        if verbose:
            self._save_print(progress, rob_acc, l2, elapsed_time, end='\n')

        if given_training:
            self.model.train()

        if return_verbose:
            return rob_acc, l2, elapsed_time

    def _save_print(self, progress, rob_acc, l2, elapsed_time, end):
        print('- Save progress: %2.2f %% / Robust accuracy: %2.2f %% / L2: %1.5f (%2.3f it/s) \t' \
              % (progress, rob_acc, l2, elapsed_time), end=end)

    @staticmethod
    def load(load_path, batch_size=128, shuffle=False,
             load_predictions=False, load_clean_images=False):
        save_dict = torch.load(load_path)
        keys = ['adv_images', 'labels']

        if load_predictions:
            keys.append('preds')
        if load_clean_images:
            keys.append('clean_images')

        if save_dict['save_type'] == 'int':
            save_dict['adv_images'] = save_dict['adv_images'].float()/255
            if load_clean_images:
                save_dict['clean_images'] = save_dict['clean_images'].float()/255

        adv_data = TensorDataset(*[save_dict[key] for key in keys])
        adv_loader = DataLoader(adv_data, batch_size=batch_size, shuffle=shuffle)
        print("Data is loaded in the following order: [%s]"%(", ".join(keys)))
        return adv_loader

    @torch.no_grad()
    def get_output_with_eval_nograd(self, images):
        given_training = self.model.training
        if given_training:
            self.model.eval()
        outputs = self.model(images)
        if given_training:
            self.model.train()
        return outputs

    def get_target_label(self, images, labels=None):
        r"""
        Function for changing the attack mode.
        Return input labels.
        """
        if self._target_map_function is None:
            raise ValueError('target_map_function is not initialized by set_mode_targeted.')
        target_labels = self._target_map_function(images, labels)
        return target_labels

    @torch.no_grad()
    def get_least_likely_label(self, images, labels=None):
        outputs = self.get_output_with_eval_nograd(images)
        if labels is None:
            _, labels = torch.max(outputs, dim=1)
        n_classses = outputs.shape[-1]

        target_labels = torch.zeros_like(labels)
        for counter in range(labels.shape[0]):
            l = list(range(n_classses))
            l.remove(labels[counter])
            _, t = torch.kthvalue(outputs[counter][l], self._kth_min)
            target_labels[counter] = l[t]

        return target_labels.long().to(self.device)

    @torch.no_grad()
    def get_random_target_label(self, images, labels=None):
        outputs = self.get_output_with_eval_nograd(images)
        if labels is None:
            _, labels = torch.max(outputs, dim=1)
        n_classses = outputs.shape[-1]

        target_labels = torch.zeros_like(labels)
        for counter in range(labels.shape[0]):
            l = list(range(n_classses))
            l.remove(labels[counter])
            t = (len(l)*torch.rand([1])).long().to(self.device)
            target_labels[counter] = l[t]

        return target_labels.long().to(self.device)

    @staticmethod
    def to_type(images, type):
        r"""
        Return images as int if float is given.
        """
        if type == 'int':
            if isinstance(images, torch.FloatTensor) or isinstance(images, torch.cuda.FloatTensor):
                return (images*255).type(torch.uint8)
        elif type == 'float':
            if isinstance(images, torch.ByteTensor) or isinstance(images, torch.cuda.ByteTensor):
                return images.float()/255
        else:
            raise ValueError(type + " is not a valid type. [Options: float, int]")
        return images

    def __str__(self):
        info = self.__dict__.copy()

        del_keys = ['model', 'attack']

        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)

        for key in del_keys:
            del info[key]

        info['attack_mode'] = self._attack_mode
        info['return_type'] = self._return_type

        return self.attack + "(" + ', '.join('{}={}'.format(key, val) for key, val in info.items()) + ")"

    def __call__(self, *input, **kwargs):
        given_training = self.model.training

        if self._model_training:
            self.model.train()
            for _, m in self.model.named_modules():
                if not self._batchnorm_training:
                    if 'BatchNorm' in m.__class__.__name__:
                        m = m.eval()
                if not self._dropout_training:
                    if 'Dropout' in m.__class__.__name__:
                        m = m.eval()
        else:
            self.model.eval()

        adv_images = self.forward(*input, **kwargs)

        if given_training:
            self.model.train()

        adv_images = self.to_type(adv_images, self._return_type)

        return adv_images
