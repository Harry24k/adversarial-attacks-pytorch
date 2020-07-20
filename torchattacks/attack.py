import torch


class Attack(object):
    r"""
    Base class for all attacks.

    .. note::
        It automatically set device to the device where given model is.
        It temporarily changes the original model's `training mode` to `test`
        by `.eval()` only during an attack process.
    """
    def __init__(self, name, model):
        r"""
        Initializes internal Attack state.

        Arguments:
            name (str) : name of attack.
            model (nn.Module): model to attack.
        """

        self.attack = name
        self.model = model
        self.model_name = str(model).split("(")[0]

        self.training = model.training
        self.device = next(model.parameters()).device

        self.mode = 'float'

    # It defines the computation performed at every call.
    # Should be overridden by all subclasses.
    def forward(self, *input):
        r"""
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    # Determine return all adversarial images as 'int' OR 'float'.
    def set_mode(self, mode):
        r"""
        Set whether return adversarial images as `int` or `float`.

        Arguments:
            mode (str) : 'float' or 'int'. (DEFAULT : 'float')

        """
        if mode == 'float':
            self.mode = 'float'
        elif mode == 'int':
            self.mode = 'int'
        else:
            raise ValueError(mode + " is not valid")

    # Save image data as torch tensor from data_loader.
    def save(self, file_name, data_loader, accuracy=True):
        r"""
        Save adversarial images as torch.tensor from data_loader.

        Arguments:
            file_name (str) : save path.
            data_loader (torch.utils.data.DataLoader) : dataloader.
            accuracy (bool) : If you don't want to know an accuaracy,
                              set accuracy as False. (DEFAULT : True)

        """
        self.model.eval()

        image_list = []
        label_list = []

        correct = 0
        total = 0

        total_batch = len(data_loader)

        for step, (images, labels) in enumerate(data_loader):
            adv_images = self.__call__(images, labels)

            image_list.append(adv_images.cpu())
            label_list.append(labels.cpu())

            if self.mode == 'int':
                adv_images = adv_images.float()/255

            if accuracy:
                outputs = self.model(adv_images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(self.device)).sum()

                acc = 100 * float(correct) / total
                print('- Save Progress : %2.2f %% / Accuracy : %2.2f %%' % ((step+1)/total_batch*100, acc), end='\r')
            else:
                print('- Save Progress : %2.2f %%        ' % ((step+1)/total_batch*100), end='\r')

        x = torch.cat(image_list, 0)
        y = torch.cat(label_list, 0)
        torch.save((x, y), file_name)
        print('\n- Save Complete!')

        self._switch_model()

    # Whole structure of the model will be NOT displayed for print pretty.
    def __str__(self):
        info = self.__dict__.copy()
        del info['model']
        del info['attack']
        return self.attack + "(" + ', '.join('{}={}'.format(key, val) for key, val in info.items()) + ")"

    def __call__(self, *input, **kwargs):
        self.model.eval()
        images = self.forward(*input, **kwargs)
        self._switch_model()

        if self.mode == 'int':
            images = self._to_uint(images)

        return images

    def _to_uint(self, images):
        return (images*255).type(torch.uint8)

    # It changes model to the original eval/train.
    def _switch_model(self):
        if self.training:
            self.model.train()
        else:
            self.model.eval()
