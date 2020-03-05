import torch

class Attack(object):
    """
    An abstract class representing attacks.

    Arguments:
        name (string): name of the attack.
        model (nn.Module): a model to attack.

    .. note:: device("cpu" or "cuda") will be automatically determined by a given model.
    
    """
    def __init__(self, name, model):
        self.attack = name
        
        self.model = model
        self.model_name = str(model).split("(")[0]
        self.training = model.training
        self.device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
        self.mode = 'float'
                
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
        
        if self.mode == 'int' :
            images = self._to_uint(images)
            
        return images
    
    def _to_uint(self, images):
        return (images*255).type(torch.uint8)
    
    # It changes model to the original eval/train.
    def _switch_model(self):
        if self.training :
            self.model.train()
        else :
            self.model.eval()
    
    # It Defines the computation performed at every call.
    # Should be overridden by all subclasses.
    def forward(self, *input):
        raise NotImplementedError
    
    # Determine return all adversarial images as 'int' OR 'float'.
    def set_mode(self, mode):
        if mode == 'float' :
            self.mode = 'float'
        elif mode == 'int' :
            self.mode = 'int'
        else :
            raise ValueError(mode + " is not valid")
    
    # DEPRECIATED : update model is not necessary because torch model is called by reference.
    '''
    # Update the model to be used
    def update_model(self, model) :
        self.model = model
        self.training = model.training
    '''
        
    # Save image data as torch tensor from data_loader.
    # If you don't want to know about accuaracy of the model, set accuracy as False.
    def save(self, file_name, data_loader, accuracy=True):
        
        self.model.eval()
        
        image_list = []
        label_list = []
        
        correct = 0
        total = 0
        
        total_batch = len(data_loader)
        
        for step, (images, labels) in enumerate(data_loader) :
            adv_images = self.__call__(images, labels)
          
            image_list.append(adv_images.cpu())
            label_list.append(labels.cpu())
            
            if self.mode == 'int' :
                adv_images = adv_images.float()/255
            
            if accuracy :
                outputs = self.model(adv_images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(self.device)).sum()

            print('- Save Progress : %2.2f %%        ' %((step+1)/total_batch*100), end='\r')
        
        if accuracy :
            acc = 100 * float(correct) / total
            print('\n- Accuracy of the model : %2.2f %%' % (acc), end='')
        
        x = torch.cat(image_list, 0)
        y = torch.cat(label_list, 0)
        torch.save((x, y), file_name)
        print('\n- Save Complete!')
        
        self._switch_model()
        
    # DEPRECIATED  
    '''
    # Load image data as torch dataset.
    # When scale=True it automatically tansforms images to [0, 1].
    def load(self, file_name, scale = True) :
        adv_images, adv_labels = torch.load(file_name)
        
        if scale :
            adv_data = torch.utils.data.TensorDataset(adv_images.float() / adv_images.max(), adv_labels)
        else :
            adv_data = torch.utils.data.TensorDataset(adv_images.float(), adv_labels)
            
        return adv_data
    '''
        
    # DEPRECIATED : eval is merged to save.
    '''
    # Evaluate accuaracy of a model
    # With default 'model = None', it will return accuracy of white box attack
    # If not, it will return accuracy of black box attack with self.model as holdout model
    
    def eval(self, data_loader, model = None) :
        
        if model is None :
            model = self.model
        else :
            model = model.eval()

        correct = 0
        total = 0

        total_batch = len(data_loader)
        
        for step, (images, labels) in enumerate(data_loader) :

            adv_images = self.__call__(images, labels)
            outputs = model(adv_images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels.to(self.device)).sum()
            
            print('- Evaluation Progress : %2.2f %%        ' %((step+1)/total_batch*100), end='\r')

        accuracy = 100 * float(correct) / total
        print('\n- Accuracy of model : %f %%' % (accuracy))

        return accuracy
    '''