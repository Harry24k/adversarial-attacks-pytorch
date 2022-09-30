"""
This file is based on the code from https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py.
"""
import pkg_resources

from torchvision.datasets.vision import VisionDataset

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image

import os
import os.path
import sys


def make_custom_dataset(root, path_imgs, class_to_idx):
    with open(pkg_resources.resource_filename(__name__, path_imgs), 'r') as f:
        fnames = f.readlines()
    images = [(os.path.join(root,
                            c.split('\n')[0]), class_to_idx[c.split('/')[0]])
              for c in fnames]

    return images


class CustomDatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self,
                 root,
                 loader,
                 extensions=None,
                 transform=None,
                 target_transform=None,
                 is_valid_file=None):
        super(CustomDatasetFolder, self).__init__(root)
        self.transform = transform
        self.target_transform = target_transform
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_custom_dataset(
            self.root, 'helper_files/imagenet_test_image_ids.txt',
            class_to_idx)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " +
                                self.root + "\n"
                                "Supported extensions are: " +
                                ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [
                d for d in os.listdir(dir)
                if os.path.isdir(os.path.join(dir, d))
            ]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, path

    def __len__(self):
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class CustomImageFolder(CustomDatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=default_loader,
                 is_valid_file=None):
        super(CustomImageFolder,
              self).__init__(root,
                             loader,
                             IMG_EXTENSIONS if is_valid_file is None else None,
                             transform=transform,
                             target_transform=target_transform,
                             is_valid_file=is_valid_file)

        self.imgs = self.samples


if __name__ == '__main__':
    data_dir = '~/imagenet/val'
    imagenet = CustomImageFolder(
        data_dir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]))

    torch.manual_seed(0)

    test_loader = data.DataLoader(imagenet,
                                  batch_size=5000,
                                  shuffle=True,
                                  num_workers=30)

    x, y, path = next(iter(test_loader))

    with open('path_imgs_2.txt', 'w') as f:
        f.write('\n'.join(path))
        f.flush()
