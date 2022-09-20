import sys
sys.path.insert(0, '..')


def import_torchattacks():
    import torchattacks
    print(torchattacks.__version__)
    assert True


# def load_model():
#     from robustbench.data import load_cifar10
#     # from robustbench.utils import load_model, clean_accuracy

#     images, labels = load_cifar10(n_examples=10)
#     device = "cpu"
#     assert True
