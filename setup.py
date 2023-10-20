import setuptools

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setuptools.setup(
    name = 'torchattacks',
    version = '3.5.1',
    description='Torchattacks is a PyTorch library that provides adversarial attacks to generate adversarial examples.',
    author = 'Harry Kim',
    author_email='24k.harry@gmail.com',
    packages = setuptools.find_packages(),
    keyword = ['deeplearning', 'adversarial', 'attack', 'pytorch', 'torch',
               'rpgd', 'eotpgd', 'pgd', 'fgsm', 'cw', 'rfgsm', 'ifgsm', 'iterll',
               'bim', 'stepll', 'deepfool', 'trades', 'fast', 'mifgsm',
               'dlr', 'apgd', 'fab', 'square', 'autoattack', 'difgsm', 'pixle'
              ],
    install_requires=[
        'torch>=1.7.1', 'torchvision>=0.8.2', 'scipy>=0.14.0', 'tqdm>=4.56.1',
        'requests~=2.25.1', 'numpy>=1.19.4',
    ],
    python_requires = '>=3',
    zip_safe = False,
    license="MIT",
    url = 'https://github.com/HarryK24/adversairal-attacks-pytorch',
    
    classifiers = [
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 5 - Production/Stable',

    # Indicate who your project is intended for
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Visualization',

    # Pick your license as you wish (should match "license" above)
    'License :: OSI Approved :: MIT License',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 3.6',

    'Operating System :: OS Independent',
    ],
    include_package_data=True,
)