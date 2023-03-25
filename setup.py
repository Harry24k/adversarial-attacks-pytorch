import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='torchattacks',
    version='3.3.0',
    author='Harry Kim',
    author_email='24K.Harry@gmail.com',
    description='Torchattacks is a PyTorch library that provides adversarial attacks to generate adversarial examples.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Harry24k/adversarial-attacks-pytorch',
    packages=setuptools.find_packages(),
    install_requires=[
        'torch>=1.7.1', 'torchvision>=0.8.2', 'scipy>=0.14.0', 'tqdm~=4.56.1',
        'requests~=2.25.1', 'numpy>=1.19.4',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    include_package_data=True,
)