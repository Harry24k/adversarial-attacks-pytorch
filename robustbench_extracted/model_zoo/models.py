from collections import OrderedDict
from typing import Any, Dict, Dict as OrderedDictType

from robustbench.model_zoo.cifar10 import cifar_10_models
from robustbench.model_zoo.cifar100 import cifar_100_models
from robustbench.model_zoo.imagenet import imagenet_models
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel

ModelsDict = OrderedDictType[str, Dict[str, Any]]
ThreatModelsDict = OrderedDictType[ThreatModel, ModelsDict]
BenchmarkDict = OrderedDictType[BenchmarkDataset, ThreatModelsDict]

model_dicts: BenchmarkDict = OrderedDict([
    (BenchmarkDataset.cifar_10, cifar_10_models),
    (BenchmarkDataset.cifar_100, cifar_100_models),
    (BenchmarkDataset.imagenet, imagenet_models)
])
