import argparse
import dataclasses
import json
import math
import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Union

import requests
# import timm
import torch
from torch import nn

from robustbench.model_zoo import model_dicts as all_models
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel


ACC_FIELDS = {
    ThreatModel.corruptions: "corruptions_acc",
    ThreatModel.L2: ("external", "autoattack_acc"),
    ThreatModel.Linf: ("external", "autoattack_acc")
}


def download_gdrive(gdrive_id, fname_save):
    """ source: https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url """
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, fname_save):
        CHUNK_SIZE = 32768

        with open(fname_save, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    print('Download started: path={} (gdrive_id={})'.format(
        fname_save, gdrive_id))

    url_base = "https://docs.google.com/uc?export=download&confirm=t"
    session = requests.Session()

    response = session.get(url_base, params={'id': gdrive_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': gdrive_id, 'confirm': token}
        response = session.get(url_base, params=params, stream=True)

    save_response_content(response, fname_save)
    session.close()
    print('Download finished: path={} (gdrive_id={})'.format(
        fname_save, gdrive_id))


def rm_substr_from_state_dict(state_dict, substr):
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        if substr in key:  # to delete prefix 'module.' if it exists
            new_key = key[len(substr):]
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict


def add_substr_to_state_dict(state_dict, substr):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[substr + k] = v
    return new_state_dict


def load_model(model_name: str,
               model_dir: Union[str, Path] = './models',
               dataset: Union[str,
                              BenchmarkDataset] = BenchmarkDataset.cifar_10,
               threat_model: Union[str, ThreatModel] = ThreatModel.Linf,
               norm: Optional[str] = None) -> nn.Module:
    """Loads a model from the model_zoo.

     The model is trained on the given ``dataset``, for the given ``threat_model``.

    :param model_name: The name used in the model zoo.
    :param model_dir: The base directory where the models are saved.
    :param dataset: The dataset on which the model is trained.
    :param threat_model: The threat model for which the model is trained.
    :param norm: Deprecated argument that can be used in place of ``threat_model``. If specified, it
      overrides ``threat_model``

    :return: A ready-to-used trained model.
    """

    # if model_name in timm.list_models():
    #     return timm.create_model(model_name, pretrained=True).eval()

    dataset_: BenchmarkDataset = BenchmarkDataset(dataset)
    if norm is None:
        threat_model_: ThreatModel = ThreatModel(threat_model)
    else:
        threat_model_ = ThreatModel(norm)
        warnings.warn(
            "`norm` has been deprecated and will be removed in a future version.",
            DeprecationWarning)

    model_dir_ = Path(model_dir) / dataset_.value / threat_model_.value
    model_path = model_dir_ / f'{model_name}.pt'

    models = all_models[dataset_][threat_model_]
    
    # if models[model_name]['gdrive_id'] is None:
    #     raise ValueError(f"Model `{model_name}` is not a timm model and has no `gdrive_id` specified.")

    if not isinstance(models[model_name]['gdrive_id'], list):
        model = models[model_name]['model']()
        if dataset_ == BenchmarkDataset.imagenet and 'Standard' in model_name:
            return model.eval()
        
        if not os.path.exists(model_dir_):
            os.makedirs(model_dir_)
        if not os.path.isfile(model_path):
            download_gdrive(models[model_name]['gdrive_id'], model_path)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        if 'Kireev2021Effectiveness' in model_name or model_name == 'Andriushchenko2020Understanding':
            checkpoint = checkpoint['last']  # we take the last model (choices: 'last', 'best')
        try:
            # needed for the model of `Carmon2019Unlabeled`
            state_dict = rm_substr_from_state_dict(checkpoint['state_dict'],
                                                   'module.')
            # needed for the model of `Chen2020Efficient`
            state_dict = rm_substr_from_state_dict(state_dict,
                                                   'model.')
        except:
            state_dict = rm_substr_from_state_dict(checkpoint, 'module.')
            state_dict = rm_substr_from_state_dict(state_dict, 'model.')

        if dataset_ == BenchmarkDataset.imagenet:
            # so far all models need input normalization, which is added as extra layer
            state_dict = add_substr_to_state_dict(state_dict, 'model.')
        
        model = _safe_load_state_dict(model, model_name, state_dict, dataset_)

        return model.eval()

    # If we have an ensemble of models (e.g., Chen2020Adversarial, Diffenderfer2021Winning_LRR_CARD_Deck)
    else:
        model = models[model_name]['model']()
        if not os.path.exists(model_dir_):
            os.makedirs(model_dir_)
        for i, gid in enumerate(models[model_name]['gdrive_id']):
            if not os.path.isfile('{}_m{}.pt'.format(model_path, i)):
                download_gdrive(gid, '{}_m{}.pt'.format(model_path, i))
            checkpoint = torch.load('{}_m{}.pt'.format(model_path, i),
                                    map_location=torch.device('cpu'))
            try:
                state_dict = rm_substr_from_state_dict(
                    checkpoint['state_dict'], 'module.')
            except KeyError:
                state_dict = rm_substr_from_state_dict(checkpoint, 'module.')

            model.models[i] = _safe_load_state_dict(model.models[i],
                                                    model_name, state_dict,
                                                    dataset_)
            model.models[i].eval()

        return model.eval()


def _safe_load_state_dict(model: nn.Module, model_name: str,
                          state_dict: Dict[str, torch.Tensor],
                          dataset_: BenchmarkDataset) -> nn.Module:
    known_failing_models = {
        "Andriushchenko2020Understanding", "Augustin2020Adversarial",
        "Engstrom2019Robustness", "Pang2020Boosting", "Rice2020Overfitting",
        "Rony2019Decoupling", "Wong2020Fast", "Hendrycks2020AugMix_WRN",
        "Hendrycks2020AugMix_ResNeXt", "Kireev2021Effectiveness_Gauss50percent",
        "Kireev2021Effectiveness_AugMixNoJSD", "Kireev2021Effectiveness_RLAT",
        "Kireev2021Effectiveness_RLATAugMixNoJSD", "Kireev2021Effectiveness_RLATAugMixNoJSD",
        "Kireev2021Effectiveness_RLATAugMix", "Chen2020Efficient",
        "Wu2020Adversarial", "Augustin2020Adversarial_34_10",
        "Augustin2020Adversarial_34_10_extra", "Diffenderfer2021Winning_LRR",
        "Diffenderfer2021Winning_LRR_CARD_Deck", "Diffenderfer2021Winning_Binary",
        "Diffenderfer2021Winning_Binary_CARD_Deck"
    }

    failure_messages = ['Missing key(s) in state_dict: "mu", "sigma".',
                        'Unexpected key(s) in state_dict: "model_preact_hl1.1.weight"',
                        'Missing key(s) in state_dict: "normalize.mean", "normalize.std"',
                        'Unexpected key(s) in state_dict: "conv1.scores"']

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        if (model_name in known_failing_models or dataset_ == BenchmarkDataset.imagenet
            ) and any([msg in str(e) for msg in failure_messages]):
            model.load_state_dict(state_dict, strict=False)
        else:
            raise e

    return model


def clean_accuracy(model: nn.Module,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   batch_size: int = 100,
                   device: torch.device = None):
    if device is None:
        device = x.device
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size].to(device)

            output = model(x_curr)
            acc += (output.max(1)[1] == y_curr).float().sum()

    return acc.item() / x.shape[0]


def get_key(x, keys):
    if isinstance(keys, str):
        return float(x[keys])
    else:
        for k in keys:
            if k in x.keys():
                return float(x[k])


def list_available_models(
        dataset: Union[str, BenchmarkDataset] = BenchmarkDataset.cifar_10,
        threat_model: Union[str, ThreatModel] = ThreatModel.Linf,
        norm: Optional[str] = None):
    dataset_: BenchmarkDataset = BenchmarkDataset(dataset)

    if norm is None:
        threat_model_: ThreatModel = ThreatModel(threat_model)
    else:
        threat_model_ = ThreatModel(norm)
        warnings.warn(
            "`norm` has been deprecated and will be removed in a future version.",
            DeprecationWarning)

    models = all_models[dataset_][threat_model_].keys()

    acc_field = ACC_FIELDS[threat_model_]

    json_dicts = []

    jsons_dir = Path("./model_info") / dataset_.value / threat_model_.value

    for model_name in models:
        json_path = jsons_dir / f"{model_name}.json"

        # Some models might not yet be in model_info
        if not json_path.exists():
            continue

        with open(json_path, 'r') as model_info:
            json_dict = json.load(model_info)

        json_dict['model_name'] = model_name
        json_dict['venue'] = 'Unpublished' if json_dict[
            'venue'] == '' else json_dict['venue']
        if isinstance(acc_field, str):
            json_dict[acc_field] = float(json_dict[acc_field]) / 100
        else:
            for k in acc_field:
                if k in json_dict.keys():
                    json_dict[k] = float(json_dict[k]) / 100
        json_dict['clean_acc'] = float(json_dict['clean_acc']) / 100
        json_dicts.append(json_dict)

    json_dicts = sorted(json_dicts, key=lambda d: -get_key(d, acc_field))
    print('| <sub>#</sub> | <sub>Model ID</sub> | <sub>Paper</sub> | <sub>Clean accuracy</sub> | <sub>Robust accuracy</sub> | <sub>Architecture</sub> | <sub>Venue</sub> |')
    print('|:---:|---|---|:---:|:---:|:---:|:---:|')
    for i, json_dict in enumerate(json_dicts):
        if json_dict['model_name'] == 'Chen2020Adversarial':
            json_dict['architecture'] = json_dict[
                'architecture'] + ' <br/> (3x ensemble)'
        if json_dict['model_name'] != 'Natural':
            print(
                '| <sub>**{}**</sub> | <sub><sup>**{}**</sup></sub> | <sub>*[{}]({})*</sub> | <sub>{:.2%}</sub> | <sub>{:.2%}</sub> | <sub>{}</sub> | <sub>{}</sub> |'
                .format(i + 1, json_dict['model_name'], json_dict['name'],
                        json_dict['link'], json_dict['clean_acc'],
                        get_key(json_dict, acc_field), json_dict['architecture'],
                        json_dict['venue']))
        else:
            print(
                '| <sub>**{}**</sub> | <sub><sup>**{}**</sup></sub> | <sub>*{}*</sub> | <sub>{:.2%}</sub> | <sub>{:.2%}</sub> | <sub>{}</sub> | <sub>{}</sub> |'
                .format(i + 1, json_dict['model_name'], json_dict['name'],
                        json_dict['clean_acc'], get_key(json_dict, acc_field),
                        json_dict['architecture'], json_dict['venue']))


def _get_bibtex_entry(model_name: str, title: str, authors: str, venue: str, year: int):
    authors = authors.replace(', ', ' and ')
    return (f"@article{{{model_name},\n"
            f"\ttitle\t= {{{title}}},\n"
            f"\tauthor\t= {{{authors}}},\n"
            f"\tjournal\t= {{{venue}}},\n"
            f"\tyear\t= {{{year}}}\n"
            "}\n")


def get_leaderboard_bibtex(dataset: Union[str, BenchmarkDataset], threat_model: Union[str, ThreatModel]):
    dataset_: BenchmarkDataset = BenchmarkDataset(dataset)
    threat_model_: ThreatModel = ThreatModel(threat_model)

    jsons_dir = Path("./model_info") / dataset_.value / threat_model_.value

    bibtex_entries = set()

    for json_path in jsons_dir.glob("*.json"):

        model_name = json_path.stem.split("_")[0]

        with open(json_path, 'r') as model_info:
            model_dict = json.load(model_info)
            title = model_dict["name"]
            authors = model_dict["authors"]
            full_venue = model_dict["venue"]
            if full_venue == "N/A":
                continue
            venue = full_venue.split(" ")[0]
            venue = venue.split(",")[0]

            year = model_dict["venue"].split(" ")[-1]

            bibtex_entry = _get_bibtex_entry(
                model_name, title, authors, venue, year)
            bibtex_entries.add(bibtex_entry)

    str_entries = ''
    for entry in bibtex_entries:
        print(entry)
        str_entries += entry

    return bibtex_entries, str_entries


def get_leaderboard_latex(dataset: Union[str, BenchmarkDataset],
                          threat_model: Union[str, ThreatModel],
                          l_keys=['clean_acc', 'external', #'autoattack_acc',
                                  'additional_data', 'architecture', 'venue',
                                  'modelzoo_id'],
                          sort_by='external' #'autoattack_acc'
                          ):
    dataset_: BenchmarkDataset = BenchmarkDataset(dataset)
    threat_model_: ThreatModel = ThreatModel(threat_model)

    models = all_models[dataset_][threat_model_]
    print(models.keys())
    
    jsons_dir = Path("./model_info") / dataset_.value / threat_model_.value
    entries = []

    for json_path in jsons_dir.glob("*.json"):
        if not json_path.stem.startswith('Standard'):
            model_name = json_path.stem.split("_")[0]
        else:
            model_name = json_path.stem
        
        with open(json_path, 'r') as model_info:
            model_dict = json.load(model_info)

        str_curr = '\\citet{{{}}}'.format(model_name) if not model_name in ['Standard', 'Standard_R50'] \
            else model_name.replace('_', '\\_')

        for k in l_keys:
            if k == 'external' and not 'external' in model_dict.keys():
                model_dict[k] = model_dict['autoattack_acc']
            if k == 'additional_data':
                v = 'Y' if model_dict[k] else 'N'
            elif k == 'architecture':
                v = model_dict[k].replace('WideResNet', 'WRN')
                v = v.replace('ResNet', 'RN')
            elif k == 'modelzoo_id':
                # print(json_path.stem)
                v = json_path.stem.split('.json')[0]
                if not v in models.keys():
                    v = 'N/A'
                else:
                    v = v.replace('_', '\\_')
            else:
                v = model_dict[k]
            str_curr += ' & {}'.format(v)
        str_curr += '\\\\'
        entries.append((str_curr, float(model_dict[sort_by])))

    entries = sorted(entries, key=lambda k: k[1], reverse=True)
    entries = ['{} &'.format(i + 1) + a for i, (a, b) in enumerate(entries)]
    entries = '\n'.join(entries).replace('<br>', ' ')

    return entries


def update_json(dataset: BenchmarkDataset, threat_model: ThreatModel,
                model_name: str, accuracy: float, adv_accuracy: float,
                eps: Optional[float]) -> None:
    json_path = Path(
        "model_info"
    ) / dataset.value / threat_model.value / f"{model_name}.json"
    if not json_path.parent.exists():
        json_path.parent.mkdir(parents=True, exist_ok=True)

    acc_field = ACC_FIELDS[threat_model]
    if isinstance(acc_field, tuple):
        acc_field = acc_field[-1]

    acc_field_kwarg = {acc_field: adv_accuracy}

    model_info = ModelInfo(dataset=dataset.value, eps=eps, clean_acc=accuracy, **acc_field_kwarg)

    with open(json_path, "w") as f:
        f.write(json.dumps(dataclasses.asdict(model_info), indent=2))


@dataclasses.dataclass
class ModelInfo:
    link: Optional[str] = None
    name: Optional[str] = None
    authors: Optional[str] = None
    additional_data: Optional[bool] = None
    number_forward_passes: Optional[int] = None
    dataset: Optional[str] = None
    venue: Optional[str] = None
    architecture: Optional[str] = None
    eps: Optional[float] = None
    clean_acc: Optional[float] = None
    reported: Optional[float] = None
    corruptions_acc: Optional[str] = None
    autoattack_acc: Optional[str] = None
    footnote: Optional[str] = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        default='Carmon2019Unlabeled')
    parser.add_argument('--threat_model',
                        type=str,
                        default='Linf',
                        choices=[x.value for x in ThreatModel])
    parser.add_argument('--dataset',
                        type=str,
                        default='cifar10',
                        choices=[x.value for x in BenchmarkDataset])
    parser.add_argument('--eps', type=float, default=8 / 255)
    parser.add_argument('--n_ex',
                        type=int,
                        default=100,
                        help='number of examples to evaluate on')
    parser.add_argument('--batch_size',
                        type=int,
                        default=500,
                        help='batch size for evaluation')
    parser.add_argument('--data_dir',
                        type=str,
                        default='./data',
                        help='where to store downloaded datasets')
    parser.add_argument('--model_dir',
                        type=str,
                        default='./models',
                        help='where to store downloaded models')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='random seed')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='device to use for computations')
    parser.add_argument('--to_disk', type=bool, default=True)
    args = parser.parse_args()
    return args
