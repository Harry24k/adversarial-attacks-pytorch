import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Union

from jinja2 import Environment, PackageLoader, select_autoescape

from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from robustbench.utils import ACC_FIELDS


def generate_leaderboard(dataset: Union[str, BenchmarkDataset],
                         threat_model: Union[str, ThreatModel],
                         models_folder: str = "model_info") -> str:
    """Prints the HTML leaderboard starting from the .json results.

    The result is a <table> that can be put directly into the RobustBench index.html page,
    and looks the same as the tables that are already existing.

    The .json results must have the same structure as the following:
    ``
    {
      "link": "https://arxiv.org/abs/2003.09461",
      "name": "Adversarial Robustness on In- and Out-Distribution Improves Explainability",
      "authors": "Maximilian Augustin, Alexander Meinke, Matthias Hein",
      "additional_data": true,
      "number_forward_passes": 1,
      "dataset": "cifar10",
      "venue": "ECCV 2020",
      "architecture": "ResNet-50",
      "eps": "0.5",
      "clean_acc": "91.08",
      "reported": "73.27",
      "autoattack_acc": "72.91"
    }
    ``

    If the model is robust to common corruptions, then the "autoattack_acc" field should be
    "corruptions_acc".

    :param dataset: The dataset of the wanted leaderboard.
    :param threat_model: The threat model of the wanted leaderboard.
    :param models_folder: The base folder of the model jsons (e.g. our "model_info" folder).

    :return: The resulting HTML table.
    """
    dataset_: BenchmarkDataset = BenchmarkDataset(dataset)
    threat_model_: ThreatModel = ThreatModel(threat_model)

    folder = Path(models_folder) / dataset_.value / threat_model_.value

    acc_field = ACC_FIELDS[threat_model_]

    models = []
    for model_path in folder.glob("*.json"):
        with open(model_path) as fp:
            model = json.load(fp)

        models.append(model)

    #models.sort(key=lambda x: x[acc_field], reverse=True)
    def get_key(x):
        if isinstance(acc_field, str):
            return float(x[acc_field])
        else:
            for k in acc_field:
                if k in x.keys():
                    return float(x[k])
    models.sort(key=get_key, reverse=True)

    env = Environment(loader=PackageLoader('robustbench', 'leaderboard'),
                      autoescape=select_autoescape(['html', 'xml']))

    template = env.get_template('leaderboard.html.j2')

    result = template.render(threat_model=threat_model, dataset=dataset,
        models=models, acc_field=acc_field if isinstance(acc_field, str) else acc_field[-1])
    print(result)
    return result


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="The dataset of the desired leaderboard."
    )
    parser.add_argument(
        "--threat_model",
        type=str,
        help="The threat model of the desired leaderboard."
    )
    parser.add_argument(
        "--models_folder",
        type=str,
        default="model_info",
        help="The base folder of the model jsons (e.g. our 'model_info' folder)"
    )
    args = parser.parse_args()

    generate_leaderboard(args.dataset, args.threat_model, args.models_folder)
