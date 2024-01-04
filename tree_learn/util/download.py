# PARTIALLY TAKEN FROM https://github.com/pdebench/PDEBench (MIT LICENSE)

import argparse
from torchvision.datasets.utils import download_url
import os	

BASE_PATH = "https://data.goettingen-research-online.de/api/access/datafile/:persistentId?persistentId=doi:10.25625/VPMPID/"

class automatically_segmented_data:
    files = [
        ["VZ2ENJ", "A1N.laz"],
        ["4BA1BD", "A1W.laz"],
        ["MOFUYC", "G1N.laz"],
        ["I0M17S", "G1W.laz"],
        ["GQAIGP", "G2N.laz"],
        ["XHACRG", "G2W.laz"],
        ["K8UOPU", "G3N.laz"],
        ["KT8EB4", "G3W.laz"],
        ["0FIQFD", "G4N.laz"],
        ["XXWS3Z", "G4W.laz"],
        ["8ZOLYB", "L1N.laz"],
        ["2OU297", "L2N.laz"],
        ["YGJKLJ", "L2W.laz"],
        ["3WBA7S", "LG1.laz"],
        ["OBJTIG", "LG2.laz"],
        ["EDG9IB", "LG3.laz"],
        ["UXBEDS", "O1N.laz"],
        ["1IE8CP", "O1W.laz"],
    ]


class benchmark_dataset:
    files = [
        ["QXXJ2I", "L1W.laz"]
    ]


class benchmark_dataset_voxelized:
    files = [
        ["AQM6VO", "L1W_voxelized01.laz"]
    ]


class model_weights_diverse_training_data:
    files = [
        ["1JMEQV", "model_weights_diverse_training_data.pth"]
    ]


class evaluated_trees:
    files = [
        ["WTIB7F", "evaluated_trees.txt"]
    ]


def get_ids(name):
    datasets = {
        "automatically_segmented_data": automatically_segmented_data,
        "benchmark_dataset": benchmark_dataset,
        "benchmark_dataset_voxelized": benchmark_dataset_voxelized,
        "model_weights_diverse_training_data": model_weights_diverse_training_data,
        "evaluated_trees": evaluated_trees
    }
    
    dataset = datasets.get(name)
    if dataset is not None:
        return dataset.files
    else:
        raise NotImplementedError (f"Dataset {name} does not exist.")


def download_data(root_folder, dataset_name):
    """ "
    Download data splits specific to a given setting.

    Args:
    root_folder: The root folder where the data will be downloaded
    dataset_name: The name of the dataset to download, must be defined in this python file.  """

    print(f"Downloading data for {dataset_name} ...")

    # Load and parse metadata csv file
    files = get_ids(dataset_name)
    os.makedirs(root_folder, exist_ok=True)

    # Iterate ids and download the files
    for id, name in files:
        url = BASE_PATH + id
        download_url(url, root_folder, name)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        prog="Download Script",
        description="Helper script to download the TreeLearn data",
        epilog="",
    )

    arg_parser.add_argument(
        "--root_folder",
        type=str,
        required=True,
        help="Root folder where the data will be downloaded",
    )
    arg_parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset setting to download",
    )

    args = arg_parser.parse_args()

    download_data(args.root_folder, args.dataset_name)