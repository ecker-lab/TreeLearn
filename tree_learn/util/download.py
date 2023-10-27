# PARTIALLY TAKEN FROM https://github.com/pdebench/PDEBench (MIT LICENSE)

import argparse
from torchvision.datasets.utils import download_url
import os	

BASE_PATH = "https://data.goettingen-research-online.de/api/access/datafile/:persistentId?persistentId=doi:10.25625/VPMPID/"

class automatically_segmented_data_npz:
    files = [
        ["LSZAKN", "A1N.npz"],
        ["L12GKR", "A1W.npz"],
        ["NHKXI5", "G1N.npz"],
        ["7WWGQN", "G1W.npz"],
        ["ARBSN6", "G2N.npz"],
        ["OBWZM7", "G2W.npz"],
        ["MRZYK7", "G3N.npz"],
        ["0HKPXL", "G3W.npz"],
        ["WYA6Q3", "G4N.npz"],
        ["FIN2NN", "G4W.npz"],
        ["83JWGL", "L1N.npz"],
        ["6PUXJX", "L2N.npz"],
        ["GZXXUJ", "L2W.npz"],
        ["2KL2W6", "LG1.npz"],
        ["16FD31", "LG2.npz"],
        ["F1LFV1", "LG3.npz"],
        ["CONPA6", "O1N.npz"],
        ["HUMKWY", "O1W.npz"],
    ]

class automatically_segmented_data_las:
    files = [["YVV2OX", "las_automatically_segmented.zip"]]

class benchmark_dataset_npz:
    files = [
        ["8ISOJQ", "L1W.npz"],
        ["JHJ2EN", "L1W_voxelized01.npz"]
    ]

class benchmark_dataset_las:
    files = [
        ["9QBPIK", "las_L1W.zip"]
    ]

class checkpoints:
    files = [
        ["67HOLF", "checkpoint_classifier.pth"],
        ["WFJLNW", "checkpoint_pointwise_prediction.pth"],
        ["E8Y1IV", "finetuned_checkpoint_classifier.pth"],
        ["JTPZRG", "finetuned_checkpoint_pointwise_prediction.pth"]
    ]

class extra:
    files = [
        ["JE6KOC", "open_files.ipynb"],
        ["UGAQTV", "evaluated_trees.txt"],
        ["PSB6SU", "ReadMe.txt"],
    ]


def get_ids(name):
    datasets = {
        "benchmark_dataset_npz": benchmark_dataset_npz,
        "benchmark_dataset_las": benchmark_dataset_las,
        "checkpoints": checkpoints,
        "automatically_segmented_data_npz": automatically_segmented_data_npz,
        "automatically_segmented_data_las": automatically_segmented_data_las,
        "extra": extra
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
    dataset_name: The name of the dataset to download, one of: reduced_resolution_V5000, reduced_resolution_G5000,
       full_resolution_V5000, full_resolution_G5000 or single_example_G5000  """

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