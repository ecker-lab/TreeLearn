# TreeLearn: A Comprehensive Deep Learning Method for Segmenting Individual Trees from Forest Point Cloud

![Architecture](./method.png)

The article is available from [arXiv](https://arxiv.org/abs/2309.08471).

Laser-scanned point clouds of forests make it possible to extract valuable information for forest management. To consider single trees, a forest point cloud needs to be segmented into individual tree point clouds. 
Existing segmentation methods are usually based on hand-crafted algorithms, such as identifying trunks and growing trees from them, and face difficulties in dense forests with overlapping tree crowns. In this study, we propose TreeLearn, a deep learning-based approach for semantic and instance segmentation of forest point clouds. Unlike previous methods, TreeLearn is trained on already segmented point clouds in a data-driven manner, making it less reliant on predefined features and algorithms. 
Additionally, we introduce a new manually segmented benchmark forest dataset containing 156 full trees, and 79 partial trees, that have been cleanly segmented by hand. This enables the evaluation of instance segmentation performance going beyond just evaluating the detection of individual trees.
We trained TreeLearn on forest point clouds of 6665 trees, labeled using the Lidar360 software. An evaluation on the benchmark dataset shows that TreeLearn performs equally well or better than the algorithm used to generate its training data. Furthermore, the method's performance can be vastly improved by fine-tuning on the cleanly labeled benchmark dataset. 

For a quick demo of the capabilities of TreeLearn without any manual setup, we prepared a google colab notebook: 

<a target="_blank" href="https://colab.research.google.com/github/ecker-lab/TreeLearn/blob/main/TreeLearn_Pipeline.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Limitations

Please note that our models have been trained on tls/mls data of forests dominated by beech. Initial results for point clouds obtained from other forest types or laser scanning methods suggest that the segmentation performance decreases substantially in this case. We expect that for a good performance on e.g. uav data and other forest types, finetuning the models is necessary. We are currently working towards including more powerful models that have been trained on a broader data basis.

## Setup

To set up the environment we recommend Conda. If Conda is set up and activated, run the following:

```
source setup/setup.sh
```

Depending on the specific gpu and cuda version of your system, you might need to adjust the spconv version specified in ``setup/requirements.txt``.

## Data

The dataset as well as trained models can be found at [this url](https://doi.org/10.25625/VPMPID). 
To download the data, we recommend using the script ``tree_learn/util/download.py``. Here, we list out the commands to download the data in either the npz or the las format:

| Data        | Download                                             | 
| ----------- | :----------------------------------------------------------- |
| Benchmark dataset (npz)   | ```python tree_learn/util/download.py --dataset_name benchmark_dataset_npz --root_folder data/benchmark_dataset``` | 
| Benchmark dataset (las)  | ```python tree_learn/util/download.py --dataset_name benchmark_dataset_las --root_folder data/benchmark_dataset``` | 
| Automatically segmented data (npz)   | ```python tree_learn/util/download.py --dataset_name automatically_segmented_data_npz --root_folder data/automatically_segmented``` | 
| Automatically segmented data (las)   | ```python tree_learn/util/download.py --dataset_name automatically_segmented_data_las --root_folder data/automatically_segmented``` |
| Model checkpoints   | ```python tree_learn/util/download.py --dataset_name checkpoints --root_folder data/checkpoints``` | 
| Extra files   | ```python tree_learn/util/download.py --dataset_name extra --root_folder data/extra``` | 13 GB        |

<!-- Please refer to [setup guide](docs/setup.md) -->
<!-- Please refer to [pipeline guide](docs/tools/pipeline.md) -->


## Segmentation pipeline

To begin with, it should be noted that all functionality in this repository can be configured using the config files located in the folder ``configs``.
We do not enforce a specific folder structure with regard to where data (e.g. point clouds or pre-trained models) is stored. You can save them where it is most suitable for you.
To ensure that the functionality of this repository works without errors, all config arguments pertaining to data paths must be changed to conform with where you store your data.
Next we explain how to obtain segmentation results of a forest point cloud into understory and trees. You need to perform the following three steps:

*1\) Download pre-trained models*
* Follow the instructions given above to obtain the pre-trained models.

*2\) Prepare forest point cloud to be segmented*
* The forest point cloud must be provided either as a npy file (contains numpy array) or a space-delimited txt file.
* The data must consist of N rows and three columns where N is the number of points in the point cloud and the columns are the x, y and z coordinates of the forest.
* The coordinates must be provided in meter scale and have a minimum resolution of one point per (0.1 m)<sup>3</sup>.
* Ground and understory points must still be part of the point cloud. Only rough noise filtering has to be performed in advance (e.g. to remove scanned particles in the air).
* The point cloud file must be placed in a folder 'forests' located in another folder that constitutes the base directory containing all pipeline-related output: ``pipeline_output/forests/your_filename.npy``
* We recommend retaining an edge around the point cloud that is of interest. E.g. for an area of interest of 100 m x 100 m, retain an edge of ~10 m to each side so that input is 120 m x 120 m.
* The pipeline automatically removes the edge which is only needed as context for network prediction. The xy-shape of the point cloud does not have to be square. Arbitrary shapes are allowed.

*3\) Run segmentation pipeline*
* To execute the segmentation pipeline, run the following command:
```
python tools/pipeline/pipeline.py --config configs/pipeline/pipeline.yaml
```


## Custom training

Here we explain how to train your own networks for semantic and offset prediction as well as fragment classification. You need to perform the following five steps:

*1\) Generate training crops for semantic and offset prediction:*
* The forest point clouds from which to generate training data must fulfil the same criteria as for the segmentation pipeline.
* However, the data consists of four columns instead of three. The last column consists of labels.
* Trees must be labeled as positive integers. Understory points must be labeled as 9999 and points to ignore during training must be labeled as -100.
* The labeled forest point clouds must be placed in a folder 'forests' that is located in another folder that constitutes the base directory for the generation of training data:
``training_data/forests/labeled_forest1.npy, training_data/forests/labeled_forest2.npy, ...``
* To generate random crops from the forest point clouds, run the following command:
```
python tools/train_data_gen/gen_pointwise_train_data.py --config configs/train_data_gen/pointwise_train_data_gen.yaml
```

*2\) Generate validation data for semantic and offset prediction:*
* The forest point cloud used to generate validation data must fulfil the same properties as for the training data.
* The folder structure is also the same: ``validation_data/forests/validation_forest.npy``
* To generate tiles used for validation, run the following command:
```
python tools/train_data_gen/gen_pointwise_val_data.py --config configs/train_data_gen/pointwise_val_data_gen.yaml
```
* If you want to use multiple forest point clouds for validation, repeat this procedure.

*3\) Train the network for semantic and offset prediction with the following command:*
```
python tools/training/train.py --config configs/training/train_pointwise.yaml
```

*4\) Generate training data for the classifier:*
* This step requires a fully trained network for semantic and offset prediction which is used to generate tree predictions.
* The tree predictions are categorized as valid trees or fragments based on their intersection over union with the ground truth segmentations.
* To generate tree point clouds with classification labels, run the following command:
```
python tools/train_data_gen/gen_cls_train_data.py --config_pipeline configs/pipeline/pipeline.yaml --config_cls_data_gen configs/train_data_gen/cls_train_data_gen.yaml
```

*5\) Train the network for classification with the following command:*
```
python tools/training/train.py --config configs/training/train_classifier.yaml
```


## Evaluation on benchmark dataset

* To evaluate the performance of an arbitrary segmentation method on the benchmark dataset, run the following command:
```
python tools/evaluation/evaluate.py --config configs/evaluation/evaluate.yaml
```
* To take a look at the evaluation results, we prepared a notebook that can be found at ``tools/evaluation/evaluation.ipynb``


## Acknowledgements

The code is built based on [SoftGroup](https://github.com/thangvubk/SoftGroup) and [spconv](https://github.com/traveller59/spconv).
