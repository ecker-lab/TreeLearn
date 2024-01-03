# TreeLearn: A Comprehensive Deep Learning Method for Segmenting Individual Trees from Forest Point Cloud

![Architecture](./method.png)

The article is available from [arXiv](https://arxiv.org/abs/2309.08471).

Laser-scanned point clouds of forests make it possible to extract valuable information for forest management. To consider single trees, a forest point cloud needs to be segmented into individual tree point clouds. 
Existing segmentation methods are usually based on hand-crafted algorithms, such as identifying trunks and growing trees from them, and face difficulties in dense forests with overlapping tree crowns. In this study, we propose TreeLearn, a deep learning-based approach for tree instance segmentation of forest point clouds. Unlike previous methods, TreeLearn is trained on already segmented point clouds in a data-driven manner, making it less reliant on predefined features and algorithms. Furthermore, TreeLearn is implemented as a fully automatic pipeline and does not rely on extensive hyperparameter tuning, which makes it easy to use. Additionally, we introduce a new manually segmented benchmark forest dataset containing 156 full trees, and 79 partial trees, that have been cleanly segmented by hand. This is an important step towards creating a large and diverse data basis for model development and fine-grained instance segmentation evaluation. We trained TreeLearn on forest point clouds of 6665 trees, labeled using the Lidar360 software. An evaluation on the benchmark dataset shows that TreeLearn performs equally well or better than the algorithm used to generate its training data. Furthermore, the method's performance can be vastly improved by fine-tuning on the cleanly labeled benchmark dataset. The TreeLearn code is availabe from https://github.com/ecker-lab/TreeLearn. The data as well as trained models can be found at https://doi.org/10.25625/VPMPID.

For a quick demo of the capabilities of TreeLearn without any manual setup, we prepared a google colab notebook: 

<a target="_blank" href="https://colab.research.google.com/github/ecker-lab/TreeLearn/blob/main/TreeLearn_Pipeline.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Setup

To set up the environment we recommend Conda. If Conda is set up and activated, run the following:

```
source setup/setup.sh
```

Depending on the specific gpu and cuda version of your system, you might need to adjust the spconv version specified in ``setup/requirements.txt``.

<!--
## Data

The dataset as well as trained models can be found at [this url](https://doi.org/10.25625/VPMPID). 
To download the data, we recommend using the script ``tree_learn/util/download.py``. Here, we list out the commands to download the data in either the npz or the las format:

| Data        | Download                                             | 
| ----------- | :----------------------------------------------------------- |
| Benchmark dataset (npz)   | ```python tree_learn/util/download.py --dataset_name benchmark_dataset_npz --root_folder data/benchmark``` | 
| Benchmark dataset (las)  | ```python tree_learn/util/download.py --dataset_name benchmark_dataset_las --root_folder data/benchmark``` | 
| Automatically segmented data (npz)   | ```python tree_learn/util/download.py --dataset_name automatically_segmented_data_npz --root_folder data/train/forests``` | 
| Automatically segmented data (las)   | ```python tree_learn/util/download.py --dataset_name automatically_segmented_data_las --root_folder data/train/forests_las``` |
| Model checkpoints   | ```python tree_learn/util/download.py --dataset_name checkpoints --root_folder data/checkpoints``` | 
| Extra files   | ```python tree_learn/util/download.py --dataset_name extra --root_folder data/extra``` | 13 GB        |
-->

## Segmentation pipeline

In the following, we explain how to run the segmentation pipeline on our benchmark dataset L1W. Running the segmentation pipeline on a custom forest point cloud works analogously. You can change the configuration of running the pipeline by editing the configuration file located at ``configs/pipeline/pipeline.yaml`. However, the default configuration is most likely adequate in the majority of cases.

*1\) Download pre-trained models and L1W forest point cloud*
* TODO: GIVE CONCRETE COMMAND FOR GETTING PRE-TRAINED MODELS
* TODO: GIVE CONCRETE COMMAND FOR GETTING BENCHMARK POINT CLOUD

*2\) Prepare forest point cloud to be segmented* (This is already fulfilled for L1W)
* The forest point cloud must be provided as a las, laz, npy, npz or a space-delimited txt file. 
* The coordinates must be provided in meter scale and have a minimum resolution of one point per (0.1 m)<sup>3</sup>.
* It is especially important that the trunks of the trees have a sufficiently high resolution. This requirement might not be fulfilled for point clouds obtained via airborne laser scanning.
* Ground and understory points must still be part of the point cloud. Only rough noise filtering has to be performed in advance (e.g. to remove scanned particles in the air). See L1W as an example.
* The point cloud file must be placed in ``data/pipeline/L1W/forest``
* Change the argument 'forest_path' in the pipeline configuration at ``configs/pipeline/pipeline.yaml`` to ``data/pipeline/L1W/forest/L1W.laz``
* We strongly recommend retaining a buffer around the point cloud that is of interest. E.g. for an area of interest of 100 m x 100 m, retain a buffer of ~13.5 m to each side so that input is 127 m x 127 m.
* The pipeline automatically removes the buffer which is only needed as context for network prediction. The xy-shape of the point cloud does not have to be square. Arbitrary shapes are allowed.

*3\) Run segmentation pipeline*
* To execute the segmentation pipeline, run the following command:
```
python tools/pipeline/pipeline.py --config configs/pipeline/pipeline.yaml
```


## Custom training

Here we explain how to train your own networks for semantic and offset prediction using the automatically segmented point clouds introduced in the paper. Training the network on custom forest point clouds works analogously.

*1\) Download point clouds for training and validation*
* TODO: GIVE CONCRETE COMMAND FOR GETTING NOISY LABELS DATA AND ALSO UNZIP IT
* TODO: GIVE CONCRETE COMMAND FOR GETTING BENCHMARK POINT CLOUD
* TODO: WE ALSO NEED TO SOMEHOW GET HAIS CHECKPOINT IN SOME WAY I WOULD SAY.

*2\) Generate training crops for semantic and offset prediction:*
* The forest point clouds from which to generate training data must fulfil the same resolution and noise filtering requirement as in the segmentation pipeline.
* Additionally, the point clouds must contain individual tree and semantic labels. We recommend you to provide the labels as part of .las or .laz files, in which case you need to adhere to the labeling scheme proposed by [this paper](https://doi.org/10.48550/arXiv.2309.01279). See also the description of our [dataset](https://doi.org/10.25625/VPMPID).
* To generate random crops from the forest point clouds, run the following command:
```
python tools/data_gen/gen_train_data.py --config configs/data_gen/gen_train_data.yaml
```

*3\) Generate validation data for semantic and offset prediction:*
* The forest point cloud used to generate validation data must fulfil the same properties as for the training data.
* To generate tiles used for validation, run the following command:
```
python tools/data_gen/gen_val_data.py --config configs/data_gen/gen_val_data.yaml
```

*4\) Train the network for semantic and offset prediction with the following command:*
```
python tools/training/train.py --config configs/training/train_pointwise.yaml
```


## Evaluation on benchmark dataset

* To evaluate the performance of an arbitrary segmentation method on the benchmark dataset, run the following command:
```
python tools/evaluation/evaluate_benchmark.py --config configs/evaluation/evaluate_benchmark.yaml
```
* To take a look at the evaluation results, we prepared a notebook that can be found at ``tools/evaluation/evaluate_benchmark.ipynb``


## Acknowledgements

The code is built based on [SoftGroup](https://github.com/thangvubk/SoftGroup) and [spconv](https://github.com/traveller59/spconv).
