# TreeLearn: A Comprehensive Deep Learning Method for Segmenting Individual Trees from Forest Point Cloud

Laser-scanned point clouds of forests make it possible to extract valuable information for forest management. To consider single trees, a forest point cloud needs to be segmented into individual tree point clouds. 
Existing segmentation methods are usually based on hand-crafted algorithms, such as identifying trunks and growing trees from them, and face difficulties in dense forests with overlapping tree crowns. In this study, we propose \mbox{TreeLearn}, a deep learning-based approach for semantic and instance segmentation of forest point clouds. Unlike previous methods, TreeLearn is trained on already segmented point clouds in a data-driven manner, making it less reliant on predefined features and algorithms. 
Additionally, we introduce a new manually segmented benchmark forest dataset containing 156 full trees, and 79 partial trees, that have been cleanly segmented by hand. This enables the evaluation of instance segmentation performance going beyond just evaluating the detection of individual trees.
We trained TreeLearn on forest point clouds of 6665 trees, labeled using the Lidar360 software. An evaluation on the benchmark dataset shows that TreeLearn performs equally well or better than the algorithm used to generate its training data. Furthermore, the method's performance can be vastly improved by fine-tuning on the cleanly labeled benchmark dataset. 

This repository is in a preliminary stage. Additional documentation will be added soon.

## Data

The dataset as well as trained models can be found at [this url](https://doi.org/10.25625/VPMPID).

## Usage

### Setup

To set up the environment we recommend Conda. If Conda is set up and activated run the following:

```
source setup/setup.sh
```

### Generate segmentation results for new plot
```
python tools/pipeline/pipeline.py --config configs/pipeline/pipeline.yaml --work_dir path/to/dir
```


### Train Backbone
```
python tools/training/train.py --config configs/training/train_pointwise.yaml --work_dir path/to/dir
```


### Train Classifier
```
python tools/training/train.py --config configs/training/train_classifier_50e.yaml --work_dir path/to/dir
```

### Evaluate training results on benchmark dataset
```
python tools/evaluation/evaluate.py --config configs/evaluation/tree_learn/evaluate_treelearn.yaml --work_dir path/to/dir
```