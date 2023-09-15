# TreeLearn: A Comprehensive Deep Learning Method for Segmenting Individual Trees from MLS and TLS Forest Point Clouds

## Data
available from: to_be_released

## Usage

Set up Environment

```
source setup/setup.sh
```


Train Backbone

```
python tools/training/train.py --config configs/training/train_pointwise.yaml --work_dir save/dir
```



Train Head

```
python tools/training/train.py --config configs/training/train_classifier_50e.yaml --work_dir save/dir
```

Evaluate training results on benchmark dataset

```
python tools/evaluation/evaluate.py --config configs/evaluation/tree_learn/evaluate_treelearn.yaml --work_dir save/dir
```


Generate segmentation results for new plot
''
python tools/pipeline/pipeline.py --config configs/pipeline/pipeline.yaml --work_dir save/dir
''
