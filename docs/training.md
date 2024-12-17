# Training
Here we explain how to train your own networks for semantic and offset prediction using the automatically segmented point clouds introduced in the paper. Training the network on custom forest point clouds works analogously.

*1\) Download training/validation point clouds and pretrained model weights*
```
python tree_learn/util/download.py --dataset_name automatically_segmented_data --root_folder data/train/forests
```
```
python tree_learn/util/download.py --dataset_name benchmark_dataset --root_folder data/val/forest
```
* Download the pretrained model weights provided by [SoftGroup](https://drive.google.com/file/d/1FABsCUnxfO_VlItAzDYAwurdfcdK-scs/view?usp=sharing). Save the file to ``data/model_weights/hais_ckpt_spconv2.pth``.

*2\) Generate training crops for semantic and offset prediction*
* The forest point clouds from which to generate training data must fulfil the same requirement as in the [segmentation pipeline](segmentation_pipeline.md).
* Additionally, the point clouds must contain individual tree and semantic labels. We recommend you to provide the labels as part of .las or .laz files, in which case you need to adhere to the labeling scheme proposed by [this paper](https://doi.org/10.48550/arXiv.2309.01279). See also the readme of our [dataset](https://doi.org/10.25625/VPMPID).
* Alternatively, you can provide the point clouds as .npy or .txt files where the first three columns are the x, y and z coordinates and the last column is the label. In this case, unclassified points should be labeled as -1, non-tree points should be labeled as 0, and trees should be labeled starting from 1. Unclassified points are ignored during training.
* To generate random crops from the forest point clouds, run the following command. Please note that generating 25000 random crops as training data takes up a large amount of space (~700 Gb). You can adjust the number of crops to be generated in the configuration file.
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
python tools/training/train.py --config configs/training/train.yaml
```