# Segmentation pipeline
In the following, we explain how to run the segmentation pipeline on our benchmark dataset L1W. Running the segmentation pipeline on a custom forest point cloud works analogously.

*1\) Download L1W forest point cloud and pre-trained model weights*
```
python tree_learn/util/download.py --dataset_name benchmark_dataset --root_folder data/pipeline/L1W/forest
```
* Download pre-trained model weights that are only able to detect trees of at least 10 m height:
```
python tree_learn/util/download.py --dataset_name model_weights_20241213 --root_folder data/model_weights
```
* Download pre-trained model weights that are also able to detect smaller trees:
```
python tree_learn/util/download.py --dataset_name model_weights_with_small_20241213 --root_folder data/model_weights
```
* By default, the pipeline will use the model weights that are only able to detect trees of at least 10 m height.
* If you want to use the model weights that are also able to detect smaller trees, you need to adjust 'pretrain' in the pipeline configuration at ``configs/pipeline/pipeline.yaml`` to ``data/model_weights/model_weights_with_small_20241213.pth``
* It should be noted that the number of errors in the crowns of larger trees increases when many small trees in between are detected (see Figure 6b in our paper for an intuition). This is currently a limitation of the method.

*2\) Prepare forest point cloud to be segmented* (This is already fulfilled for L1W)
* The forest point cloud must be provided as a las, laz, npy, npz or a space-delimited txt file. 
* The coordinates must be provided in meter scale and have a minimum resolution of one point per (0.1 m)<sup>3</sup>. It is especially important that the trunks of the trees have a sufficiently high resolution.
* Terrain and low-vegetation points must still be part of the point cloud. Only rough noise filtering has to be performed in advance (e.g. to remove scanned particles in the air). See L1W as an example.
* The point cloud file must be placed in ``data/pipeline/L1W/forest``
* Change the argument 'forest_path' in the pipeline configuration at ``configs/pipeline/pipeline.yaml`` to ``data/pipeline/L1W/forest/L1W.laz``
* We recommend retaining a buffer around the point cloud that is of interest. E.g. for an area of interest of 100 m x 100 m, retain a buffer of ~13.5 m to each side so that input is 127 m x 127 m.
* Optionally, the pipeline automatically removes the buffer which is only needed as context for network prediction. The xy-shape of the point cloud does not have to be square. Arbitrary shapes are allowed.

*3\) Run segmentation pipeline*
* To execute the segmentation pipeline, run the following command:
```
python tools/pipeline/pipeline.py --config configs/pipeline/pipeline.yaml
```
It should be noted that the results obtained in this guide will be inflated since L1W was included for training the checkpoint. 

## Explanation of some args for running the pipeline
In general, a short explanation of all arguments is given in the config files themselves. In the following, we will explain those arguments (in case they are not self-explaining) that can be adapted by the user depending on their needs:

*1\) Modifying tree detection behavior by modifying ``configs/_modular_/grouping.yaml``*

* Note: By default, the segmentation pipeline now uses HDBSCAN, which does not rely on tau_group, but only on tau_min. Therefore, unless you use DBSCAN, a modification of tau_group is not needed anymore.

* ``grouping: tau_group``: This parameter determines the minimum distance for two points to be considered connected during the clustering of the offset-shifted coordinates. When the resolution of the tree trunks is lower than the expected one point per 0.1 m voxel, the default grouping radius might fail due to tree clusters being too sparse. In that case, you can experiment with increasing tau_group and check if results get better. However, it should be noted that setting tau_group too high will at some point lead to merged trees.

* ``grouping: tau_min``. This parameter determines the minimum number of points required for a valid tree cluster. Points belonging to clusters with less points will be discarded and assigned to nearby trees. Once again, if the resolution of the tree trunks is very low, valid tree clusters might only contain very few points. In that case, you can experiment with decreasing tau_min and see if results improve. However, it should be noted that setting tau_min too low will at some point result in a large number of false positive detections.

*2\) Automatic removal of outer points by modifying ``configs/pipeline/pipeline.yaml``*
* Model predictions at the edge of the forest point cloud will be worse since the model does not have any context for making predictions (e.g. branches without the corresponding trunk in the point cloud). It is possible to remove x meters from the edge of the point cloud automatically. You can determine how much should be removed by setting in ``shape_cfg: outer_remove`` to the desired value in meters. Setting it to "~" performs no removal.
* If the xy shape of your point cloud is convex, it is ok to set ``shape_cfg: alpha`` to 0. If it is concave, you need to increase the value a bit so that an adequate concave hull can be calculated for outer points removal. Empircally, we found 0.6 to be a good value in this case.

## Limitations
* Our method requires a sufficiently high resolution of the tree trunks since it identifies and clusters trunk points for tree detection. This requirement might not be fulfilled for point clouds obtained via ALS. In case that the trunks are captured well enough, our method might also work for point clouds captured via low-flying UAV.
* Although the provided model weights have been trained on diverse data (see readme of our [dataset](https://doi.org/10.25625/VPMPID)), it has not been trained and tested on all forest types, e.g. dense tropical forests. We cannot make any assessment about the performance of TreeLearn in such cases, but expect a drop in performance.
* As research code, our pipeline is not optimized for minimal runtime and resource usage. CPU operations are often not parallelized and some parts of the pipeline, such as tile generation, require high amounts of RAM. For example, processing L1W has a peak RAM usage of almost 100 GB. VRAM requirements are around 10 GB. 
* Due to clustering trees solely based on information of the offset-shifted coordinates, the method can result in odd-looking mistakes. For an in-depth analysis of these errors, we refer to Section 3.4 of the paper.
