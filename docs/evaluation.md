# Evaluation
The evaluation functionality provided in this repository can be used to obtain evaluation metrics on *arbitrary* forest point clouds obtained by *arbitrary* segmentation methods. For a thorough description of the evaluation metrics that are calculated, we refer to our paper. In the following, we will explain what format exactly is required for the ground truth and predicted forest point cloud to perform the evaluation.

## Ground truth forest point cloud
* Easiest is to provide the ground truth point clouds as las or laz files that are labeled as described in the [this paper](https://doi.org/10.48550/arXiv.2309.01279) (see L1W as an example).
* Alternatively, provide them as npy or space-delimited txt files with following attributes:
    * The first three rows are the x, y and z coordinates in m scale.
    * The last row contains the labels: -1 (non-annotated points), 0 (non-tree points), 1 (tree 1), 2 (tree 2), ...
* Additional information:
    * Non-annotated points do not necessarily have to be included in the ground truth dataset. However, some published forest datasets include it and we think this is a good idea, as we argue below.
    * Non-tree points are ground points, understory vegetation, etc. that are not part of any tree. However, often smaller trees are also labeled as non-tree points.
    * Strictly speaking, ground truth trees need only be labeled with integers != -1 and != 0. As stated above, the simplest would be 1, 2, 3, ... but in principle they do not have to be consecutive, e.g. 1, 3, 4, 5, 7, ... is also ok.

## Predicted forest point cloud
* The predicted point cloud has *exactly* the same formatting requirements as the ground truth point cloud
* Additional information:
    * The predicted forest point cloud is the result of some algorithm and will usually not contain the label "-1".
    * Predictions are *automatically* matched with ground truths based on intersection over union during evaluation. This is why it is *NOT* required that the prediction for a ground truth tree has the same label.
    * Many segmentation algorithms subsample or permute the point cloud, leading to a different ordering or number of points compared to the ground truth point cloud. This is *NOT* a problem for the evaluation, as the predicted point cloud is propagated to the coordinates of the ground truth point cloud at the beginning. It must only be ensured that the coordinates of the predicted point cloud are not shifted compared to the ground truth point cloud. 
    * It is also ok if the predicted forest point cloud includes tree predictions for trees that are unlabeled in the ground truth forest point cloud (e.g. smaller trees or trees at the edge of the point cloud). The evaluation script only takes into account trees that are annotated in the ground truth point cloud. Predictions that do not correspond to any annotated ground truth tree are automatically identified and will not count as commission errors.

## General information
* We recommend to run the evaluation on point clouds subsampled with a voxel size of 0.1 m to avoid a disproportionate weighting of regions with a higher density. 
* Furthermore, the ground truth forest point cloud should contain both non-tree and tree points, with unlabeled points at the edge included in the ground truth point cloud (see [L1W](https://data.goettingen-research-online.de/dataset.xhtml?persistentId=doi:10.25625/VPMPID&version=DRAFT) and [Wytham Woods](https://data.goettingen-research-online.de/dataset.xhtml?persistentId=doi:10.25625/QUTUWU) as an example). The reason for this is twofold:

    1.) Both non-tree points and trees at the edge of the labeled segment represent a potential source of segmentation errors, e.g. non-tree points erroneously assigned to trees or segmentation errors that include points from unlabeled trees at the edge. Therefore, a realistic measure of segmentation performance must also take these points into account.

    2.) When complete forest point clouds are used as ground truths, it is easy to determine whether a prediction that was not matched to a ground truth tree represents a commission error or whether it is simply associated with an unlabeled tree or the understory. For example, in our evaluation script we determine this via precision.
* It should be noted that the evaluation is slightly inaccurate in case that unlabeled points actually belong to ground truth trees (see e.g. [Wytham Woods](https://data.goettingen-research-online.de/dataset.xhtml?persistentId=doi:10.25625/QUTUWU)). In this case, predicting these unlabeled points as tree points will negatively influence the segmentation metrics. However, we consider this inaccuracy to be negligible since the number of non-annotated points that belong to ground truth trees is small in published forest point clouds. 



## Running the evaluation
To run the evaluation on L1W, you need to first obtain segmentation results as described [here](segmentation_pipeline.md). Then run the following commands:

*1\) Download evaluation dataset*
```
python tree_learn/util/download.py --dataset_name benchmark_dataset_evaluation --root_folder data/benchmark
```

*2\) Run the evaluation*
```
python tools/evaluation/evaluate.py --config configs/evaluation/evaluate.yaml
```

*3) Obtain more fine-grained evaluation results*
* Aggregated segmentation results will be automatically saved in the log of the evaluation script.
* To take a look at segmentation results and errors of individual trees, we prepared a notebook that can be found at ``tools/evaluation/evaluate.ipynb``

Running the evaluation script on arbitrary forest point clouds works analogously. You only need to change the argument 'pred_forest_path' and 'gt_forest_path' in the evaluate configuration at ``configs/evaluation/evaluate.yaml`` to where your point clouds are located.
