import torch
import numpy as np
import argparse
import shutil
import os
import laspy
from tree_learn.util import (get_root_logger, make_labels_consecutive, get_config,
                             get_detections, get_detection_failures, filter_detection_stuff,
                             evaluate_instance_segmentation, propagate_preds, load_data)

START_NUM_PREDS = 0
INSTANCE_LABEL_IGNORE_IN_RAW_DATA = -1 # ignore label in benchmark dataset numpy array
NON_TREE_LABEL = 0 # label for non-tree points both in predictions and benchmark dataset numpy array


def evaluate(config, config_path=None):
    pred_forest_path = config.pred_forest_path
    config.base_dir = os.path.dirname(pred_forest_path)
    
    documentation_dir = os.path.join(config.base_dir, 'documentation')
    os.makedirs(documentation_dir, exist_ok=True)

    logger = get_root_logger(os.path.join(documentation_dir, 'evaluate_log.txt'))
    if config_path is not None:
        shutil.copy(args.config, os.path.join(documentation_dir, os.path.basename(args.config)))

    # load pred and ground truth
    benchmark_forest = load_data(config.benchmark_forest_path)
    outpoints_trees = laspy.read(config.benchmark_forest_path).OutpointsTreeID
    benchmark_forest[:, 3][outpoints_trees != 0] = outpoints_trees[outpoints_trees != 0]
    benchmark_forest = benchmark_forest[benchmark_forest[:, 3] != INSTANCE_LABEL_IGNORE_IN_RAW_DATA]
    benchmark_forest_coords = benchmark_forest[:, :3]
    benchmark_forest_instance_labels = benchmark_forest[:, 3]
    benchmark_forest_instance_labels = benchmark_forest_instance_labels - 1

    pred_forest = load_data(pred_forest_path)
    pred_forest_coords = pred_forest[:, :3]
    instance_preds = pred_forest[:, 3].astype('int')

    # propagate instance predictions to benchmark pointcloud
    logger.info('propagating predictions to coords of benchmark...')
    instance_preds = propagate_preds(pred_forest_coords, instance_preds, benchmark_forest_coords, 5)
    instance_preds = instance_preds - 1

    # make labels consecutive and get unique instance preds
    instance_preds[instance_preds != (NON_TREE_LABEL-1)] = make_labels_consecutive(instance_preds[instance_preds != (NON_TREE_LABEL-1)], start_num=START_NUM_PREDS)
    unique_instance_preds = np.unique(instance_preds)
    unique_instance_preds = unique_instance_preds[unique_instance_preds != (NON_TREE_LABEL-1)]

    # get unique instance labels
    unique_instance_labels = np.unique(benchmark_forest_instance_labels)
    unique_instance_labels = unique_instance_labels[unique_instance_labels != (NON_TREE_LABEL-1)]

    # instantiate results dict
    results_dict = dict()
    results_dict['detection_results'] = dict()
    results_dict['segmentation_results'] = dict()
    results_dict = dict()
    results_dict['detection_results'] = dict()
    results_dict['segmentation_results'] = dict()
    results_dict['instance_preds_propagated_to_benchmark_pointcloud'] = instance_preds + 1
    
    # hungarian matching
    logger.info('getting detection results...')
    matched_gts, matched_preds, iou_matrix = get_detections(benchmark_forest_instance_labels, instance_preds, unique_instance_labels, unique_instance_preds, config.min_iou_for_match, (NON_TREE_LABEL-1))

    # get detection failures
    detection_failures = get_detection_failures(matched_gts, matched_preds, unique_instance_labels, unique_instance_preds, iou_matrix)
    non_matched_gts, non_matched_preds, non_matched_preds_corresponding_gt, non_matched_gts_corresponding_larger_tree = detection_failures
    mask_nan = np.isnan(non_matched_preds_corresponding_gt)
    non_matched_preds_where_corresponding_gt_is_nan = non_matched_preds[mask_nan]
    non_matched_preds = non_matched_preds[~mask_nan]
    non_matched_preds_corresponding_gt = non_matched_preds_corresponding_gt[~mask_nan]

    # filter detections based on instances to evaluate
    instances_to_evaluate = np.loadtxt(config.tree_nums_used_for_evaluation_path, skiprows=1)
    instances_to_evaluate = instances_to_evaluate - 1
    filtered_result =  filter_detection_stuff(matched_gts, matched_preds, instances_to_evaluate)
    matched_gts, matched_preds = filtered_result

    # filter false positives based on instances to evaluate
    filtered_result =  filter_detection_stuff(non_matched_preds_corresponding_gt, non_matched_preds, instances_to_evaluate)
    non_matched_preds_corresponding_gt, non_matched_preds = filtered_result

    # filter false negatives based on instances to evaluate
    filtered_result =  filter_detection_stuff(non_matched_gts, non_matched_gts_corresponding_larger_tree, instances_to_evaluate)
    non_matched_gts, non_matched_gts_corresponding_larger_tree = filtered_result
    
    # instance segmentation results
    logger.info('getting segmentation results...')
    mask_part_of_merged_tree = np.isin(matched_gts, non_matched_gts_corresponding_larger_tree)
    matched_gts_no_merges = matched_gts[~mask_part_of_merged_tree]
    matched_preds_no_merges = matched_preds[~mask_part_of_merged_tree]
    instance_segmentation_results = evaluate_instance_segmentation(instance_preds, benchmark_forest_instance_labels, matched_gts_no_merges, 
                                                                   matched_preds_no_merges, benchmark_forest_coords, config.partitions.xy_partition_relative, config.partitions.xy_partition_absolute, 
                                                                   config.partitions.z_partition_relative, config.partitions.z_partition_absolute)
    no_partition, xy_relative, xy_absolute, z_relative, z_absolute = instance_segmentation_results

    # save results
    logger.info('saving results...')
    results_dict['detection_results']['matched_gts'] = matched_gts + 1
    results_dict['detection_results']['matched_preds'] = matched_preds + 1
    results_dict['detection_results']['non_matched_preds'] = non_matched_preds + 1
    results_dict['detection_results']['non_matched_preds_corresponding_gt'] = non_matched_preds_corresponding_gt + 1
    results_dict['detection_results']['non_matched_gts'] = non_matched_gts + 1
    results_dict['detection_results']['non_matched_gts_corresponding_larger_tree'] = non_matched_gts_corresponding_larger_tree + 1
    results_dict['detection_results']['non_matched_preds_where_corresponding_gt_is_nan'] = non_matched_preds_where_corresponding_gt_is_nan + 1
    results_dict['segmentation_results']['no_partition'] = no_partition
    results_dict['segmentation_results']['xy_partition_relative'] = xy_relative
    results_dict['segmentation_results']['xy_partition_absolute'] = xy_absolute
    results_dict['segmentation_results']['z_partition_relative'] = z_relative
    results_dict['segmentation_results']['z_partition_absolute'] = z_absolute
    torch.save(results_dict, os.path.join(config.base_dir, 'instance_evaluation.pt'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser('eval')
    parser.add_argument('--config', type=str, help='path to config file for evaluation')
    args = parser.parse_args()
    config = get_config(args.config)
    evaluate(config, args.config)
