import torch
from tqdm import tqdm
import numpy as np
import argparse
import os
import shutil
from tree_learn.util import (get_root_logger, make_labels_consecutive, merge_undecided_with_forest_cluster_space,
                             get_detections, get_detection_failures, filter_detection_stuff, get_config,
                             evaluate_instance_segmentation, propagate_preds, get_unique_instance_labels_for_splits,
                             get_instances, assign_remaining_points_nearest_neighbor, load_checkpoint, get_classifier_preds)
from tree_learn.model import TreeLearn


PROB_THRESHOLD_VALID_INSTANCES = 0.5
NOISE_LABEL_IN_GROUPING = -2
UNDERSTORY_LABEL_IN_GROUPING = -1
FLOOR_CLASS_IN_RAW_DATA = 9999
TREE_CLASS_IN_DATASET = 0
MIN_IOU_FOR_MATCH = 0.2
BENCHMARK_FOREST_PATH = '../datasets_simlink/data_trees/for_evaluation/benchmark_forest.npy'
TREE_NUMS_USED_FOR_EVALUATION_PATH = '../datasets_simlink/data_trees/for_evaluation/treenums_used_for_evaluation.npy'


def evaluate_treelearn(config, config_path=None):
    pointwise_data_dir = os.path.join(config.base_dir, 'pointwise_data')
    documentation_dir = os.path.join(config.base_dir, 'documentation')
    eval_results_dir = os.path.join(config.base_dir, 'eval_results')
    os.makedirs(eval_results_dir, exist_ok=True)
    os.makedirs(documentation_dir, exist_ok=True)

    if config_path is not None:
        shutil.copy(args.config, os.path.join(documentation_dir, os.path.basename(args.config)))

    logger = get_root_logger(os.path.join(documentation_dir, 'evaluate_treelearn_log.txt'))

    # load and prepare data
    semantic_prediction_logits = np.load(os.path.join(pointwise_data_dir, 'semantic_prediction_logits.npy'))
    offset_predictions = np.load(os.path.join(pointwise_data_dir, 'offset_predictions.npy'))
    coords = np.load(os.path.join(pointwise_data_dir, 'coords.npy'))
    instance_labels = np.load(os.path.join(pointwise_data_dir, 'instance_labels.npy'))
    feats = np.load(os.path.join(pointwise_data_dir, 'feats.npy'))

    # get unique instance labels
    unique_instance_labels = np.unique(instance_labels)
    unique_instance_labels = unique_instance_labels[unique_instance_labels != FLOOR_CLASS_IN_RAW_DATA]
    unique_instance_labels_y_greater_zero, unique_instance_labels_y_not_greater_zero = get_unique_instance_labels_for_splits(coords, instance_labels, unique_instance_labels)
    unique_instance_labels_y_greater_zero = np.array(unique_instance_labels_y_greater_zero)
    unique_instance_labels_y_not_greater_zero = np.array(unique_instance_labels_y_not_greater_zero)

    # filter unique instance labels based on whether they are in instances_to_evaluate
    instances_to_evaluate = np.load(TREE_NUMS_USED_FOR_EVALUATION_PATH)
    unique_instance_labels_y_greater_zero = unique_instance_labels_y_greater_zero[np.isin(unique_instance_labels_y_greater_zero, instances_to_evaluate)]
    unique_instance_labels_y_not_greater_zero = unique_instance_labels_y_not_greater_zero[np.isin(unique_instance_labels_y_not_greater_zero, instances_to_evaluate)]
    
    # instantiate results dict
    results_dict = dict()

    ########################### get false positives and false negatives on both test set halves
    for i, grouping_radius in tqdm(enumerate(config.grouping_radii)):
        results_dict = dict()
        results_dict['radius'] = grouping_radius
        results_dict['y_greater_zero'] = dict()
        results_dict['y_greater_zero']['detection_results'] = dict()
        results_dict['y_greater_zero']['segmentation_results'] = dict()
        results_dict['y_not_greater_zero'] = dict()
        results_dict['y_not_greater_zero']['detection_results'] = dict()
        results_dict['y_not_greater_zero']['segmentation_results'] = dict()

        logger.info(f'evaluation with grouping radius {grouping_radius}')
        config.grouping.radius = grouping_radius

        # instance evaluation
        logger.info('getting instances...')
        instance_preds = get_instances(coords, offset_predictions, semantic_prediction_logits, config.grouping, TREE_CLASS_IN_DATASET, local_filtering=config.local_filtering, global_filtering=config.global_filtering)
        # instance_preds = merge_undecided_with_forest_cluster_space(coords, offset_predictions, instance_preds)
        tree_mask = instance_preds != UNDERSTORY_LABEL_IN_GROUPING
        instance_preds[tree_mask] = assign_remaining_points_nearest_neighbor(coords[tree_mask] + offset_predictions[tree_mask], instance_preds[tree_mask])

        if config.use_classifier:
            # Run classifier on preliminary instances and add low confidence predictions to noise
            model_classifier = TreeLearn(**config.model).cuda()
            load_checkpoint(config.pretrain_classifier, logger, model_classifier)
            tree_prediction_probs = get_classifier_preds(model_classifier, coords, feats, instance_preds, config)
            invalid_instances = torch.where(tree_prediction_probs < PROB_THRESHOLD_VALID_INSTANCES)[0].numpy()
            results_dict['tree_prediction_probs'] = tree_prediction_probs
            results_dict['instance_preds_not_propagated_to_benchmark_pointcloud_unfiltered'] = np.copy(instance_preds)

            instance_preds[np.isin(instance_preds, invalid_instances)] = NOISE_LABEL_IN_GROUPING
            instance_preds[(instance_preds != NOISE_LABEL_IN_GROUPING) & (instance_preds != UNDERSTORY_LABEL_IN_GROUPING)] = make_labels_consecutive(instance_preds[(instance_preds != NOISE_LABEL_IN_GROUPING) & (instance_preds != UNDERSTORY_LABEL_IN_GROUPING)], start_num=0)
            # instance_preds = merge_undecided_with_forest_cluster_space(coords, offset_predictions, instance_preds)
            tree_mask = instance_preds != UNDERSTORY_LABEL_IN_GROUPING
            instance_preds[tree_mask] = assign_remaining_points_nearest_neighbor(coords[tree_mask] + offset_predictions[tree_mask], instance_preds[tree_mask])

        results_dict['instance_preds_not_propagated_to_benchmark_pointcloud'] = instance_preds

        # hungarian matching
        logger.info('getting detection results...')
        unique_instance_preds = np.arange(len(np.unique(instance_preds)) - 1)
        matched_gts, matched_preds, iou_matrix = get_detections(instance_labels, instance_preds, unique_instance_labels, unique_instance_preds, MIN_IOU_FOR_MATCH)

        # get detection failures and separate cases where non_matched_preds_corresponding_gt is nan
        detection_failures = get_detection_failures(matched_gts, matched_preds, unique_instance_labels, unique_instance_preds, iou_matrix)
        non_matched_gts, non_matched_preds, non_matched_preds_corresponding_gt, non_matched_gts_corresponding_larger_tree = detection_failures
        mask_nan = np.isnan(non_matched_preds_corresponding_gt)
        non_matched_preds_where_corresponding_gt_is_nan = non_matched_preds[mask_nan]
        non_matched_preds = non_matched_preds[~mask_nan]
        non_matched_preds_corresponding_gt = non_matched_preds_corresponding_gt[~mask_nan]

        # filter detections based on instances to evaluate in y > 0 and y <= 0
        filtered_result =  filter_detection_stuff(matched_gts, matched_preds, unique_instance_labels_y_greater_zero)
        matched_gts_y_greater_zero, matched_preds_y_greater_zero = filtered_result
        filtered_result = filter_detection_stuff(matched_gts, matched_preds, unique_instance_labels_y_not_greater_zero)
        matched_gts_y_not_greater_zero, matched_preds_y_not_greater_zero = filtered_result

        # filter false positives based on instances to evaluate in y > 0 and y <= 0
        filtered_result =  filter_detection_stuff(non_matched_preds_corresponding_gt, non_matched_preds, unique_instance_labels_y_greater_zero)
        non_matched_preds_corresponding_gt_y_greater_zero, non_matched_preds_y_greater_zero = filtered_result
        filtered_result = filter_detection_stuff(non_matched_preds_corresponding_gt, non_matched_preds, unique_instance_labels_y_not_greater_zero)
        non_matched_preds_corresponding_gt_y_not_greater_zero, non_matched_preds_y_not_greater_zero = filtered_result

        # filter false negatives based on instances to evaluate in y > 0 and y <= 0
        filtered_result =  filter_detection_stuff(non_matched_gts, non_matched_gts_corresponding_larger_tree, unique_instance_labels_y_greater_zero)
        non_matched_gts_y_greater_zero, non_matched_gts_corresponding_larger_tree_y_greater_zero = filtered_result
        filtered_result = filter_detection_stuff(non_matched_gts, non_matched_gts_corresponding_larger_tree, unique_instance_labels_y_not_greater_zero)
        non_matched_gts_y_not_greater_zero, non_matched_gts_corresponding_larger_tree_y_not_greater_zero = filtered_result

        # propagate instance predictions to benchmark pointcloud (also use benchmark instance labels from here then)
        logger.info('propagating predictions to coords of benchmark...')
        benchmark_forest = np.load(BENCHMARK_FOREST_PATH)
        benchmark_forest_coords = benchmark_forest[:, :3]
        benchmark_forest_instance_labels = benchmark_forest[:, 3]

        instance_preds = propagate_preds(coords, instance_preds, benchmark_forest_coords, 5)
        results_dict['instance_preds_propagated_to_benchmark_pointcloud'] = instance_preds

        # instance segmentation results y greater
        logger.info('getting segmentation results...')
        mask_part_of_merged_tree = np.isin(matched_gts_y_greater_zero, non_matched_gts_corresponding_larger_tree_y_greater_zero)
        matched_gts_y_greater_zero_no_merges = matched_gts_y_greater_zero[~mask_part_of_merged_tree]
        matched_preds_y_greater_zero_no_merges = matched_preds_y_greater_zero[~mask_part_of_merged_tree]
        instance_segmentation_results = evaluate_instance_segmentation(instance_preds, benchmark_forest_instance_labels, matched_gts_y_greater_zero_no_merges, 
                                                                    matched_preds_y_greater_zero_no_merges, benchmark_forest_coords, config.partitions.xy_partition_relative, 
                                                                    config.partitions.xy_partition_absolute, config.partitions.z_partition_relative, config.partitions.z_partition_absolute)
        no_partition_y_greater_zero, xy_relative_y_greater_zero, xy_absolute_y_greater_zero, z_relative_y_greater_zero, z_absolute_y_greater_zero = instance_segmentation_results

        # instance segmentation results y not greater
        mask_part_of_merged_tree = np.isin(matched_gts_y_not_greater_zero, non_matched_gts_corresponding_larger_tree_y_not_greater_zero)
        matched_gts_y_not_greater_zero_no_merges = matched_gts_y_not_greater_zero[~mask_part_of_merged_tree]
        matched_preds_y_not_greater_zero_no_merges = matched_preds_y_not_greater_zero[~mask_part_of_merged_tree]
        instance_segmentation_results = evaluate_instance_segmentation(instance_preds, benchmark_forest_instance_labels, matched_gts_y_not_greater_zero_no_merges, 
                                                                       matched_preds_y_not_greater_zero_no_merges, benchmark_forest_coords, config.partitions.xy_partition_relative, config.partitions.xy_partition_absolute, 
                                                                    config.partitions.z_partition_relative, config.partitions.z_partition_absolute)
        no_partition_y_not_greater_zero, xy_relative_y_not_greater_zero, xy_absolute_y_not_greater_zero, z_relative_y_not_greater_zero, z_absolute_y_not_greater_zero = instance_segmentation_results
        
        # save results
        logger.info('saving results...')
        results_dict['y_greater_zero']['detection_results']['matched_gts'] = matched_gts_y_greater_zero
        results_dict['y_greater_zero']['detection_results']['matched_preds'] = matched_preds_y_greater_zero
        results_dict['y_greater_zero']['detection_results']['non_matched_preds'] = non_matched_preds_y_greater_zero
        results_dict['y_greater_zero']['detection_results']['non_matched_preds_corresponding_gt'] = non_matched_preds_corresponding_gt_y_greater_zero
        results_dict['y_greater_zero']['detection_results']['non_matched_gts'] = non_matched_gts_y_greater_zero
        results_dict['y_greater_zero']['detection_results']['non_matched_gts_corresponding_larger_tree'] = non_matched_gts_corresponding_larger_tree_y_greater_zero
        results_dict['y_greater_zero']['segmentation_results']['no_partition'] = no_partition_y_greater_zero
        results_dict['y_greater_zero']['segmentation_results']['xy_partition_relative'] = xy_relative_y_greater_zero
        results_dict['y_greater_zero']['segmentation_results']['xy_partition_absolute'] = xy_absolute_y_greater_zero
        results_dict['y_greater_zero']['segmentation_results']['z_partition_relative'] = z_relative_y_greater_zero
        results_dict['y_greater_zero']['segmentation_results']['z_partition_absolute'] = z_absolute_y_greater_zero
         # this could be put either in y greater or not greater zero since it cannot be assigned. Just check manually where it belongs
        results_dict['y_greater_zero']['detection_results']['non_matched_preds_where_corresponding_gt_is_nan'] = non_matched_preds_where_corresponding_gt_is_nan

        results_dict['y_not_greater_zero']['detection_results']['matched_gts'] = matched_gts_y_not_greater_zero
        results_dict['y_not_greater_zero']['detection_results']['matched_preds'] = matched_preds_y_not_greater_zero
        results_dict['y_not_greater_zero']['detection_results']['non_matched_preds'] = non_matched_preds_y_not_greater_zero
        results_dict['y_not_greater_zero']['detection_results']['non_matched_preds_corresponding_gt'] = non_matched_preds_corresponding_gt_y_not_greater_zero
        results_dict['y_not_greater_zero']['detection_results']['non_matched_gts'] = non_matched_gts_y_not_greater_zero
        results_dict['y_not_greater_zero']['detection_results']['non_matched_gts_corresponding_larger_tree'] = non_matched_gts_corresponding_larger_tree_y_not_greater_zero
        results_dict['y_not_greater_zero']['segmentation_results']['no_partition'] = no_partition_y_not_greater_zero
        results_dict['y_not_greater_zero']['segmentation_results']['xy_partition_relative'] = xy_relative_y_not_greater_zero
        results_dict['y_not_greater_zero']['segmentation_results']['xy_partition_absolute'] = xy_absolute_y_not_greater_zero
        results_dict['y_not_greater_zero']['segmentation_results']['z_partition_relative'] = z_relative_y_not_greater_zero
        results_dict['y_not_greater_zero']['segmentation_results']['z_partition_absolute'] = z_absolute_y_not_greater_zero

        torch.save(results_dict, os.path.join(eval_results_dir, f'instance_evaluation{i}.pt'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser('eval')
    parser.add_argument('--config', type=str, help='path to config file for evaluation for treelearn')
    args = parser.parse_args()
    config = get_config(args.config)
    evaluate_treelearn(config, args)
