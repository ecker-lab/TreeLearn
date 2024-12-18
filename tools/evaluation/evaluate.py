import torch
import numpy as np
import argparse
import shutil
import os
import laspy
from tree_learn.util import (get_root_logger, make_labels_consecutive, get_config,
                             get_detections, get_detection_failures, save_data,
                             evaluate_instance_segmentation, propagate_preds, load_data)
NON_TREE_LABEL = 0


def evaluate(config, config_path=None):
    ################################## DOCUMENTATION AND LOGGING
    config.base_dir = os.path.join(os.path.dirname(config.paths.pred_forest_path), 'evaluation')
    documentation_dir = os.path.join(config.base_dir, 'documentation')
    os.makedirs(documentation_dir, exist_ok=True)

    logger = get_root_logger(os.path.join(documentation_dir, 'evaluate_log.txt'))
    if config_path is not None:
        shutil.copy(args.config, os.path.join(documentation_dir, os.path.basename(args.config)))

    ################################## LOAD GROUND TRUTH
    # get coords and labels
    gt_forest = load_data(config.paths.gt_forest_path)
    gt_forest_coords = gt_forest[:, :3]
    gt_forest_instance_labels = gt_forest[:, 3].astype('int')
    
    # set non-tree label to -1 and tree nums to 0, 1, 2, etc. for the purpose of evaluation
    gt_forest_instance_labels[gt_forest_instance_labels == NON_TREE_LABEL] = -1
    tree_mask = gt_forest_instance_labels != -1
    gt_forest_instance_labels[tree_mask], mapping_to_original_gt_nums = make_labels_consecutive(gt_forest_instance_labels[tree_mask], start_num=0)
    mapping_to_original_gt_nums[-1] = NON_TREE_LABEL
    
    ################################## LOAD PREDICTIONS
    # get coords and labels
    pred_forest = load_data(config.paths.pred_forest_path)
    pred_forest_coords = pred_forest[:, :3]
    instance_preds = pred_forest[:, 3].astype('int')

    # propagate instance predictions to gt pointcloud coordinates via nearest neighbors so that they can be compared
    logger.info('propagating predictions to coords of ground truth...')
    instance_preds = propagate_preds(pred_forest_coords, instance_preds, gt_forest_coords, 5)
    
    # set non-tree label to -1 and tree nums to 0, 1, 2, etc. for the purpose of evaluation
    instance_preds[instance_preds == NON_TREE_LABEL] = -1
    tree_mask = instance_preds != -1
    instance_preds[tree_mask], mapping_to_original_pred_nums = make_labels_consecutive(instance_preds[tree_mask], start_num=0)
    mapping_to_original_pred_nums[-1] = NON_TREE_LABEL
    
    ################################## GETTING DETECTION RESULTS
    # hungarian matching
    logger.info('getting detection results...')
    detection_results = get_detections(gt_forest_instance_labels, instance_preds, config.thresholds.min_iou_for_match, -1)
    matched_gts, matched_preds, iou_matrix, precision_matrix, recall_matrix = detection_results

    # get detection failures by comparing matched gts and preds to all gts and preds
    unique_instance_labels = np.arange(np.max(gt_forest_instance_labels) + 1)
    unique_instance_preds = np.arange(np.max(instance_preds) + 1)
    detection_failures = get_detection_failures(matched_gts, matched_preds, unique_instance_labels, unique_instance_preds, iou_matrix, precision_matrix, recall_matrix, 
                                                config.thresholds.min_precision_for_pred, config.thresholds.min_recall_for_gt)
    non_matched_gts, non_matched_preds, non_matched_preds_corresponding_gt, non_matched_gts_corresponding_pred, non_matched_gts_corresponding_other_tree = detection_failures
    
    ################################## GETTING SEGMENTATION RESULTS
    logger.info('getting segmentation results...')
    # unique gts and corresponding preds based on maximum iou to calculate coverage (according to ForAINet paper), as well as precision and recall on the point level
    unique_gts = np.arange(iou_matrix.shape[1])
    unique_preds = iou_matrix.argmax(axis=0)
    assert np.max(gt_forest_instance_labels) == np.max(unique_gts)
    assert np.max(instance_preds) == iou_matrix.shape[0] - 1
    # get fine-grained instance segmentation evaluation
    instance_segmentation_results = evaluate_instance_segmentation(instance_preds, gt_forest_instance_labels, unique_gts, unique_preds, gt_forest_coords, 
                                                                   mapping_to_original_gt_nums, mapping_to_original_pred_nums, config.partitions.xy_partition, 
                                                                   config.partitions.z_partition)
    no_partition, xy_partition, z_partition = instance_segmentation_results

    ################################## MAP RESULTS TO ORIGINAL LABELS AND CALCULATE METRICS
    # matched preds and gts
    matched_gts = np.array([mapping_to_original_gt_nums[label] for label in matched_gts])
    matched_preds = np.array([mapping_to_original_pred_nums[label] for label in matched_preds])
    # non matched preds and information related to them
    non_matched_preds = np.array([mapping_to_original_pred_nums[label] for label in non_matched_preds])
    non_matched_preds_corresponding_gt = np.array([mapping_to_original_gt_nums[label] if not np.isnan(label) else np.nan for label in non_matched_preds_corresponding_gt])
    non_matched_preds_filtered = np.array([pred for pred, gt in zip(non_matched_preds, non_matched_preds_corresponding_gt) if not np.isnan(gt)]) # filter non_matched_preds (see eval section in paper)
    non_matched_preds_corresponding_gt_filtered = np.array([gt for gt in non_matched_preds_corresponding_gt if not np.isnan(gt)]) # apply same filtering
    # non matched gts and information related to them
    non_matched_gts = np.array([mapping_to_original_gt_nums[label] for label in non_matched_gts])
    non_matched_gts_corresponding_other_tree = np.array([mapping_to_original_gt_nums[label] if not np.isnan(label) else np.nan for label in non_matched_gts_corresponding_other_tree])
    non_matched_gts_corresponding_pred = np.array([mapping_to_original_pred_nums[label] if not np.isnan(label) else np.nan for label in non_matched_gts_corresponding_pred])
    
    # calculate aggregated detection metrics
    completeness = len(matched_gts) / (len(matched_gts) + len(non_matched_gts))
    omission_error_rate = 1 - completeness
    commission_error_rate = len(non_matched_preds_filtered) / (len(matched_preds) + len(non_matched_preds_filtered))
    f1_score = 2 * ((1 - commission_error_rate) * (1 - omission_error_rate)) / (2 - (commission_error_rate + omission_error_rate))
    completeness = np.round(completeness * 100, 1)
    omission_error_rate = np.round(omission_error_rate * 100, 1)
    commission_error_rate = np.round(commission_error_rate * 100, 1)
    f1_score = np.round(f1_score * 100, 1)
    # calculate aggregated segmentation metrics
    prec_rec_iou = no_partition[['prec', 'rec', 'iou']]
    evaluation_scores_meaned = np.round(prec_rec_iou.mean(0) * 100, 1)
    
    ################################## LOG METRICS
    # Core detection metrics
    logger.info("\n===== Results detection evaluation =====")
    logger.info(f"Completeness: {completeness}%")
    logger.info(f"Omission Error Rate: {omission_error_rate}%")
    logger.info(f"Commission Error Rate: {commission_error_rate}%")
    logger.info(f"F1 Score: {f1_score}%")

    # Core segmentation metrics
    logger.info("\n===== Results segmentation evaluation =====")
    logger.info(f"Precision: {evaluation_scores_meaned['prec']}%")
    logger.info(f"Recall: {evaluation_scores_meaned['rec']}%")
    logger.info(f"Coverage: {evaluation_scores_meaned['iou']}%")

    ################################## SAVE RESULTS
    # save predictions propagated to gt point cloud (same points and point ordering as gt point cloud, might be useful for some analyses)
    instance_preds = np.array([mapping_to_original_pred_nums[label] for label in instance_preds])
    pred_forest_propagated_to_gt_pointcloud = np.hstack([gt_forest_coords, instance_preds.reshape(-1, 1)])
    save_data(pred_forest_propagated_to_gt_pointcloud, 'laz', 'pred_forest_propagated_to_gt_pointcloud', config.base_dir)
    # aggregated detection metrics
    results_dict = dict()
    results_dict['detection_results'] = dict()
    results_dict['detection_results']['completeness'] = completeness
    results_dict['detection_results']['omission_error_rate'] = omission_error_rate
    results_dict['detection_results']['commission_error_rate'] = commission_error_rate
    results_dict['detection_results']['f1_score'] = f1_score
    # matched preds and ground truths
    results_dict['detection_results']['matched_gts'] = matched_gts
    results_dict['detection_results']['matched_preds'] = matched_preds
    # non matched preds and information related to them
    results_dict['detection_results']['non_matched_preds'] = non_matched_preds
    results_dict['detection_results']['non_matched_preds_corresponding_gt'] = non_matched_preds_corresponding_gt
    results_dict['detection_results']['non_matched_preds_filtered'] = non_matched_preds_filtered
    results_dict['detection_results']['non_matched_preds_corresponding_gt_filtered'] = non_matched_preds_corresponding_gt_filtered
    # non matched gts and information related to them
    results_dict['detection_results']['non_matched_gts'] = non_matched_gts
    results_dict['detection_results']['non_matched_gts_corresponding_other_tree'] = non_matched_gts_corresponding_other_tree
    results_dict['detection_results']['non_matched_gts_corresponding_pred'] = non_matched_gts_corresponding_pred
    # aggregated segmentation results
    results_dict['segmentation_results'] = dict()
    results_dict['segmentation_results']['precision'] = evaluation_scores_meaned['prec']
    results_dict['segmentation_results']['recall'] = evaluation_scores_meaned['rec']
    results_dict['segmentation_results']['iou'] = evaluation_scores_meaned['iou']
    # segmentation results
    results_dict['segmentation_results']['no_partition'] = no_partition
    results_dict['segmentation_results']['xy_partition'] = xy_partition
    results_dict['segmentation_results']['z_partition'] = z_partition
    torch.save(results_dict, os.path.join(config.base_dir, 'evaluation_results.pt'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser('eval')
    parser.add_argument('--config', type=str, help='path to config file for evaluation')
    args = parser.parse_args()
    config = get_config(args.config)
    evaluate(config, args.config)
