import os
import argparse
import torch
from tqdm import tqdm
import numpy as np
from tree_learn.util import (get_root_logger, get_detections, make_labels_consecutive, 
                             get_config, get_instances, merge_undecided_with_forest_cluster_space,
                             assign_remaining_points_nearest_neighbor)

# pipeline tool is used here which is not in tree_learn module
import sys
sys.path.append('./tools/pipeline')
from pipeline import run_treelearn_pipeline

UNDERSTORY_LABEL_IN_GROUPING = -1
TREE_CLASS_IN_DATASET = 0
FLOOR_CLASS_IN_RAW_DATA = 9999
N = 200


def get_classifier_training_data(config_classifier_data_generation, config_pipeline):
    # logger
    documentation_dir = os.path.join(config_classifier_data_generation.base_dir, 'documentation')
    os.makedirs(documentation_dir, exist_ok=True)
    logger = get_root_logger(os.path.join(documentation_dir, 'log_classifier_training_data_generation'))

    # modify pipeline args
    for key in config_classifier_data_generation.pipeline:
        config_pipeline[key] = config_classifier_data_generation.pipeline[key]

    # directories
    forest_dir = os.path.join(config_classifier_data_generation.base_dir, 'forests')
    save_dir_classifier_data = os.path.join(config_classifier_data_generation.base_dir, 'classifier_data')
    os.makedirs(save_dir_classifier_data, exist_ok=True)

    ################################################# get pointwise results for all plots
    logger.info('Getting pointwise results on forests')
    for plot_file in os.listdir(forest_dir):
        forest_path = os.path.join(forest_dir, plot_file)
        config_pipeline.forest_path = forest_path
        run_treelearn_pipeline(config_pipeline)


    ################################################# get instance results for all plots
    logger.info('Getting instance results on forests')
    for plot_file in tqdm(os.listdir(forest_dir)):
        # dirs 
        plot_name = plot_file[:-4]
        plot_pointwise_results_dir = os.path.join(config_classifier_data_generation.base_dir, 'results', plot_name, 'pointwise_gt_and_preds')
        plot_instance_results_dir = os.path.join(config_classifier_data_generation.base_dir, 'results', plot_name, 'instance_preds')
        os.makedirs(plot_instance_results_dir, exist_ok=True)

        # load pointwise
        coords = np.load(os.path.join(plot_pointwise_results_dir, 'coords.npy'))
        offset_predictions = np.load(os.path.join(plot_pointwise_results_dir, 'offset_predictions.npy'))
        semantic_prediction_logits = np.load(os.path.join(plot_pointwise_results_dir, 'semantic_prediction_logits.npy'))

        for radius in config_classifier_data_generation.instances.grouping_radii:
            # get instances by grouping based on pointwise predictions
            config_classifier_data_generation.grouping.radius = radius
            instance_preds = get_instances(coords, offset_predictions, semantic_prediction_logits, 
                                        config_classifier_data_generation.grouping, TREE_CLASS_IN_DATASET, 
                                        global_filtering=config_classifier_data_generation.instances.global_filtering, 
                                        local_filtering=config_classifier_data_generation.instances.local_filtering)
            tree_mask = instance_preds != UNDERSTORY_LABEL_IN_GROUPING
            instance_preds[tree_mask] = assign_remaining_points_nearest_neighbor(coords[tree_mask] + offset_predictions[tree_mask], instance_preds[tree_mask])
            # instance_preds = merge_undecided_with_forest_cluster_space(coords, offset_predictions, instance_preds)
            np.save(os.path.join(plot_instance_results_dir, f'instance_preds_radius{radius}.npy'), instance_preds)


    ################################################# get classifier training data
    logger.info('Getting classifier training data on forests')
    for plot_file in tqdm(os.listdir(forest_dir)):
        # dirs 
        plot_name = plot_file[:-4]
        plot_pointwise_results_dir = os.path.join(config_classifier_data_generation.base_dir, 'results', plot_name, 'pointwise_gt_and_preds')
        plot_instance_results_dir = os.path.join(config_classifier_data_generation.base_dir, 'results', plot_name, 'instance_preds')

        # load pointwise
        coords = np.load(os.path.join(plot_pointwise_results_dir, 'coords.npy'))
        feats = np.load(os.path.join(plot_pointwise_results_dir, 'feats.npy'))
        instance_labels = np.load(os.path.join(plot_pointwise_results_dir, 'instance_labels.npy'))
        instance_labels[instance_labels != FLOOR_CLASS_IN_RAW_DATA] = make_labels_consecutive(instance_labels[instance_labels != FLOOR_CLASS_IN_RAW_DATA], start_num=0)

        # get unique instance labels
        unique_instance_labels = np.unique(instance_labels)
        unique_instance_labels = unique_instance_labels[unique_instance_labels != FLOOR_CLASS_IN_RAW_DATA]

        # get unique instance labels of non-edge trees
        masks_inner_coords = np.load(os.path.join(plot_pointwise_results_dir, 'masks_inner_coords.npy'))
        instance_labels_outer = instance_labels[~masks_inner_coords]
        unique_instance_labels_inner = np.setdiff1d(unique_instance_labels, np.unique(instance_labels_outer))
        del instance_labels_outer, masks_inner_coords

        for i, radius in enumerate(config_classifier_data_generation.instances.grouping_radii):
            # get instances by grouping based on pointwise predictions
            instance_preds = np.load(os.path.join(plot_instance_results_dir, f'instance_preds_radius{radius}.npy'))
            instance_preds[(instance_preds != -2) & (instance_preds != -1)] = make_labels_consecutive(instance_preds[(instance_preds != -2) & (instance_preds != -1)], start_num=0)
            unique_instance_preds = np.unique(instance_preds)
            unique_instance_preds = unique_instance_preds[(unique_instance_preds != -1) & (unique_instance_preds != -2)]

            # get indices of predictions that are actually understory (these should be removed since they might be small trees which i dont wanna classify as noise)
            ind_understory = []
            for pred_num in unique_instance_preds:
                mask_pred = instance_preds == pred_num
                instance_labels_mask_pred = instance_labels[mask_pred]
                if (instance_labels_mask_pred == FLOOR_CLASS_IN_RAW_DATA).sum() / len(instance_labels_mask_pred) > 0.7:
                    ind_understory.append(pred_num)
            ind_understory = np.array(ind_understory)

            # get iou
            _, _, iou_matrix = get_detections(instance_labels, instance_preds, unique_instance_labels, unique_instance_preds, 0.3)
            max_iou = iou_matrix.max(1)
            corresponding_gt = iou_matrix.argmax(1)


            # get indices of positive and negative predictions that correspond to inner ground truth
            mask_corresponding_gt_in_inner = np.isin(corresponding_gt, unique_instance_labels_inner)
            mask_positive = max_iou >= config_classifier_data_generation.classifier_training_data.iou_threshold_positive[i]
            mask_negative = max_iou <= config_classifier_data_generation.classifier_training_data.iou_threshold_negative[i]
            mask_positive_inner = mask_positive & mask_corresponding_gt_in_inner
            mask_negative_inner = mask_negative & mask_corresponding_gt_in_inner
            ind_positive_inner = np.where(mask_positive_inner)[0]
            ind_negative_inner = np.where(mask_negative_inner)[0]

            # sort indices of negative and positive predictions by iou (positive predictions ascending and negative predictions descending)
            argsort_ascending = np.argsort(max_iou)
            argsort_descending = np.argsort(max_iou)[::-1]
            ind_positive_inner_ascending = argsort_ascending[np.isin(argsort_ascending, ind_positive_inner) & ~np.isin(argsort_ascending, ind_understory)]
            ind_negative_inner_descending = argsort_descending[np.isin(argsort_descending, ind_negative_inner) & ~np.isin(argsort_descending, ind_understory)]

            # sample at maximum N positives and negatives and save them in the format required for training
            for j, ind_positive in enumerate(ind_positive_inner_ascending[:N]):
                mask_positive = instance_preds == ind_positive
                
                res = dict()
                res['points'] = coords[mask_positive]
                res['feat'] = feats[mask_positive]
                res['cls_label'] = 1
                save_name = plot_name + f'_radius{radius}' + f'_positive{j}.pt'
                torch.save(res, os.path.join(save_dir_classifier_data, save_name))

            for j, ind_negative in enumerate(ind_negative_inner_descending[:N]):
                mask_negative = instance_preds == ind_negative
                
                res = dict()
                res['points'] = coords[mask_negative]
                res['feat'] = feats[mask_negative]
                res['cls_label'] = 0
                save_name = plot_name + f'_radius{radius}' + f'_negative{j}.pt'
                torch.save(res, os.path.join(save_dir_classifier_data, save_name))




if __name__ == '__main__':
    parser = argparse.ArgumentParser('tree_learn')
    parser.add_argument('--config_pipeline', type=str, help='path to config file for pipeline')
    parser.add_argument('--config_classifier_data_generation', type=str, help='path to config file for plot results generation')
    args = parser.parse_args()
    config_pipeline = get_config(args.config_pipeline)
    config_classifier_data_generation = get_config(args.config_classifier_data_generation)
    get_classifier_training_data(config_classifier_data_generation, config_pipeline)
