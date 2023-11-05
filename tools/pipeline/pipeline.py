import torch
import os
import numpy as np
import argparse
import pprint
import shutil
from tree_learn.dataset import TreeDataset
from tree_learn.model import TreeLearn, Classifier
from tree_learn.util import (build_dataloader, get_root_logger, load_checkpoint, ensemble, 
                             get_coords_within_shape, get_hull_buffer, get_hull, get_cluster_means,
                             propagate_preds, save_treewise, load_data, save, make_labels_consecutive, 
                             get_config, convert_to_pcd, generate_tiles, assign_remaining_points_nearest_neighbor,
                             get_pointwise_preds, get_instances, get_classifier_preds)

TREE_CLASS_IN_DATASET = 0
NOISE_LABEL_IN_GROUPING = -2
UNDERSTORY_LABEL_IN_GROUPING = -1
PROB_THRESHOLD_VALID_INSTANCES = 0.5




def run_treelearn_pipeline(config, config_path=None):
    # make dirs
    plot_name = os.path.basename(config.forest_path)[:-4]
    base_dir = os.path.dirname(os.path.dirname(config.forest_path))
    documentation_dir = os.path.join(base_dir, 'documentation')
    unvoxelized_data_dir = os.path.join(base_dir, 'forests')
    voxelized_data_dir = os.path.join(base_dir, f'forests_voxelized{config.sample_generation.voxel_size}')
    tiles_dir = os.path.join(base_dir, 'tiles')
    results_dir = os.path.join(base_dir, 'results')
    plot_results_dir = os.path.join(results_dir, plot_name)
    os.makedirs(documentation_dir, exist_ok=True)
    os.makedirs(unvoxelized_data_dir, exist_ok=True)
    os.makedirs(voxelized_data_dir, exist_ok=True)
    os.makedirs(tiles_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plot_results_dir, exist_ok=True)

    # documentation
    logger = get_root_logger(os.path.join(documentation_dir, 'log_pipeline'))
    logger.info(pprint.pformat(config, indent=2))
    if config_path is not None:
        shutil.copy(args.config, os.path.join(documentation_dir, os.path.basename(args.config)))

    # generate tiles used for inference and specify path to it in dataset config
    config.dataset_test.data_root = os.path.join(tiles_dir, plot_name)
    if config.tile_generation:
        logger.info('#################### generating tiles ####################')
        generate_tiles(config.sample_generation, base_dir, plot_name, logger)

    # Make pointwise predictions with pretrained model
    logger.info(f'{plot_name}: #################### getting pointwise predictions ####################')
    model_pointwise = TreeLearn(**config.model).cuda()
    dataset = TreeDataset(**config.dataset_test, logger=logger)
    dataloader = build_dataloader(dataset, training=False, **config.dataloader)
    load_checkpoint(config.pretrain_pointwise, logger, model_pointwise)
    semantic_prediction_logits, semantic_labels, offset_predictions, offset_labels, coords, instance_labels, feats = get_pointwise_preds(model_pointwise, dataloader, config.model)
    del model_pointwise

    # ensemble predictions from overlapping tiles
    logger.info(f'{plot_name}: #################### ensembling predictions ####################')
    data = ensemble(coords, semantic_prediction_logits, semantic_labels, offset_predictions, 
                    offset_labels, instance_labels, feats)
    coords, semantic_prediction_logits, semantic_labels, offset_predictions, offset_labels, instance_labels, feats = data

    # get mask of inner coords if outer points should be removed
    if config.shape_cfg.outer_remove:
        logger.info(f'{plot_name}: #################### remove outer points ####################')
        hull_buffer_large = get_hull_buffer(coords, config.shape_cfg.alpha, buffersize=config.shape_cfg.outer_remove)
        mask_coords_within_hull_buffer_large = get_coords_within_shape(coords, hull_buffer_large)
        masks_inner_coords = np.logical_not(mask_coords_within_hull_buffer_large)

    # save pointwise results
    if config.save_cfg.save_pointwise:
        pointwise_dir = os.path.join(plot_results_dir, 'pointwise_gt_and_preds')
        os.makedirs(pointwise_dir, exist_ok=True)
        np.save(os.path.join(pointwise_dir, 'coords.npy'), coords)
        np.save(os.path.join(pointwise_dir, 'offset_predictions.npy'), offset_predictions)
        np.save(os.path.join(pointwise_dir, 'offset_labels.npy'), offset_labels)
        np.save(os.path.join(pointwise_dir, 'semantic_prediction_logits.npy'), semantic_prediction_logits)
        np.save(os.path.join(pointwise_dir, 'semantic_labels.npy'), semantic_labels)
        np.save(os.path.join(pointwise_dir, 'instance_labels.npy'), instance_labels)
        np.save(os.path.join(pointwise_dir, 'feats.npy'), feats)
        np.save(os.path.join(pointwise_dir, 'masks_inner_coords.npy'), masks_inner_coords)
        hull_buffer_large.to_pickle(os.path.join(pointwise_dir, 'hull_buffer_large.pkl'))

    # if only pointwise predictions are to be generated, terminate here
    if config.save_cfg.only_pointwise:
        return

    # get instances by grouping based on pointwise predictions
    logger.info(f'{plot_name}: #################### getting predicted instances ####################')
    instance_preds = get_instances(coords, offset_predictions, semantic_prediction_logits, config.grouping, TREE_CLASS_IN_DATASET, 
                                   global_filtering=config.global_filtering, local_filtering=config.local_filtering)
    tree_mask = instance_preds != UNDERSTORY_LABEL_IN_GROUPING
    instance_preds[tree_mask] = assign_remaining_points_nearest_neighbor(coords[tree_mask] + offset_predictions[tree_mask], instance_preds[tree_mask])
    
    # Run classifier on preliminary instances and add low confidence predictions to noise
    logger.info(f'{plot_name}: #################### Run classifier on preliminary instances ####################')
    for key in config.model_classifier:
        config.model[key] = config.model_classifier[key]
    model_classifier = Classifier(**config.model).cuda()
    load_checkpoint(config.pretrain_classifier, logger, model_classifier)
    tree_prediction_probs = get_classifier_preds(model_classifier, coords, feats, instance_preds, config)

    # also save instance related info (before adding insecure instances to noise)
    if config.save_cfg.save_pointwise:
        np.save(os.path.join(pointwise_dir, 'instance_preds.npy'), instance_preds)
        np.save(os.path.join(pointwise_dir, 'tree_prediction_probs.npy'), tree_prediction_probs.numpy())

    invalid_instances = torch.where(tree_prediction_probs < PROB_THRESHOLD_VALID_INSTANCES)[0].numpy()
    instance_preds[np.isin(instance_preds, invalid_instances)] = NOISE_LABEL_IN_GROUPING

    # remove outer points with buffer
    if config.shape_cfg.outer_remove:
        coords, semantic_prediction_logits, semantic_labels, offset_predictions, offset_labels, instance_labels, instance_preds = \
            coords[masks_inner_coords], semantic_prediction_logits[masks_inner_coords], \
            semantic_labels[masks_inner_coords], offset_predictions[masks_inner_coords], \
            offset_labels[masks_inner_coords], instance_labels[masks_inner_coords], \
            instance_preds[masks_inner_coords]
        instance_preds[(instance_preds != NOISE_LABEL_IN_GROUPING) & (instance_preds != UNDERSTORY_LABEL_IN_GROUPING)] = make_labels_consecutive(instance_preds[(instance_preds != NOISE_LABEL_IN_GROUPING) & (instance_preds != UNDERSTORY_LABEL_IN_GROUPING)], start_num=0)

    # get information whether tree clusters are within or outside hull (used for saving tree in different categories later)
    if config.save_cfg.save_treewise:
        cluster_means = get_cluster_means(coords[(instance_preds != NOISE_LABEL_IN_GROUPING) & (instance_preds != UNDERSTORY_LABEL_IN_GROUPING)] + offset_predictions[(instance_preds != NOISE_LABEL_IN_GROUPING) & (instance_preds != UNDERSTORY_LABEL_IN_GROUPING)], 
                                          instance_preds[(instance_preds != NOISE_LABEL_IN_GROUPING) & (instance_preds != UNDERSTORY_LABEL_IN_GROUPING)])
        hull = get_hull(coords, config.shape_cfg.alpha)
        cluster_means_within_hull = get_coords_within_shape(cluster_means, hull)

        # get information whether trees have points very close to hull (used for saving trees in different categories later)
        hull_buffer_small = get_hull_buffer(coords, config.shape_cfg.alpha, buffersize=config.shape_cfg.buffer_size_to_determine_edge_trees)
        mask_coords_at_edge = get_coords_within_shape(coords, hull_buffer_small)
        instance_preds_at_edge = np.unique(instance_preds[mask_coords_at_edge])
        instance_preds_at_edge = np.delete(instance_preds_at_edge, np.where((instance_preds_at_edge == NOISE_LABEL_IN_GROUPING) | (instance_preds_at_edge == UNDERSTORY_LABEL_IN_GROUPING)))
        insts_not_at_edge = np.ones(len(cluster_means_within_hull))
        insts_not_at_edge[instance_preds_at_edge] = 0
        insts_not_at_edge = insts_not_at_edge.astype('bool')

    # prediction propagation
    if NOISE_LABEL_IN_GROUPING in instance_preds:
        tree_mask = instance_preds != UNDERSTORY_LABEL_IN_GROUPING
        instance_preds[tree_mask] = assign_remaining_points_nearest_neighbor(coords[tree_mask] + offset_predictions[tree_mask], instance_preds[tree_mask])

    # propagate predictions to original forest
    if config.save_cfg.return_type == 'original':
        logger.info(f'{plot_name}: Propagating predictions to original points')
        coords_to_return = load_data(config.forest_path)[:, :3]
        if config.shape_cfg.outer_remove:
            mask_coords_to_return_within_hull_buffer_large = get_coords_within_shape(coords_to_return, hull_buffer_large)
            masks_inner_coords_to_return = np.logical_not(mask_coords_to_return_within_hull_buffer_large)
            coords_to_return = coords_to_return[masks_inner_coords_to_return]
        preds_to_return = propagate_preds(coords, instance_preds, coords_to_return, n_neighbors=5)
    elif config.save_cfg.return_type == 'voxelized':
        logger.info(f'{plot_name}: Propagating predictions to voxelized points')
        voxelized_forest_path = os.path.join(voxelized_data_dir, os.path.basename(config.forest_path))
        coords_to_return = load_data(voxelized_forest_path)[:, :3]
        if config.shape_cfg.outer_remove:
            mask_coords_to_return_within_hull_buffer_large = get_coords_within_shape(coords_to_return, hull_buffer_large)
            masks_inner_coords_to_return = np.logical_not(mask_coords_to_return_within_hull_buffer_large)
            coords_to_return = coords_to_return[masks_inner_coords_to_return]
        preds_to_return = propagate_preds(coords, instance_preds, coords_to_return, n_neighbors=5)
    elif config.save_cfg.return_type == 'voxelized_and_denoised':
        coords_to_return = coords
        preds_to_return = instance_preds
        
    # save
    logger.info(f'{plot_name}: #################### Saving ####################')
    trees_dir = os.path.join(plot_results_dir, 'individual_trees')
    full_dir = os.path.join(plot_results_dir, 'full_forest')
    os.makedirs(trees_dir, exist_ok=True)
    os.makedirs(full_dir, exist_ok=True)
    save(np.hstack([coords_to_return, preds_to_return.reshape(-1, 1)]), config.save_cfg.save_format, plot_name, full_dir)
    convert_to_pcd(os.path.join(full_dir, f'{plot_name}.ply'), np.hstack((coords_to_return, preds_to_return.reshape(-1, 1))), color=True)
    if config.save_cfg.save_treewise:
        save_treewise(coords_to_return, preds_to_return, cluster_means_within_hull, insts_not_at_edge, config.save_cfg.save_format, trees_dir)
    return




if __name__ == '__main__':
    parser = argparse.ArgumentParser('tree_learn')
    parser.add_argument('--config', type=str, help='path to config file for pipeline')
    args = parser.parse_args()
    config = get_config(args.config)
    run_treelearn_pipeline(config, args.config)
