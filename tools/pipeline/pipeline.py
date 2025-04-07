import os
import numpy as np
import argparse
import pickle
import pprint
import shutil
from tree_learn.dataset import TreeDataset
from tree_learn.model import TreeLearn
from tree_learn.util import (munch_to_dict, build_dataloader, get_root_logger, load_checkpoint, ensemble, 
                             get_coords_within_shape, get_hull_buffer, get_hull, get_cluster_means,
                             propagate_preds, save_treewise, load_data, save_data, make_labels_consecutive, 
                             get_config, generate_tiles, assign_remaining_points_nearest_neighbor,
                             get_pointwise_preds, get_instances, propagate_preds_hash_full, propagate_preds_hash_vox)

TREE_CLASS_IN_PYTORCH_DATASET = 0
NON_TREES_LABEL_IN_GROUPING = 0
NOT_ASSIGNED_LABEL_IN_GROUPING = -1
START_NUM_PREDS = 1



def run_treelearn_pipeline(config, config_path=None):
    # make dirs
    plot_name = os.path.basename(config.forest_path)[:-4]
    base_dir = os.path.dirname(os.path.dirname(config.forest_path))
    documentation_dir = os.path.join(base_dir, 'documentation')
    unvoxelized_data_dir = os.path.join(base_dir, 'forest')
    voxelized_data_dir = os.path.join(base_dir, f'forest_voxelized{config.sample_generation.voxel_size}')
    tiles_dir = os.path.join(base_dir, 'tiles')
    results_dir_name = getattr(config.save_cfg, 'results_dir', 'results')
    results_dir = os.path.join(base_dir, results_dir_name)

    os.makedirs(documentation_dir, exist_ok=True)
    os.makedirs(unvoxelized_data_dir, exist_ok=True)
    os.makedirs(voxelized_data_dir, exist_ok=True)
    os.makedirs(tiles_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # quick and dirty fix for the fact that method throws errors/does not work with high-magnitude coords
    # --> center coords and de-center at the end
    data = load_data(config.forest_path)
    xyz = data[:, :3].astype(np.float64)
    xyz_mean = np.mean(xyz, 0).astype(np.float64)
    xyz_centered = xyz - xyz_mean
    # avoids overwriting of original file
    if not config.forest_path.endswith('.npz'):
        config.forest_path = config.forest_path[:-4] + '.npz'
    else:
        config.forest_path = config.forest_path[:-4] + '.npy'
    np.savez_compressed(config.forest_path, points=xyz_centered)
    
    # documentation
    logger = get_root_logger(os.path.join(documentation_dir, 'log_pipeline.txt'))
    logger.info(pprint.pformat(munch_to_dict(config), indent=2))
    if config_path is not None:
        shutil.copy(args.config, os.path.join(documentation_dir, os.path.basename(args.config)))

    # generate tiles used for inference and specify path to it in dataset config
    config.dataset_test.data_root = os.path.join(tiles_dir, 'npz')
    if config.tile_generation:
        logger.info('#################### generating tiles ####################')
        generate_tiles(config.sample_generation, config.forest_path, logger, config.save_cfg.return_type)

    # Make pointwise predictions with pretrained model
    logger.info(f'{plot_name}: #################### getting pointwise predictions ####################')
    model = TreeLearn(**config.model).cuda()
    dataset = TreeDataset(**config.dataset_test, logger=logger)
    dataloader = build_dataloader(dataset, training=False, **config.dataloader)
    load_checkpoint(config.pretrain, logger, model)
    pointwise_results = get_pointwise_preds(model, dataloader, config.model, logger)
    semantic_prediction_logits, semantic_labels, offset_predictions, offset_labels, coords, instance_labels, backbone_feats, input_feats = pointwise_results
    del model

    # ensemble predictions from overlapping tiles
    logger.info(f'{plot_name}: #################### ensembling predictions ####################')
    data = ensemble(coords, semantic_prediction_logits, semantic_labels, offset_predictions, 
                    offset_labels, instance_labels, backbone_feats, input_feats)
    coords, semantic_prediction_logits, semantic_labels, offset_predictions, offset_labels, instance_labels, backbone_feats, input_feats = data

    # get mask of inner coords if outer points should be removed
    if config.shape_cfg.outer_remove:
        logger.info(f'{plot_name}: #################### prepare remove outer points ####################')
        hull_buffer_large = get_hull_buffer(coords[:, :2], config.shape_cfg.alpha, buffersize=config.shape_cfg.outer_remove)
        mask_coords_within_hull_buffer_large = get_coords_within_shape(coords, hull_buffer_large)
        masks_inner_coords = np.logical_not(mask_coords_within_hull_buffer_large)

    # get tree detections
    logger.info(f'{plot_name}: #################### getting predicted instances ####################')
    instance_preds = get_instances(coords, offset_predictions, semantic_prediction_logits, config.grouping, input_feats[:, -1], TREE_CLASS_IN_PYTORCH_DATASET, NON_TREES_LABEL_IN_GROUPING, NOT_ASSIGNED_LABEL_IN_GROUPING, START_NUM_PREDS)
    instance_preds_after_initial_clustering = np.copy(instance_preds)

    # assign remaining points
    tree_mask = instance_preds != NON_TREES_LABEL_IN_GROUPING
    instance_preds[tree_mask] = assign_remaining_points_nearest_neighbor(coords[tree_mask] + offset_predictions[tree_mask], instance_preds[tree_mask], NOT_ASSIGNED_LABEL_IN_GROUPING)
    
    # save pointwise results
    if config.save_cfg.save_pointwise:
        pointwise_dir = os.path.join(results_dir, 'pointwise_results')
        os.makedirs(pointwise_dir, exist_ok=True)
        pointwise_results = {
            'coords': coords,
            'offset_predictions': offset_predictions,
            'offset_labels': offset_labels,
            'semantic_prediction_logits': semantic_prediction_logits,
            'semantic_labels': semantic_labels,
            'instance_labels': instance_labels,
            'backbone_feats': backbone_feats,
            'input_feats': input_feats,
            'instance_preds': instance_preds,
            'instance_preds_after_initial_clustering': instance_preds_after_initial_clustering
        }
        if config.shape_cfg.outer_remove:
            pointwise_results['masks_inner_coords'] = masks_inner_coords
            hull_buffer_large.to_pickle(os.path.join(pointwise_dir, 'hull_buffer_large.pkl'))
        np.savez_compressed(os.path.join(pointwise_dir, 'pointwise_results.npz'), **pointwise_results)
        
        # offset-shifted coordinates filtered by verticality and offset (initial clustering results); save as laz file for visualization
        verticality = input_feats[:, -1]
        verticality_mask = verticality >= config.grouping.tau_vert
        offset_mask = np.abs(offset_predictions[:, 2]) <= config.grouping.tau_off
        sem_mask = instance_preds != NON_TREES_LABEL_IN_GROUPING
        mask = verticality_mask & offset_mask & sem_mask
        cluster_coords = coords[mask] + offset_predictions[mask]
        cluster_coords = np.hstack([cluster_coords, instance_preds[mask].reshape(-1, 1)])
        save_data(cluster_coords, 'laz', 'cluster_coords_initial', pointwise_dir)
        
        # complete offset-shifted coordinates with instance predictions (clustering results after assigning remaining points); save as laz file for visualization
        cluster_coords = coords + offset_predictions
        cluster_coords = cluster_coords[instance_preds != NON_TREES_LABEL_IN_GROUPING]
        cluster_coords = np.hstack([cluster_coords, instance_preds[instance_preds != NON_TREES_LABEL_IN_GROUPING].reshape(-1, 1)])
        save_data(cluster_coords, 'laz', 'cluster_coords', pointwise_dir)

    # remove outer points with buffer
    if config.shape_cfg.outer_remove:
        coords, semantic_prediction_logits, semantic_labels, offset_predictions, offset_labels, instance_labels, instance_preds, input_feats = \
            coords[masks_inner_coords], semantic_prediction_logits[masks_inner_coords], \
            semantic_labels[masks_inner_coords], offset_predictions[masks_inner_coords], \
            offset_labels[masks_inner_coords], instance_labels[masks_inner_coords], \
            instance_preds[masks_inner_coords], input_feats[masks_inner_coords]
        instance_preds[instance_preds != NON_TREES_LABEL_IN_GROUPING], _ = make_labels_consecutive(instance_preds[instance_preds != NON_TREES_LABEL_IN_GROUPING], start_num=1)

    # get information whether tree clusters are within or outside hull (used for saving tree in different categories later)
    if config.save_cfg.save_treewise:
        cluster_means = get_cluster_means(coords[instance_preds != NON_TREES_LABEL_IN_GROUPING] + offset_predictions[instance_preds != NON_TREES_LABEL_IN_GROUPING], 
                                          instance_preds[instance_preds != NON_TREES_LABEL_IN_GROUPING])
        hull = get_hull(coords[:, :2], config.shape_cfg.alpha)
        cluster_means_within_hull = get_coords_within_shape(cluster_means, hull)

        # get information whether trees have points very close to hull (used for saving trees in different categories later)
        hull_buffer_small = get_hull_buffer(coords[:, :2], config.shape_cfg.alpha, buffersize=config.shape_cfg.buffer_size_to_determine_edge_trees)
        mask_coords_at_edge = get_coords_within_shape(coords, hull_buffer_small)
        instance_preds_at_edge = np.unique(instance_preds[mask_coords_at_edge])
        instance_preds_at_edge = np.delete(instance_preds_at_edge, np.where(instance_preds_at_edge == NON_TREES_LABEL_IN_GROUPING))
        insts_not_at_edge = np.ones(len(cluster_means_within_hull))
        insts_not_at_edge[instance_preds_at_edge-1] = 0
        insts_not_at_edge = insts_not_at_edge.astype('bool')
        
    # propagate predictions to original forest
    if config.save_cfg.return_type == 'original':
        logger.info(f'{plot_name}: Propagating predictions to original points')
        coords_to_return = load_data(config.forest_path)[:, :3]
        hash_mapping_path = os.path.join(voxelized_data_dir, f'{plot_name}_hash_mapping.pkl')
        with open(hash_mapping_path, 'rb') as pickle_file:
            hash_mapping = pickle.load(pickle_file)
        preds_to_return, not_yet_propagated = propagate_preds_hash_full(coords, instance_preds, coords_to_return, hash_mapping)
    elif config.save_cfg.return_type == 'voxelized':
        logger.info(f'{plot_name}: Propagating predictions to voxelized points')
        voxelized_forest_path = os.path.join(voxelized_data_dir, f'{plot_name}.npz')
        coords_to_return = load_data(voxelized_forest_path)[:, :3]
        preds_to_return, not_yet_propagated = propagate_preds_hash_vox(coords, instance_preds, coords_to_return)
    elif config.save_cfg.return_type == 'voxelized_and_filtered': # 'voxelized_and_filtered' is identical to 'voxelized' if no point filtering is specified in configs/_modular/sample_generation.yaml
        coords_to_return = coords
        preds_to_return = instance_preds
        not_yet_propagated = np.zeros(len(coords_to_return), dtype=bool)
    # optionally remove outer points
    if config.shape_cfg.outer_remove:
        mask_coords_to_return_within_hull_buffer_large = get_coords_within_shape(coords_to_return, hull_buffer_large)
        masks_inner_coords_to_return = np.logical_not(mask_coords_to_return_within_hull_buffer_large)
        coords_to_return = coords_to_return[masks_inner_coords_to_return]
        preds_to_return = preds_to_return[masks_inner_coords_to_return]
        not_yet_propagated = not_yet_propagated[masks_inner_coords_to_return]
    # propagate predictions to points that were not yet propagated
    if not_yet_propagated.any():
        preds_to_return[not_yet_propagated] = propagate_preds(coords, instance_preds, coords_to_return[not_yet_propagated], n_neighbors=5)
        
    # # add xyz_mean again which was potentially subtracted at the beginning
    coords_to_return = coords_to_return.astype(np.float64) + xyz_mean
        
    # save
    logger.info(f'{plot_name}: #################### Saving ####################')
    full_dir = os.path.join(results_dir, 'full_forest')
    os.makedirs(full_dir, exist_ok=True)

    for save_format in config.save_cfg.save_formats:
        save_data(np.hstack([coords_to_return, preds_to_return.reshape(-1, 1)]), save_format, plot_name, full_dir)
    if config.save_cfg.save_treewise:
        trees_dir = os.path.join(results_dir, 'individual_trees')
        os.makedirs(trees_dir, exist_ok=True)
        save_treewise(coords_to_return, preds_to_return, cluster_means_within_hull, insts_not_at_edge, "las", trees_dir, NON_TREES_LABEL_IN_GROUPING)
    return




if __name__ == '__main__':
    parser = argparse.ArgumentParser('tree_learn')
    parser.add_argument('--config', type=str, help='path to config file for pipeline')
    args = parser.parse_args()
    config = get_config(args.config)
    run_treelearn_pipeline(config, args.config)
