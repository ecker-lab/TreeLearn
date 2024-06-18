import geopandas
import alphashape

import numpy as np
import pandas as pd
import pickle
import os
import os.path as osp
import tqdm
import torch
import random
import laspy
from shapely.geometry import Point
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from scipy import stats
from sklearn.cluster import DBSCAN
from tree_learn.util.data_preparation import voxelize, compute_features, load_data, rad_filter, SampleGenerator


N_JOBS = 10 # number of threads/processes to use for several functions that have multiprocessing/multithreading enabled



# function to generate tiles
def generate_tiles(cfg, forest_path, logger, return_type='voxelized'):
    plot_name = os.path.basename(forest_path)[:-4]
    base_dir = os.path.dirname(os.path.dirname(forest_path))

    # dirs for data saving
    voxelized_dir = osp.join(base_dir, f'forest_voxelized{cfg.voxel_size}')
    features_dir = osp.join(base_dir, 'features')
    save_dir = osp.join(base_dir, 'tiles')
    os.makedirs(voxelized_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # voxelize forest and optionally calculate hash mapping to original points
    logger.info('voxelizing forest...')
    save_path_voxelized = osp.join(voxelized_dir, f'{plot_name}.npz')
    save_path_voxelized_original_idx = osp.join(voxelized_dir, f'{plot_name}_original_idx.pkl')
    save_path_hash_mapping = osp.join(voxelized_dir, f'{plot_name}_hash_mapping.pkl')
    if not osp.exists(save_path_voxelized) or (return_type == 'original' and not osp.exists(save_path_voxelized_original_idx)):
        data = load_data(forest_path)
        data, original_idx = voxelize(data, cfg.voxel_size)
        data = np.round(data, 2)
        data = data.astype(np.float32)
        np.savez_compressed(save_path_voxelized, points=data[:, :3], labels=data[:, 3])

        if return_type == 'original':
            original_idx = [list(item) for item in original_idx]
            # hash mapping
            hash_values = get_hash_values(data[:, :3])
            hash_mapping = get_hash_mapping(hash_values, original_idx)
            with open(save_path_voxelized_original_idx, 'wb') as f:
                pickle.dump(original_idx, f)
            with open(save_path_hash_mapping, 'wb') as f:
                pickle.dump(hash_mapping, f)
            del hash_values, original_idx, hash_mapping

    # calculating features
    logger.info('calculating features...')
    save_path_features = osp.join(features_dir, f'{plot_name}.npz')
    if not osp.exists(save_path_features):
        data = load_data(save_path_voxelized)
        features = compute_features(points=data[:, :3].astype(np.float64), search_radius=cfg.search_radius_features, feature_names=['verticality'], num_threads=N_JOBS)
        np.savez_compressed(save_path_features, features=features)

    # add cfg args that were generated dynamically based on plot
    logger.info('getting tiles...')
    cfg.sample_generator.plot_path = osp.join(voxelized_dir, f'{plot_name}.npz')
    cfg.sample_generator.features_path = osp.join(features_dir, f'{plot_name}.npz')
    cfg.sample_generator.save_dir = save_dir

    # generate tiles
    obj = SampleGenerator(**cfg.sample_generator)
    obj.tile_generate_and_save(cfg.inner_edge, cfg.outer_edge, cfg.stride, logger=logger)


def get_pointwise_preds(model, dataloader, config, logger=None):
    with torch.no_grad():
        model.eval()
        semantic_prediction_logits, offset_predictions, semantic_labels, offset_labels, coords, instance_labels, backbone_feats, input_feats = [], [], [], [], [], [], [], []
        for batch in tqdm.tqdm(dataloader):
            # get voxel_sizes to use in forward
            batch['voxel_size'] = config.voxel_size
            # forward
            try:
                output = model(batch, return_loss=False)
                offset_prediction, semantic_prediction_logit, backbone_feat = output['offset_predictions'], output['semantic_prediction_logits'], output['backbone_feats']
                offset_prediction, semantic_prediction_logit, backbone_feat = offset_prediction.cpu(), semantic_prediction_logit.cpu(), backbone_feat.cpu()
            except Exception as e:
                if "reach zero!!!" in str(e):
                    if logger:
                        logger.info('Error in forward pass due to axis size collapse to zero during contraction of U-Net. If this does not happen too often, the results should not be influenced.')
                    continue 
                else:
                    raise

            batch['coords'] = batch['coords'] + batch['centers']
            input_feats.append(batch['input_feats'][batch['masks_inner']])
            semantic_prediction_logits.append(semantic_prediction_logit[batch['masks_inner']]), semantic_labels.append(batch['semantic_labels'][batch['masks_inner']])
            offset_predictions.append(offset_prediction[batch['masks_inner']]), offset_labels.append(batch['offset_labels'][batch['masks_inner']])
            coords.append(batch['coords'][batch['masks_inner']]), instance_labels.append(batch['instance_labels'][batch['masks_inner']]), backbone_feats.append(backbone_feat[batch['masks_inner']])

    input_feats = torch.cat(input_feats, 0).numpy()
    semantic_prediction_logits, semantic_labels = torch.cat(semantic_prediction_logits, 0).numpy(), torch.cat(semantic_labels, 0).numpy()
    offset_predictions, offset_labels = torch.cat(offset_predictions, 0).numpy(), torch.cat(offset_labels, 0).numpy()
    coords, instance_labels, backbone_feats = torch.cat(coords, 0).numpy(), torch.cat(instance_labels).numpy(), torch.cat(backbone_feats, 0).numpy()

    return semantic_prediction_logits, semantic_labels, offset_predictions, offset_labels, coords, instance_labels, backbone_feats, input_feats


def ensemble(coords, semantic_scores, semantic_labels, offset_predictions, offset_labels, instance_labels, feats, input_feats):
    feats_col_names = [f'feats{i}' for i in range(feats.shape[1])]
    feats = pd.DataFrame(feats, columns=feats_col_names)
    
    input_feats_col_names = [f'input_feats{i}' for i in range(input_feats.shape[1])]
    input_feats = pd.DataFrame(input_feats, columns=input_feats_col_names)

    coords = pd.DataFrame(coords, columns=['x', 'y', 'z'])
    semantic_scores = pd.DataFrame(semantic_scores, columns=['sem_scores1', 'sem_scores2'])
    semantic_labels = pd.DataFrame(semantic_labels.reshape(-1, 1), columns=['semantic_labels'])
    offset_predictions = pd.DataFrame(offset_predictions, columns=['offset_pred1', 'offset_pred2', 'offset_pred3'])
    offset_labels = pd.DataFrame(offset_labels, columns=['offset_lab1', 'offset_lab2', 'offset_lab3'])
    instance_labels = pd.DataFrame(instance_labels.reshape(-1, 1), columns=['instance_labels'])

    df = pd.concat([coords, semantic_scores, semantic_labels, offset_predictions, offset_labels, instance_labels, feats, input_feats], axis=1)

    df = df.round({'x': 2, 'y': 2, 'z': 2})
    grouped = df.groupby(['x', 'y', 'z']).mean().reset_index()

    # Convert columns to desired data types
    coords = grouped[['x', 'y', 'z']].to_numpy().astype('float32')
    semantic_scores = grouped[['sem_scores1', 'sem_scores2']].to_numpy().astype('float32')
    semantic_labels = grouped[['semantic_labels']].to_numpy().astype('int64').flatten()
    offset_predictions = grouped[['offset_pred1', 'offset_pred2', 'offset_pred3']].to_numpy().astype('float32')
    offset_labels = grouped[['offset_lab1', 'offset_lab2', 'offset_lab3']].to_numpy().astype('float32')
    instance_labels = grouped[['instance_labels']].to_numpy().astype('int64').flatten()
    feats = grouped[feats_col_names].to_numpy().astype('float32')
    input_feats = grouped[input_feats_col_names].to_numpy().astype('float32')

    return coords, semantic_scores, semantic_labels, offset_predictions, offset_labels, instance_labels, feats, input_feats


def get_instances(coords, offset, semantic_prediction_logits, grouping_cfg, verticality_feat, tree_class_in_dataset, non_trees_label_in_grouping, not_assigned_label_in_grouping, start_num_preds):
    cluster_coords = coords + offset
    # cluster_coords[:, 2] = 0
    cluster_coords = cluster_coords[:, :3]

    # get tree coords whose offset magnitude and verticality feature is appropriate
    semantic_prediction_probs = torch.from_numpy(semantic_prediction_logits).float().softmax(dim=-1)
    tree_mask = semantic_prediction_probs[:, tree_class_in_dataset] >= grouping_cfg.tree_conf_thresh
    vertical_mask = verticality_feat > grouping_cfg.tau_vert
    offset_mask = np.abs(offset[:, 2]) < grouping_cfg.tau_off
    mask_before_filter = tree_mask.numpy() & vertical_mask & offset_mask
    ind_before_filter = np.where(mask_before_filter)[0]
    cluster_coords_tree_before_filter = cluster_coords[ind_before_filter]
    
    # further filter tree coords by removing points that are "floating around alone" in cluster space
    mask_filtering = rad_filter(cluster_coords_tree_before_filter, rad=0.05, npoints_rad=2)
    ind_after_filter = ind_before_filter[mask_filtering]
    cluster_coords_tree_after_filter = cluster_coords[ind_after_filter]
    cluster_coords_tree_after_filter = cluster_coords_tree_after_filter[:, :2]

    # get predictions
    predictions = non_trees_label_in_grouping * np.ones(len(cluster_coords))
    predictions[tree_mask] = not_assigned_label_in_grouping

    # GET PREDICTED INSTANCES
    pred_instances = group_dbscan(cluster_coords_tree_after_filter, grouping_cfg.tau_group, grouping_cfg.tau_min, not_assigned_label_in_grouping, start_num_preds)

    predictions[ind_after_filter] = pred_instances 
    return predictions.astype(np.int64)


def group_dbscan(cluster_coords, radius, npoint_thr, not_assigned_label_in_grouping, start_num_preds):
    # downsampling to avoid excessive RAM usage during DBSCAN.
    clustering = DBSCAN(eps=radius, min_samples=2, n_jobs=N_JOBS).fit(cluster_coords)
    cluster_nums, n_points = np.unique(clustering.labels_, return_counts=True)
    valid_cluster_nums = cluster_nums[(n_points >= npoint_thr) & (cluster_nums != -1)]
    ind_valid = np.isin(clustering.labels_, valid_cluster_nums)
    clustering.labels_[ind_valid] = make_labels_consecutive(clustering.labels_[ind_valid], start_num=start_num_preds)
    clustering.labels_[np.logical_not(ind_valid)] = not_assigned_label_in_grouping
    return clustering.labels_


def make_labels_consecutive(labels, start_num):
    palette = np.unique(labels)
    palette = np.sort(palette)
    key = np.arange(0, len(palette))
    index = np.digitize(labels, palette, right=True)
    labels = key[index]
    labels = labels + start_num
    return labels


def get_coords_within_shape(coords, shape):
    coords_df = pd.DataFrame(coords, columns=["x", "y", "z"])
    coords_df["xy"] = list(zip(coords_df["x"], coords_df["y"]))
    coords_df["xy"] = coords_df["xy"].apply(Point)
    coords_geodf = geopandas.GeoDataFrame(coords_df, geometry='xy')

    joined = coords_geodf.sjoin(shape, how="left", predicate="within")
    ind_within = np.array(joined["index_right"])
    ind_within[ind_within == 0] = 1
    ind_within[np.isnan(ind_within)] = 0
    ind_within = ind_within.astype("bool")
    return ind_within


def get_hull_buffer(coords, alpha, buffersize):
    # create 2-dimensional hull of forest xy-coordinates and from this create hull buffer
    hull_polygon = alphashape.alphashape(coords[:, :2], alpha)
    hull_line = hull_polygon.boundary
    hull_line_geoseries = geopandas.GeoSeries(hull_line)
    hull_buffer = hull_line_geoseries.buffer(buffersize)
    hull_buffer_geodf = geopandas.GeoDataFrame(geometry=hull_buffer)
    return hull_buffer_geodf


def get_hull(coords, alpha):
    # create 2-dimensional hull of forest xy-coordinates
    coords = pd.DataFrame(coords, columns=['x', 'y'])
    # coords = coords.round({'x': 2, 'y': 2, 'z': 2})
    coords.drop_duplicates(inplace=True)
    coords = coords.to_numpy()

    hull_polygon = alphashape.alphashape(coords, alpha)
    hull_polygon_geoseries = geopandas.GeoSeries(hull_polygon)
    hull_polygon_geodf = geopandas.GeoDataFrame(geometry=hull_polygon_geoseries)
    return hull_polygon_geodf


def get_cluster_means(coords, labels):
    df = pd.DataFrame(coords, columns=['x', 'y', 'z'])
    df['label'] = labels
    cluster_means = df.groupby('label').mean().values 
    return cluster_means


def assign_remaining_points_nearest_neighbor(coords, predictions, remaining_points_idx, n_neighbors=5):
    predictions = np.copy(predictions)
    assert len(coords) == len(predictions) # input variable should be of same size
    query_idx = np.argwhere(predictions == remaining_points_idx).reshape(-1)
    reference_idx = np.argwhere(predictions != remaining_points_idx).reshape(-1)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=N_JOBS)
    knn.fit(coords[reference_idx].copy(), predictions[reference_idx].copy())
    neighbors_predictions = knn.predict(coords[query_idx].copy())
    predictions[query_idx] = neighbors_predictions
    return predictions.astype(np.int64)


def propagate_preds(source_coords, source_preds, target_coords, n_neighbors):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
    nbrs.fit(source_coords)
    neighbours_indices = nbrs.kneighbors(target_coords, n_neighbors, return_distance=False) 
    neighbours_preds = source_preds[neighbours_indices] 

    target_preds = stats.mode(neighbours_preds, axis=1, nan_policy='raise', keepdims=False)
    target_preds = target_preds[0]

    return target_preds.astype(np.int64)


def generate_random_color():
    return [random.randint(0, 255) for _ in range(3)]


def save_data(data, save_format, save_name, save_folder, use_offset=True):
    if save_format == "las" or save_format == "laz":
        # get points and labels
        assert data.shape[1] == 4
        points = data[:, :3]
        labels = data[:, 3]
        classification = np.ones_like(labels)
        classification[labels == 0] = 2 # terrain according to For-Instance labeling convention (https://zenodo.org/records/8287792)
        classification[labels != 0] = 4 # stem according to For-Instance labeling convention (https://zenodo.org/records/8287792)

        # Create a new LAS file
        header = laspy.LasHeader(version="1.2", point_format=3)
        if use_offset:
            mean_x, mean_y, _ = points.mean(0)
            header.offsets = [mean_x, mean_y, 0]
        else:
            header.offsets = [0, 0, 0]
        
        points = points + header.offsets
        header.scales = [0.001, 0.001, 0.001]
        las = laspy.LasData(header)

        # Set the points and additional fields
        las.x = points[:, 0]
        las.y = points[:, 1]
        las.z = points[:, 2]

        las.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.uint32))
        las.treeID = labels
        las.classification = classification

        # Generate a color for each unique label
        unique_labels = np.unique(labels)
        color_map = {label: generate_random_color() for label in unique_labels}

        # Assign colors based on label
        colors = np.array([color_map[label] for label in labels], dtype=np.uint16)
        colors[classification == 2] = [0, 0, 0]

        # Set RGB colors in the LAS file
        las.red = colors[:, 0]
        las.green = colors[:, 1]
        las.blue = colors[:, 2]

        # Write the LAS file to disk
        save_path = osp.join(save_folder, f'{save_name}.{save_format}')
        las.write(save_path)
    elif save_format == "npy":
        save_path = osp.join(save_folder, f'{save_name}.{save_format}')
        np.save(save_path, data)
    elif save_format == "npz":
        save_path = osp.join(save_folder, f'{save_name}.{save_format}')
        np.savez_compressed(save_path, points=data[:, :3], labels=data[:, 3])
    elif save_format == "txt":
        save_path = osp.join(save_folder, f'{save_name}.{save_format}')
        np.savetxt(save_path, data)


def save_treewise(coords, instance_preds, cluster_means_within_hull, insts_not_at_edge, save_format, plot_results_dir, non_trees_label_in_grouping):
    completely_inside_dir = os.path.join(plot_results_dir, 'completely_inside')
    trunk_base_inside_dir = os.path.join(plot_results_dir, 'trunk_base_inside')
    trunk_base_outside_dir = os.path.join(plot_results_dir, 'trunk_base_outside')
    os.makedirs(completely_inside_dir, exist_ok=True)
    os.makedirs(trunk_base_inside_dir, exist_ok=True)
    os.makedirs(trunk_base_outside_dir, exist_ok=True)

    for i in np.unique(instance_preds):
        pred_coord = coords[instance_preds == i]
        pred_coord = np.hstack([pred_coord, i * np.ones(len(pred_coord))[:, None]])
        if i == non_trees_label_in_grouping:
            save_data(pred_coord, save_format, 'non_trees', plot_results_dir)
            continue

        if cluster_means_within_hull[i-1] and insts_not_at_edge[i-1]:
            save_data(pred_coord, save_format, str(int(i)), completely_inside_dir, use_offset=False)
        elif cluster_means_within_hull[i-1] and not insts_not_at_edge[i-1]:
            save_data(pred_coord, save_format, str(int(i)), trunk_base_inside_dir, use_offset=False)
        elif not cluster_means_within_hull[i-1]:
            save_data(pred_coord, save_format, str(int(i)), trunk_base_outside_dir, use_offset=False)


def get_hash_values(voxelized_points):
    hash_values = []
    for point in voxelized_points:
        point_tuple = tuple(point)
        hash_value = hash(point_tuple)
        hash_values.append(hash_value)
    return hash_values


def get_hash_mapping(hash_values, original_idx):
    hash_mapping = {}
    for i, hash_value in enumerate(hash_values):
        hash_mapping[hash_value] = original_idx[i]
    return hash_mapping


def propagate_preds_hash_full(coords, instance_preds, coords_to_return, hash_mapping):
    coords = np.round(coords, 2)
    hash_values = get_hash_values(coords)

    target_preds = np.empty(coords_to_return.shape[0], np.int64)
    not_yet_propagated = np.ones(coords_to_return.shape[0], np.bool)
    for i, hash_value in enumerate(hash_values):
        target_preds[np.array(hash_mapping[hash_value], np.int64)] = instance_preds[i]
        not_yet_propagated[np.array(hash_mapping[hash_value], np.int64)] = False

    return target_preds, not_yet_propagated


def propagate_preds_hash_vox(coords, instance_preds, coords_to_return):
    hash_values_original = np.array(get_hash_values(coords_to_return), np.int64)
    hash_values_current = np.array(get_hash_values(np.round(coords, 2)), np.int64)
    not_yet_propagated = ~np.isin(hash_values_original, hash_values_current)

    mapping  = np.argsort(hash_values_current)[np.argsort(np.argsort(hash_values_original[~not_yet_propagated]))]
    preds_to_return = np.empty(coords_to_return.shape[0], np.int64)
    preds_to_return[~not_yet_propagated] = instance_preds[mapping]
    return preds_to_return, not_yet_propagated