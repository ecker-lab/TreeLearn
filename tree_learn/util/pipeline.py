import geopandas
import alphashape

import numpy as np
import pandas as pd
import open3d as o3d
import os
import os.path as osp
import tqdm
import torch
from shapely.geometry import Point
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from scipy import stats
from sklearn.cluster import DBSCAN
from tree_learn.util.data_preparation import voxelize, compute_features, load_data, sor_filter, SampleGenerator
from tree_learn.util.train import get_voxel_sizes


N_CPUS = 4
NOISE_LABEL_IN_GROUPING = -2
EPSILON_EUCLIDEAN_CLUSTERING = 0.15
EPSILON_SOR_FILTER = 0.00001 # ensures that every point with roughly above average distance is discarded
UNDERSTORY_LABEL_IN_GROUPING = -1
LOCAL_FILTERING_RADIUS = 0.2


# function to generate tiles
def generate_tiles(cfg, base_dir, plot_name, logger):

    # dirs for data saving
    forests_dir = osp.join(base_dir, 'forests')
    voxelized_dir = osp.join(base_dir, f'forests_voxelized{cfg.voxel_size}')
    save_dir = osp.join(base_dir, 'tiles')
    os.makedirs(forests_dir, exist_ok=True)
    os.makedirs(voxelized_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)


    # calculate features
    if cfg.search_radius_features is not None:
        features_dir = osp.join(base_dir, 'features')
        os.makedirs(features_dir, exist_ok=True)

        logger.info('calculating features...')
        save_path_features = osp.join(features_dir, f'{plot_name}.npy')
        if not osp.exists(save_path_features):
            data = load_data(osp.join(forests_dir, f'{plot_name}.npy'))
            features = compute_features(points=data[:, :3].astype(np.float64), search_radius=cfg.search_radius_features)
            np.save(save_path_features, features)

    # voxelize data and features
    logger.info('voxelizing data and features...')
    save_path_data = osp.join(voxelized_dir, f'{plot_name}.npy')
    if not osp.exists(save_path_data):
        data = load_data(osp.join(forests_dir, f'{plot_name}.npy'))

        if cfg.search_radius_features is not None:
            features = np.load(save_path_features)
            data_features = voxelize(np.hstack((data, features)), cfg.voxel_size)
            data = data_features[:, :4]
            data = np.round(data, 2)
            features = data_features[:, 4:]
            np.save(save_path_data, data.astype(np.float32))
            np.save(save_path_features, features.astype(np.float32))
        else:
            data = voxelize(data, cfg.voxel_size)
            np.save(save_path_data, data.astype(np.float32))

    logger.info('getting tiles...')

    # add cfg args that were generated dynamically based on plot
    cfg.sample_generator.plot_path = osp.join(voxelized_dir, f'{plot_name}.npy')
    cfg.sample_generator.features_path = osp.join(features_dir, f'{plot_name}.npy') if cfg.search_radius_features is not None else None
    save_dir_plot = osp.join(save_dir, plot_name)
    cfg.sample_generator.save_dir = save_dir_plot
    os.makedirs(save_dir_plot, exist_ok=True)


    # generate tiles
    obj = SampleGenerator(**cfg.sample_generator)
    obj.tile_generate_and_save(cfg.inner_edge, cfg.outer_edge, cfg.stride, plot_corners=None, logger=logger)


def get_pointwise_preds(model, dataloader, config):
    with torch.no_grad():
        model.eval()
        semantic_prediction_logits, offset_predictions, semantic_labels, offset_labels, coords, instance_labels, feats = [], [], [], [], [], [], []
        for batch in tqdm.tqdm(dataloader):
            # get voxel_sizes to use in forward
            voxel_sizes = get_voxel_sizes(batch, config)
            batch['voxel_sizes'] = voxel_sizes
            # forward
            try:
                output = model(batch, return_loss=False)
                offset_prediction, semantic_prediction_logit, feat = output['offset_predictions'], output['semantic_prediction_logits'], output['backbone_feats']
            except:
                continue
            offset_prediction, semantic_prediction_logit, feat = offset_prediction.cpu(), semantic_prediction_logit.cpu(), feat.cpu()
            batch['coords'] = batch['coords'] + batch['centers']
            semantic_prediction_logits.append(semantic_prediction_logit[batch['masks_inner']]), semantic_labels.append(batch['semantic_labels'][batch['masks_inner']])
            offset_predictions.append(offset_prediction[batch['masks_inner']]), offset_labels.append(batch['offset_labels'][batch['masks_inner']])
            coords.append(batch['coords'][batch['masks_inner']]), instance_labels.append(batch['instance_labels'][batch['masks_inner']]), feats.append(feat[batch['masks_inner']])

    semantic_prediction_logits, semantic_labels = torch.cat(semantic_prediction_logits, 0).numpy(), torch.cat(semantic_labels, 0).numpy()
    offset_predictions, offset_labels = torch.cat(offset_predictions, 0).numpy(), torch.cat(offset_labels, 0).numpy()
    coords, instance_labels, feats = torch.cat(coords, 0).numpy(), torch.cat(instance_labels).numpy(), torch.cat(feats, 0).numpy()
    
    return semantic_prediction_logits, semantic_labels, offset_predictions, offset_labels, coords, instance_labels, feats

def get_classifier_preds(model, coords, feats, instance_preds, config):

    tree_prediction_probs = []
    n_pred_instances = np.max(instance_preds) + 1 # instances enumerated start with 0
    with torch.no_grad():
        model.eval()
        for pred_instance_num in tqdm.tqdm(range(n_pred_instances)):
            input = dict()
            input['coords'] = torch.from_numpy(coords[instance_preds == pred_instance_num])
            input['feats'] = torch.from_numpy(feats[instance_preds == pred_instance_num])
            input['batch_ids'] = torch.zeros(len(input['coords'])).long()
            input['batch_size'] = 1
            input['voxel_sizes'] = get_voxel_sizes(input, config.model)
            
            output = model(input, return_loss=False)
            cls_prediction_logits = output['cls_prediction_logits']
  
            cls_prediction_prob = cls_prediction_logits.softmax(dim=-1)
            tree_prediction_probs.append(cls_prediction_prob.squeeze()[1].item())

    return torch.tensor(tree_prediction_probs)


def ensemble(coords, semantic_scores, semantic_labels, offset_predictions, offset_labels, instance_labels, feats):
    feats_col_names = [f'feats{i}' for i in range(feats.shape[1])]
    feats = pd.DataFrame(feats, columns=feats_col_names)
    coords = pd.DataFrame(coords, columns=['x', 'y', 'z'])
    semantic_scores = pd.DataFrame(semantic_scores, columns=['sem_scores1', 'sem_scores2'])
    semantic_labels = pd.DataFrame(semantic_labels.reshape(-1, 1), columns=['semantic_labels'])
    offset_predictions = pd.DataFrame(offset_predictions, columns=['offset_pred1', 'offset_pred2', 'offset_pred3'])
    offset_labels = pd.DataFrame(offset_labels, columns=['offset_lab1', 'offset_lab2', 'offset_lab3'])
    instance_labels = pd.DataFrame(instance_labels.reshape(-1, 1), columns=['instance_labels'])
    df = pd.concat([coords, semantic_scores, semantic_labels, offset_predictions, offset_labels, instance_labels, feats], axis=1)

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

    return coords, semantic_scores, semantic_labels, offset_predictions, offset_labels, instance_labels, feats


def get_instances(coords, offset, semantic_prediction_logits, grouping_cfg, tree_class_in_dataset, global_filtering=False, local_filtering=False, start_num_preds=0):
    cluster_coords = coords + offset
    # cluster_coords[:, 2] = 0
    semantic_prediction_probs = torch.from_numpy(semantic_prediction_logits).float().softmax(dim=-1)
    predictions = UNDERSTORY_LABEL_IN_GROUPING * np.ones(len(cluster_coords))
    tree_inds_added_to_noise = np.ones(0, dtype='int')
    
    # NO FILTERING
    tree_inds = (semantic_prediction_probs[:, tree_class_in_dataset] >= grouping_cfg.tree_conf_thresh).nonzero().view(-1).numpy()
    cluster_coords_tree = cluster_coords[tree_inds]
    tree_inds_no_filtering = np.copy(tree_inds)
    cluster_coords_tree_no_filtering = np.copy(cluster_coords_tree)

    # GLOBAL FILTERING
    if global_filtering:
        mask_filtering = sor_filter(cluster_coords_tree_no_filtering, n_neigh_sor=100, multiplier_sor=EPSILON_SOR_FILTER)
        cluster_coords_tree = cluster_coords_tree_no_filtering[mask_filtering]
        tree_inds = tree_inds_no_filtering[mask_filtering]
        tree_inds_added_to_noise = np.append(tree_inds_added_to_noise, tree_inds_no_filtering[~mask_filtering])
    
    # LOCAL FILTERING
    if local_filtering:
        pred_instances = group_dbscan(cluster_coords_tree, LOCAL_FILTERING_RADIUS, grouping_cfg.npoint_thr, start_num_preds, downsample=True)

        for pred_num in np.unique(pred_instances):
            pred_mask = pred_instances == pred_num
            tree_inds_pred = tree_inds[pred_mask]
            cluster_coords_pred = cluster_coords_tree[pred_mask]

            mask_filtering = sor_filter(cluster_coords_pred, n_neigh_sor=100, multiplier_sor=EPSILON_SOR_FILTER)
            tree_inds_added_to_noise = np.append(tree_inds_added_to_noise, tree_inds_pred[~mask_filtering])

        mask_filtering = ~np.isin(tree_inds_no_filtering, tree_inds_added_to_noise)
        cluster_coords_tree = cluster_coords_tree_no_filtering[mask_filtering]
        tree_inds = tree_inds_no_filtering[mask_filtering]

    # GET PREDICTED INSTANCES
    pred_instances = group_dbscan(cluster_coords_tree, grouping_cfg.radius, grouping_cfg.npoint_thr, start_num_preds, downsample=True)


    predictions[tree_inds] = pred_instances 
    predictions[tree_inds_added_to_noise] = NOISE_LABEL_IN_GROUPING
    return predictions.astype(np.int64)


def group_dbscan(cluster_coords, radius, npoint_thr, start_num_preds, downsample=False):
    # downsampling to avoid excessive RAM usage during DBSCAN.
    if downsample:
        voxel_size = radius/4
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cluster_coords)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
        voxels = voxel_grid.get_voxels()

        # get cluster coords of voxelized point cloud and mapping from voxel grid coordinates to integers
        mapping = dict()
        voxel_coords = []
        for i, voxel in tqdm.tqdm(enumerate(voxels)):
            voxel_coords.append(voxel_grid.get_voxel_center_coordinate(voxel.grid_index))
            mapping[tuple(voxel.grid_index)] = i
        voxel_coords = np.array(voxel_coords)

        # associate original points with voxel grid coordinates using the integers obtained from mapping above
        v2p_map = []
        for row in tqdm.tqdm(cluster_coords):
            v2p_map.append(mapping[tuple(voxel_grid.get_voxel(row))])
        v2p_map = np.array(v2p_map)
        cluster_coords = voxel_coords

        # DBSCAN
        print('clustering')
        clustering = DBSCAN(eps=radius, min_samples=2, n_jobs=N_CPUS).fit(cluster_coords)
        clustering.labels_ = clustering.labels_[v2p_map]

        cluster_nums, n_points = np.unique(clustering.labels_, return_counts=True)
        valid_cluster_nums = cluster_nums[(n_points >= npoint_thr) & (cluster_nums != -1)]
        ind_valid = np.isin(clustering.labels_, valid_cluster_nums)
        clustering.labels_[ind_valid] = make_labels_consecutive(clustering.labels_[ind_valid], start_num=start_num_preds)
        clustering.labels_[np.logical_not(ind_valid)] = NOISE_LABEL_IN_GROUPING


    else:
        clustering = DBSCAN(eps=radius, min_samples=2, n_jobs=N_CPUS).fit(cluster_coords)
        cluster_nums, n_points = np.unique(clustering.labels_, return_counts=True)
        valid_cluster_nums = cluster_nums[(n_points >= npoint_thr) & (cluster_nums != -1)]
        ind_valid = np.isin(clustering.labels_, valid_cluster_nums)
        clustering.labels_[ind_valid] = make_labels_consecutive(clustering.labels_[ind_valid], start_num=start_num_preds)
        clustering.labels_[np.logical_not(ind_valid)] = NOISE_LABEL_IN_GROUPING
    return clustering.labels_


def make_labels_consecutive(labels, start_num=0):
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
    hull_polygon = alphashape.alphashape(coords[:, :2], alpha)
    hull_polygon_geoseries = geopandas.GeoSeries(hull_polygon)
    hull_polygon_geodf = geopandas.GeoDataFrame(geometry=hull_polygon_geoseries)
    return hull_polygon_geodf


def get_cluster_means(coords, labels):
    df = pd.DataFrame(coords, columns=['x', 'y', 'z'])
    df['label'] = labels
    cluster_means = df.groupby('label').mean().values 
    return cluster_means


def merge_undecided_with_forest_euclidean_space(coords, instance_preds):
    instance_preds = np.copy(instance_preds)
    ind_undecided = instance_preds == NOISE_LABEL_IN_GROUPING
    ind_undecided = np.where(ind_undecided)[0]
    coords_undecided = coords[ind_undecided]
    clustering_undecided = DBSCAN(eps=EPSILON_EUCLIDEAN_CLUSTERING, min_samples=2).fit(coords_undecided)

    ind_undecided_valid = clustering_undecided.labels_ != -1 # this has to stay -1 since it is implemented as undecided in DBSCAN
    coords_undecided_valid = coords_undecided[ind_undecided_valid]
    labels_undecided_valid = clustering_undecided.labels_[ind_undecided_valid]
    coords_undecided_noise = coords_undecided[np.logical_not(ind_undecided_valid)]

    # propagate clusters to forest
    coords_undecided_valid_cluster_means = get_cluster_means(coords_undecided_valid, labels_undecided_valid)
    labels_undecided_valid_propagated = propagate_preds(coords[instance_preds != NOISE_LABEL_IN_GROUPING], instance_preds[instance_preds != NOISE_LABEL_IN_GROUPING], coords_undecided_valid_cluster_means, 10)
    labels_undecided_valid_propagated = labels_undecided_valid_propagated[labels_undecided_valid]
    instance_preds[ind_undecided[ind_undecided_valid]] = labels_undecided_valid_propagated

    # propagate noise to forest
    labels_undecided_noise_propagated = propagate_preds(coords[instance_preds != NOISE_LABEL_IN_GROUPING], instance_preds[instance_preds != NOISE_LABEL_IN_GROUPING], coords_undecided_noise, 10)
    instance_preds[ind_undecided[np.logical_not(ind_undecided_valid)]] = labels_undecided_noise_propagated

    return instance_preds.astype(np.int64)


def merge_undecided_with_forest_cluster_space(coords, offset_preds, instance_preds):
    cluster_coords = coords + offset_preds
    undecided_inds = instance_preds == NOISE_LABEL_IN_GROUPING
    
    # first assign ground
    instance_preds = assign_remaining_points_nearest_neighbor(coords, instance_preds)
    
    # those points that have not been assigned to ground are now assigned to trees via nearest neighors in cluster space
    tree_ind = instance_preds != UNDERSTORY_LABEL_IN_GROUPING
    undecided_tree_ind = undecided_inds & tree_ind
    instance_preds[undecided_tree_ind] = NOISE_LABEL_IN_GROUPING

    # then use cluster space to assign to tree instance
    instance_preds[tree_ind] = assign_remaining_points_nearest_neighbor(cluster_coords[tree_ind], instance_preds[tree_ind])
    return instance_preds


def assign_remaining_points_nearest_neighbor(coords, predictions, remaining_points_idx=NOISE_LABEL_IN_GROUPING, n_neighbors=5):
    predictions = np.copy(predictions)
    assert len(coords) == len(predictions) # input variable should be of same size
    query_idx = np.argwhere(predictions == remaining_points_idx).reshape(-1)
    reference_idx = np.argwhere(predictions != remaining_points_idx).reshape(-1)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=N_CPUS)
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


def save(coords, save_format, save_name, save_folder):
    if save_format == "ply":
        pcd = o3d.geometry.PointCloud()
        if coords.shape[1] == 3:
            pcd.points = o3d.utility.Vector3dVector(coords)
            pcd.colors = o3d.utility.Vector3dVector(np.tile(np.random.rand(1, 3), (len(coords), 1)))
            save_path = osp.join(save_folder, f'{save_name}.ply')
            o3d.io.write_point_cloud(save_path, pcd)
        elif coords.shape[1] == 4:
            preds = coords[:, -1]
            coords = coords[:, :3]
            preds_unique = np.unique(preds)
            num_drawpoints = len(coords)

            n_color_palette = len(preds_unique)
            color_palette = np.random.uniform(size=(n_color_palette, 3))
            # define how preds_unique get mapped to color palette
            color_palette_mapping = {j: i for i, j in enumerate(np.sort(preds_unique))}
            color_palette[-1] = [0,0,0]
            colors = np.empty((num_drawpoints, 3))

            for i in range(num_drawpoints):
                ind = int(preds[i])
                colors[i] = color_palette[color_palette_mapping[ind]]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coords)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            save_path = osp.join(save_folder, f'{save_name}.ply')
            o3d.io.write_point_cloud(save_path, pcd)
    if save_format == "npy":
        save_path = osp.join(save_folder, f'{save_name}.npy')
        np.save(save_path, coords)
    elif save_format == "txt":
        save_path = osp.join(save_folder, f'{save_name}.txt')
        np.savetxt(save_path, coords)


def save_treewise(coords, instance_preds, cluster_means_within_hull, insts_not_at_edge, save_format, plot_results_dir):
    completely_inside_dir = os.path.join(plot_results_dir, 'completely_inside')
    trunk_base_inside_dir = os.path.join(plot_results_dir, 'trunk_base_inside')
    trunk_base_outside_dir = os.path.join(plot_results_dir, 'trunk_base_outside')
    os.makedirs(completely_inside_dir, exist_ok=True)
    os.makedirs(trunk_base_inside_dir, exist_ok=True)
    os.makedirs(trunk_base_outside_dir, exist_ok=True)

    for i in np.unique(instance_preds):
        pred_coord = coords[instance_preds == i]
        if i == UNDERSTORY_LABEL_IN_GROUPING:
            save(pred_coord, save_format, 'understory', plot_results_dir)
            continue

        if cluster_means_within_hull[i] and insts_not_at_edge[i]:
            save(pred_coord, save_format, str(int(i)), completely_inside_dir)
        elif cluster_means_within_hull[i] and not insts_not_at_edge[i]:
            save(pred_coord, save_format, str(int(i)), trunk_base_inside_dir)
        elif not cluster_means_within_hull[i]:
            save(pred_coord, save_format, str(int(i)), trunk_base_outside_dir)
