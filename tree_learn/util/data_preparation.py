import os
import numpy as np
import torch
import open3d as o3d
from jakteristics import compute_features as compute_features_jakteristics
import pandas as pd
import json
import laspy
from tqdm import tqdm

INSTANCE_LABEL_IGNORE_IN_RAW_DATA = -1 # label for unlabeled in raw data
NON_TREE_CLASS_IN_RAW_DATA = 0 # label for non-trees in raw data


def load_data(path):
    assert path.endswith('npy') or path.endswith('npz') or path.endswith('las') or path.endswith('laz') or path.endswith('txt')
    if path.endswith('npy'):
        data = np.load(path)
    elif path.endswith('npz'):
        data = np.load(path)
        assert 'points' in data
        if 'points' in data and not 'labels' in data:
            data = data['points']
        else:
            data = np.hstack((data["points"], data["labels"][:,np.newaxis]))
    elif path.endswith('.las') or path.endswith('.laz'):
        las_file = laspy.read(path)
        if hasattr(las_file, 'treeID') and hasattr(las_file, 'classification'):
            treeID = np.array(las_file.treeID)
            classes = np.array(las_file.classification)

            tree_mask = treeID != 0
            non_tree_mask = np.isin(classes, [1, 2]) # terrain or vegetation according to For-Instance labeling convention (https://zenodo.org/records/8287792)
            unlabeled_mask = np.logical_not(tree_mask) & np.logical_not(non_tree_mask)
            assert (tree_mask & non_tree_mask & unlabeled_mask).sum() == 0

            points = np.vstack((las_file.x, las_file.y, las_file.z)).T
            points = points - las_file.header.offset
            labels = np.ones(len(points))
            labels[tree_mask] = treeID[tree_mask]
            labels[non_tree_mask] = NON_TREE_CLASS_IN_RAW_DATA
            labels[unlabeled_mask] = INSTANCE_LABEL_IGNORE_IN_RAW_DATA
            data = np.hstack([points, labels[:,np.newaxis]])
        else:
            data = np.vstack((las_file.x, las_file.y, las_file.z)).T
            data = data - las_file.header.offset
    elif path.endswith('txt'):
        data = pd.read_csv(path, delimiter=' ').to_numpy()
    
    assert data.shape[1] == 3 or data.shape[1] == 4
    if data.shape[1] == 3:
        data = np.hstack([data, INSTANCE_LABEL_IGNORE_IN_RAW_DATA * np.ones(len(data))[:,np.newaxis]])
    return data


def voxelize(data, voxel_size):
    points = data[:, :3]
    if data.shape[1] >= 4:
        other = data[:, 3:]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    bound = np.max(np.abs(points)) + 100
    min_bound, max_bound = np.array([-bound, -bound, -bound]), np.array([bound, bound, bound])
    downpcd, _, idx = pcd.voxel_down_sample_and_trace(voxel_size, min_bound, max_bound)

    if data.shape[1] >= 4:
        idx = [item[0] for item in idx]
        other = other[idx]
        data = np.hstack((np.asarray(downpcd.points), other))
    else:
        data = np.asarray(downpcd.points)
    return data


def compute_features(points, search_radius=0.35, feature_names=['verticality'], num_threads=4):
    assert points.shape[1] == 3
    features = compute_features_jakteristics(points, search_radius=search_radius, num_threads=num_threads, feature_names=feature_names)
    features = replace_nanfeatures(features)
    features = features.astype(np.float32)
    return features


def replace_nanfeatures(features):
    ind_nan = np.isnan(features)
    mean_values = np.nanmean(features, axis=0)
    print(f"There are {ind_nan.sum()} nan features in the whole forest. Replacing them with mean feature values")

    for i in range(features.shape[1]):
        if ind_nan[:, i].sum() > 0:
            features[:, i][ind_nan[:, i]] = mean_values[i]
    print(f"After replacement there are {np.isnan(features).sum()} nan features in the whole forest.")
    return features


###################################################################################################################
############################################################################################## CROP/TILE GENERATION
###################################################################################################################


class SampleGenerator:
    def __init__(self, plot_path, features_path, save_dir, n_neigh_sor, 
            multiplier_sor, rad, npoints_rad):

        data = np.load(plot_path)
        data = np.hstack((data["points"], data["labels"][:,np.newaxis]))
        self.feats = np.load(features_path)
        self.feats = self.feats['features']
        self.plot_name = os.path.basename(plot_path)[:-4]
        self.points = data[:, :3]
        self.label = data[:, 3]
        self.x_range, self.y_range = get_ranges(self.points)
        self.x_range = self.x_range[0]
        self.y_range = self.y_range[0]
        self.save_dir_data = os.path.join(save_dir, 'npz')
        self.save_dir_meta_data = os.path.join(save_dir, 'json')
        os.makedirs(self.save_dir_data, exist_ok=True)
        os.makedirs(self.save_dir_meta_data, exist_ok=True)

        # generic generation attributes
        self.n_neigh_sor = n_neigh_sor
        self.multiplier_sor = multiplier_sor
        self.rad = rad
        self.npoints_rad = npoints_rad


    # get occupancy grid
    def get_occupancy_grid(self, occupancy_path, occupancy_res, n_points_to_calculate_occupancy, how_far_fill, min_percent_occupied_fill, ignore_for_occupancy):
        self.occupancy_res = occupancy_res
        self.how_far_fill = how_far_fill
        self.min_percent_occupied_fill = min_percent_occupied_fill
        res = occupancy_res

        if os.path.exists(occupancy_path):
            occupancy_grid = np.load(occupancy_path)
            occupancy_grid = occupancy_grid['occupancy_grid']
            self.occupancy_grid = occupancy_grid
            return

        n = n_points_to_calculate_occupancy
        (x_res, x_dim), (y_res, y_dim) = adjust_res(self.x_range, res), adjust_res(self.y_range, res)
        y_steps = np.arange(self.y_range[0], self.y_range[1]+1e-3, step=y_res)
        x_steps = np.arange(self.x_range[0], self.x_range[1]+1e-3, step=x_res)
        occupancy_grid = np.ones((x_dim, y_dim, 3)) * 10

        mask_valid_points = self.label != ignore_for_occupancy
        points = self.points[mask_valid_points]
        idx = np.random.randint(0, len(points), size=n)
        points = points[idx]


        x_coord, y_coord = points[:,0], points[:,1]

        for i in range(x_dim):
            for j in range(y_dim):
                point_exist = np.any((x_coord > x_steps[i]) & (x_coord <= x_steps[i + 1]) & (y_coord > y_steps[j]) & (y_coord <= y_steps[j + 1]))
                occupancy_grid[i,j, 2] = point_exist
                occupancy_grid[i,j, 0:2] = [np.mean(x_steps[i:i+2]), np.mean(y_steps[j:j+2])]

        occupancy_grid = fill_holes(occupancy_grid, how_far_fill, min_percent_occupied_fill)

        np.savez_compressed(occupancy_path, occupancy_grid=occupancy_grid)
        self.occupancy_grid = occupancy_grid
        return


    # generate candidates
    def generate_candidates(self, n_samples_total, n_samples_plot, chunk_size):
        self.chunk_size = chunk_size
        self.n_samples_plot = n_samples_plot
        # generate enough candidates
        n_candidates = np.max([n_samples_total, 5 * n_samples_plot])
        n_samples_sqrt = int(np.sqrt(n_candidates))

        # generate chunk_centers and rotation_angles
        x_centers = np.linspace(self.x_range[0], self.x_range[1], n_samples_sqrt)
        x_centers = np.repeat(x_centers, n_samples_sqrt)
        x_centers = np.round(x_centers, 2)
        y_centers = np.linspace(self.y_range[0], self.y_range[1], n_samples_sqrt)
        y_centers = np.tile(y_centers, n_samples_sqrt)
        y_centers = np.round(y_centers, 2)
        centers = np.hstack([x_centers.reshape(-1, 1), y_centers.reshape(-1, 1)])
        rotation_angles = np.random.uniform(0, 2*np.pi, size=n_samples_sqrt*n_samples_sqrt)
        rotation_angles = np.round(rotation_angles, 2)

        # generate vertices for candidates
        rotated_vertices = rotate_vertices(rotation_angles, chunk_size)
        shifted_vertices = shift_vertices(rotated_vertices, centers)

        # get ranges of rotated and shifted candidates
        ranges_x, ranges_y = get_ranges(shifted_vertices)
        self.ranges_x = ranges_x
        self.ranges_y = ranges_y
        self.vertices = shifted_vertices
        self.rotation_angles = rotation_angles
        self.centers = centers
        return
    

    # check occupancy of candidates to filter out invalid candidates according to occupancy
    def check_occupancy(self, min_percent_occupied_choose):
        self.min_percent_occupied_choose = min_percent_occupied_choose
        occupancy_grid = self.occupancy_grid.reshape(-1, self.occupancy_grid.shape[-1])

        # We first select a rectangular area A from the occupancy grid
        As = generate_views(occupancy_grid, self.ranges_x, self.ranges_y)

        # We invert rotation and shift on views
        A_invs = [invert_rotate_and_shift(A[:, :2], rotation_angle, center) for A, rotation_angle, center in zip(As, self.rotation_angles, self.centers)]

        # Get occupancy values for all grid elements that have infinity norm <= size/2
        inds_within_proposal = [np.linalg.norm(A_inv, ord=np.inf, axis=1) <= self.chunk_size/2 for A_inv in A_invs]
        occupancies = [A[:, -1][ind_within_proposal] for A, ind_within_proposal in zip(As, inds_within_proposal)]

        # calculate occupancy statistics
        denominator = (self.chunk_size / self.occupancy_res) ** 2
        percent_occupied = [np.sum(occupancy) / denominator for occupancy in occupancies]
        percent_occupied = np.array(percent_occupied)

        filter = percent_occupied > min_percent_occupied_choose
        self.filter = filter
        return
    

    # save
    def save(self, compressed=False):
        # get points
        points = np.hstack((self.points, self.label.reshape(-1, 1)))
        points = np.hstack([points, self.feats])

        # filter valid candidates
        vertices = self.vertices[self.filter]
        rotation_angles = self.rotation_angles[self.filter]
        centers = self.centers[self.filter]

        # only take specified amount of candidates
        if self.n_samples_plot <= len(vertices):
            inds = np.random.choice(range(len(vertices)), self.n_samples_plot, replace=False)
        else:
            inds = np.random.choice(range(len(vertices)), len(vertices), replace=False)
        vertices = vertices[inds]
        rotation_angles = rotation_angles[inds]
        centers = centers[inds]

        # make sure that computational burden is not too high by splitting up the saving process
        n_splits = len(vertices) // 10 if len(vertices) // 10 > 0 else 1
        vertices_list = np.array_split(vertices, n_splits, 0)
        rotation_angles_list = np.array_split(rotation_angles, n_splits, 0)
        centers_list = np.array_split(centers, n_splits, 0)

        chunk_counter = 0
        for vertices, rotation_angles, centers in zip(vertices_list, rotation_angles_list, centers_list):
            # We first select a rectangular area A from the point cloud, that is larger than the rotated rectangle
            ranges_x, ranges_y = get_ranges(vertices)
            As = generate_views(points, ranges_x, ranges_y)

            # We invert rotation and shift on views
            A_invs = [invert_rotate_and_shift(A[:, :2], rotation_angle, center) for A, rotation_angle, center in zip(As, rotation_angles, centers)]

            # Get indices of all points that have infinity norm <= size/2 (these are within the proposed chunk) and use these indices to subset rotated views
            threshold = self.chunk_size/2
            inds_within_proposal = [np.linalg.norm(A_inv, ord=np.inf, axis=1) <= threshold for A_inv in A_invs]
            A_invs = [np.hstack([A_inv, A[:, 2:]]) for A_inv, A in zip(A_invs, As)]
            A_subsets = [A_inv[ind_within_proposal] for A_inv, ind_within_proposal in zip(A_invs, inds_within_proposal)]

            for A_subset, center, rotation_angle in zip(A_subsets, centers, rotation_angles):

                # denoise
                if self.n_neigh_sor is not None and self.multiplier_sor is not None:
                    sor_filter_idx = sor_filter(A_subset, n_neigh_sor=self.n_neigh_sor, multiplier_sor=self.multiplier_sor)
                    A_subset = A_subset[sor_filter_idx]

                if self.rad is not None and self.npoints_rad is not None:
                    rad_filter_idx = rad_filter(A_subset, rad=self.rad, npoints_rad=self.npoints_rad)
                    A_subset = A_subset[rad_filter_idx]

                A_subset = A_subset.astype(np.float32)

                # meta data
                meta_data = dict()
                meta_data['plot_name'] = self.plot_name
                meta_data['rotation_angle'] = rotation_angle
                meta_data['occupancy_res'] = self.occupancy_res
                meta_data['min_percent_occupied_fill'] = self.min_percent_occupied_fill
                meta_data['how_far_fill'] = self.how_far_fill
                meta_data['chunk_size'] = self.chunk_size
                meta_data['min_percent_occupied_choose'] = self.min_percent_occupied_choose
                meta_data['n_neigh_sor'] = self.n_neigh_sor
                meta_data['multiplier_sor'] = self.multiplier_sor
                meta_data['rad'] = self.rad
                meta_data['npoints_rad'] = self.npoints_rad

                # data
                points_save = A_subset[:, :3]
                instance_label_save = A_subset[:, 3]
                feat_save = A_subset[:, 4:]
                center = np.array([center[0], center[1], 0])

                data = dict()
                data['points'] = points_save
                data['feat'] = feat_save
                data['instance_label'] = instance_label_save.astype(np.int32)
                data['center'] = center

                # saving
                save_name_data = self.plot_name + '_' + str(chunk_counter) + '.npz'
                save_path_data = os.path.join(self.save_dir_data, save_name_data)
                save_name_meta_data = self.plot_name + '_' + str(chunk_counter) + '.json'
                save_path_meta_data = os.path.join(self.save_dir_meta_data, save_name_meta_data)
                chunk_counter += 1

                if compressed:
                    np.savez_compressed(save_path_data, **data)
                else:
                    np.savez(save_path_data, **data)
                with open(save_path_meta_data, 'w') as json_file:
                    json.dump(meta_data, json_file)



    def tile_generate_and_save(self, inner_edge, outer_edge, stride, compressed=False, plot_corners=None, logger=None):
        logger.info('defining plot corners')
        if plot_corners is not None:
            plot_corners = np.array(plot_corners)
            # center plot corners and points
            plot_corners_center = np.mean(plot_corners, axis=0)
            plot_corners = plot_corners - plot_corners_center
            self.points = self.points - plot_corners_center
            
            # get angle to rotate points in a way that results in axis aligned plot_corners
            alpha = get_angle_to_align_square_with_axes(plot_corners)

            # align points and plot_corners with axes
            self.points = align_square_with_axes(self.points, alpha)
            plot_corners = align_square_with_axes(plot_corners, alpha)

            # get min max values of plot_corners
            plot_corners_x_range, plot_corners_y_range = get_ranges(plot_corners)
            plot_corners_x_range = plot_corners_x_range[0]
            plot_corners_y_range = plot_corners_y_range[0]

            xmin = plot_corners_x_range[0]
            xmax = plot_corners_x_range[1]
            ymin = plot_corners_y_range[0]
            ymax = plot_corners_y_range[1]

        # if no plot_corners are given, calculate min max values based on ranges directly
        else:
            xmin = np.round(self.x_range[0] - 1.5 * outer_edge, 2)
            xmax = np.round(self.x_range[1] + 1.5 * outer_edge, 2)
            ymin = np.round(self.y_range[0] - 1.5 * outer_edge, 2)
            ymax = np.round(self.y_range[1] + 1.5 * outer_edge, 2)

        logger.info('setting up grid')

        # calculate number of columns based on desired inner_edge length (will not be fulfilled perfectly)
        ncols = int(np.round((xmax - xmin - 2 * outer_edge) / inner_edge))
        inner_edge_x = (xmax - xmin - 2 * outer_edge) / ncols
        inner_edge_x = np.round(inner_edge_x, 5)
        ncols = int((ncols - 1) / stride + 1) # adapt ncols for overlapping predictions

        # calculate number of rows based on desired inner_edge length (will not be fulfilled perfectly)
        nrows = int(np.round((ymax - ymin - 2 * outer_edge) / inner_edge))
        inner_edge_y = (ymax - ymin - 2 * outer_edge) / nrows
        inner_edge_y = np.round(inner_edge_y, 5)
        nrows = int((nrows - 1) / stride + 1) # adapt nrows for overlapping predictions

        inner_square_extension = np.empty((nrows * ncols, 4))
        for i in range(nrows):
            for j in range(ncols):
                inner_square_extension[i * ncols + j] = np.array([xmin + outer_edge + stride * j * inner_edge_x, xmin + outer_edge + (stride * j + 1) * inner_edge_x, \
                                                                   ymax - outer_edge - (stride * i + 1) * inner_edge_y, ymax - outer_edge - stride * i * inner_edge_y])
        inner_square_extension = np.round(inner_square_extension, 5)
        outer_square_extension = inner_square_extension + np.array([-outer_edge, outer_edge, -outer_edge, outer_edge]).reshape(1, 4)

        # add label and feats to points
        points = np.hstack((self.points, self.label.reshape(-1, 1)))
        points = np.hstack([points, self.feats])

        # to cuda for faster performance
        points = torch.from_numpy(points).cuda()
        outer_square_extension = torch.from_numpy(outer_square_extension).cuda()
        x_points = points[:, 0]
        y_points = points[:, 1]


        logger.info('subset all points with outer square extensions')
        chunks = []
        for xmin_outer, xmax_outer, ymin_outer, ymax_outer in outer_square_extension:

            ind_xmin_outer = x_points >= xmin_outer
            ind_xmax_outer = x_points <= xmax_outer
            ind_ymin_outer = y_points >= ymin_outer
            ind_ymax_outer = y_points <= ymax_outer

            ind_outer = ind_xmin_outer & ind_xmax_outer & ind_ymin_outer & ind_ymax_outer
            chunk = points[ind_outer].cpu().numpy()
            chunks.append(chunk)


        logger.info('only select chunks whose inner squares contain points')
        valid_chunks = []
        valid_inner_square_extension = []
        for i in range(len(chunks)):
            x_chunk = chunks[i][:, 0]
            y_chunk = chunks[i][:, 1]

            ind_xmin_inner = x_chunk >= inner_square_extension[i][0]
            ind_xmax_inner = x_chunk < inner_square_extension[i][1]
            ind_ymin_inner = y_chunk > inner_square_extension[i][2]
            ind_ymax_inner = y_chunk <= inner_square_extension[i][3]

            temp_ind_inner = ind_xmin_inner & ind_xmax_inner & ind_ymin_inner & ind_ymax_inner
            
            if len(chunks[i][temp_ind_inner]) > 0:
                valid_chunks.append(chunks[i])
                valid_inner_square_extension.append(inner_square_extension[i])
        del chunks # free memory
        valid_inner_square_extension = np.array(valid_inner_square_extension).astype(np.float32)


        logger.info('center chunks')
        for i in range(len(valid_chunks)):
            chunk_center_x = np.round((valid_inner_square_extension[i][0] + valid_inner_square_extension[i][1]) / 2, 6)
            chunk_center_y = np.round((valid_inner_square_extension[i][2] + valid_inner_square_extension[i][3]) / 2, 6)
            center = np.concatenate([np.array([chunk_center_x, chunk_center_y, 0, 0]), np.zeros(self.feats.shape[1])]).reshape(1, -1)
            valid_chunks[i] = (torch.from_numpy(valid_chunks[i]).cuda() - torch.from_numpy(center).cuda()).cpu().numpy()


        logger.info('denoise')
        for i, valid_chunk in tqdm(enumerate(valid_chunks)):

            # denoise
            if self.n_neigh_sor is not None and self.multiplier_sor is not None:
                sor_filter_idx = sor_filter(valid_chunk, n_neigh_sor=self.n_neigh_sor, multiplier_sor=self.multiplier_sor)
                valid_chunk = valid_chunk[sor_filter_idx]
                valid_chunks[i] = None # free memory

            if self.rad is not None and self.npoints_rad is not None:
                rad_filter_idx = rad_filter(valid_chunk, rad=self.rad, npoints_rad=self.npoints_rad)
                valid_chunk = valid_chunk[rad_filter_idx]
                valid_chunks[i] = None # free memory

            valid_chunk = valid_chunk.astype(np.float32)

            # meta data
            meta_data = dict()
            meta_data['plot_name'] = self.plot_name
            meta_data['n_neigh_sor'] = self.n_neigh_sor
            meta_data['multiplier_sor'] = self.multiplier_sor
            meta_data['rad'] = self.rad
            meta_data['npoints_rad'] = self.npoints_rad
            meta_data['inner_edge'] = inner_edge
            meta_data['outer_edge'] = outer_edge

            # data
            points = valid_chunk[:, :3]
            instance_label = valid_chunk[:, 3]
            feat = valid_chunk[:, 4:]
            chunk_center_x = np.round((valid_inner_square_extension[i][0] + valid_inner_square_extension[i][1]) / 2, 6)
            chunk_center_y = np.round((valid_inner_square_extension[i][2] + valid_inner_square_extension[i][3]) / 2, 6)
            center = np.array([chunk_center_x, chunk_center_y, 0])

            data = dict()
            data['points'] = points
            data['feat'] = feat
            data['instance_label'] = instance_label.astype(np.int32)
            data['center'] = center
            
            # saving
            save_name_data = self.plot_name + '_' + str(i) + '.npz'
            save_path_data = os.path.join(self.save_dir_data, save_name_data)
            save_name_meta_data = self.plot_name + '_' + str(i) + '.json'
            save_path_meta_data = os.path.join(self.save_dir_meta_data, save_name_meta_data)

            if compressed:
                np.savez_compressed(save_path_data, **data)
            else:  
                np.savez(save_path_data, **data)
            with open(save_path_meta_data, 'w') as json_file:
                json.dump(meta_data, json_file)
        return


def get_ranges(points):

    x = points[...,0]
    y = points[...,1]

    xmin = np.min(x, axis=-1)
    xmax = np.max(x, axis=-1)
    ymin = np.min(y, axis=-1)
    ymax = np.max(y, axis=-1)

    rng = (np.hstack([xmin.reshape(-1, 1), xmax.reshape(-1, 1)]), np.hstack([ymin.reshape(-1, 1), ymax.reshape(-1, 1)]))

    return rng


def rotate_vertices(rotation_angles, size):

    base_vertices = np.array([[size/2, size/2], 
                            [size/2, -size/2],
                            [-size/2, -size/2], 
                            [-size/2, size/2]])
    base_vertices = base_vertices[np.newaxis, ...]

    rotation_angles = rotation_angles.reshape(-1, 1)
    cosines = np.cos(rotation_angles)
    sines = np.sin(rotation_angles)

    rotation_matrices = np.hstack([cosines, -sines, sines, cosines])
    rotation_matrices = rotation_matrices.reshape(-1, 2, 2)
    rotation_matrices = np.transpose(rotation_matrices, (0, 2, 1))

    rotated_vertices = base_vertices @ rotation_matrices

    return rotated_vertices


def shift_vertices(rotated_vertices, centers):

    centers = centers.reshape(-1, 1, 2)
    shifted_vertices = rotated_vertices + centers

    return shifted_vertices


def invert_rotate_and_shift(view, rotation_angle, center):

    cosine = np.cos(rotation_angle).item()
    sine = np.sin(rotation_angle).item()

    rotation_matrix = np.array([[cosine, -sine],
                                [sine, cosine]])
    rotation_matrix = np.linalg.inv(rotation_matrix)

    shifted_view = view - center
    rotated_and_shifted_view = shifted_view @ rotation_matrix.T

    return rotated_and_shifted_view


def generate_views(arr, ranges_x, ranges_y):

    # make sure that enough is cut out (be generous)
    x_lower = ranges_x[:, 0] - 3
    x_upper = ranges_x[:, 1] + 3
    y_lower = ranges_y[:, 0] - 3
    y_upper = ranges_y[:, 1] + 3

    filters = (arr[:, 0][np.newaxis, ...] > x_lower[..., np.newaxis]) & \
                (arr[:, 0][np.newaxis, ...] < x_upper[..., np.newaxis]) & \
                (arr[:, 1][np.newaxis, ...] > y_lower[..., np.newaxis]) & \
                (arr[:, 1][np.newaxis, ...] < y_upper[..., np.newaxis])

    views = [arr[filter] for filter in filters]

    return views


def adjust_res(range, res):
    diff = np.abs(range[0] - range[1])
    times_fit = np.floor(diff / res)
    adj_res = diff / times_fit
    return adj_res, times_fit.astype("int")


def fill_holes(grid, how_far_fill, min_percent_occupied_fill):

    grid_new = grid.copy()
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not grid[i, j, 2]:
                lower_row = min(how_far_fill, i)
                upper_row = min(how_far_fill + 1, grid.shape[0] - i)
                lower_col = min(how_far_fill, j)
                upper_col = min(how_far_fill + 1, grid.shape[1] - j)

                view = grid[i-lower_row:i+upper_row, j-lower_col:j+upper_col, 2]
                n_occupied = np.sum(view) 
                percent_occupied = n_occupied / (view.shape[0] * view.shape[1])

                grid_new[i, j, 2] = percent_occupied >= min_percent_occupied_fill
                
    return grid_new


def sor_filter(chunk, n_neigh_sor, multiplier_sor):

    points = chunk[:, :3]
    assert len(points) > 0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # filter out based on stastistical outlier
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=n_neigh_sor,  std_ratio=multiplier_sor)
    pcd = pcd.select_by_index(ind)
    mask = np.zeros(len(points),dtype=bool)
    mask[ind] = True
    return mask


def rad_filter(chunk, rad, npoints_rad):

    points = chunk[:, :3]
    assert len(points) > 0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # filter out isolated points
    _, ind = pcd.remove_radius_outlier(nb_points=npoints_rad, radius=rad)
    pcd = pcd.select_by_index(ind)
    mask = np.zeros(len(points),dtype=bool)
    mask[ind] = True
    return mask


def get_angle_to_align_square_with_axes(corner_points):
        # if corner points already form an axis aligned square return rotation angle of 0
        if len(np.unique(corner_points[:, 0])) != 4:
            alpha = 0
            return alpha
            
        # calculate rotation angle alpha to align arbitrary square given by corner_points with x and y axes
        corner_point1 = corner_points[corner_points[:, 0].argmin()]
        corner_point2 = corner_points[corner_points[:, 1].argmax()]
        edge_vector = corner_point2 - corner_point1

        hypothenuse_length = 0.5 * np.linalg.norm(edge_vector)
        adjacent_length = 0.5 * edge_vector[0]
        alpha = np.arccos(adjacent_length / hypothenuse_length)
        return alpha


def align_square_with_axes(points, angle):
    # rotate back plot_extension and points such that they live in normal coordinate system
    cosine = np.cos(angle)
    sine = np.sin(angle)
    rotation_matrix = np.array([[cosine, -sine, 0],
                                [sine, cosine, 0],
                                [0, 0, 1]])

    inv_rotation_matrix = np.linalg.inv(rotation_matrix)
    inv_rotation_matrix = inv_rotation_matrix.T

    points = points @ inv_rotation_matrix
    return points
    