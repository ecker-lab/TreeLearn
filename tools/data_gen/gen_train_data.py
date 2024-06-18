import os 
import os.path as osp
from tqdm import tqdm
import os.path as osp
import numpy as np
import argparse
from tree_learn.util import SampleGenerator, get_root_logger, get_config, voxelize, compute_features, load_data

INSTANCE_LABEL_IGNORE_IN_RAW_DATA = -1 # instance label in raw data to ignore during crop generation since it represent unlabeled data
N_JOBS = 10 # number of threads for feature calculations




def generate_random_crops(cfg):
    # logger
    documentation_dir = os.path.join(cfg.base_dir, 'documentation')
    os.makedirs(documentation_dir, exist_ok=True)
    logger = get_root_logger(os.path.join(documentation_dir, 'log_random_crop_generation'))

    # make dirs for data saving
    forests_dir = osp.join(cfg.base_dir, 'forests')
    voxelized_dir = osp.join(cfg.base_dir, f'forests_voxelized{cfg.sample_generation.voxel_size}')
    features_dir = osp.join(cfg.base_dir, 'features')
    occupancy_dir = osp.join(cfg.base_dir, 'occupancy')
    save_dir = osp.join(cfg.base_dir, 'random_crops')
    os.makedirs(voxelized_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(occupancy_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # voxelize forests
    logger.info('voxelizing forests...')
    for plot_file in tqdm(os.listdir(forests_dir)):
        plot_name = plot_file[:-4]
        save_path_voxelized = osp.join(voxelized_dir, f'{plot_name}.npz')
        if osp.exists(save_path_voxelized):
            continue
        data = load_data(osp.join(forests_dir, plot_file))
        data, _ = voxelize(data, cfg.sample_generation.voxel_size)
        data = np.round(data, 2)
        data = data.astype(np.float32)
        np.savez_compressed(save_path_voxelized, points=data[:, :3], labels=data[:, 3])
    
    # calculate features
    logger.info('calculating features...')
    for plot_file in tqdm(os.listdir(voxelized_dir)):
        plot_name = plot_file[:-4]
        save_path_features = osp.join(features_dir, f'{plot_name}.npz')
        if osp.exists(save_path_features):
            continue
        data = load_data(osp.join(voxelized_dir, plot_file))
        features = compute_features(points=data[:, :3].astype(np.float64), search_radius=cfg.sample_generation.search_radius_features, feature_names=['verticality'], num_threads=N_JOBS)
        np.savez_compressed(save_path_features, features=features)

    # get occupancy map and number of occupied locations
    logger.info('calculating occupancy...')
    n_occupied_locations = dict()
    for plot_file in tqdm(os.listdir(voxelized_dir)):
        # change cfg args according to current plot
        cfg.sample_generation.sample_generator.plot_path = osp.join(voxelized_dir, plot_file)
        cfg.sample_generation.sample_generator.features_path = osp.join(features_dir, plot_file)
        cfg.sample_generation.sample_generator.save_dir = save_dir
        obj = SampleGenerator(**cfg.sample_generation.sample_generator)

        occupancy_path = osp.join(occupancy_dir, plot_file)
        obj.get_occupancy_grid(occupancy_path, cfg.occupancy_res, cfg.n_points_to_calculate_occupancy, cfg.how_far_fill, cfg.min_percent_occupied_fill, ignore_for_occupancy=INSTANCE_LABEL_IGNORE_IN_RAW_DATA)
        n_occupied_locations[plot_file.replace('.npz', '')] = np.sum(obj.occupancy_grid[:, :, 2])

    # get n_samples
    n_samples = dict()
    n_occupied_total = sum(n_occupied_locations.values())
    for plot in n_occupied_locations:
        n_samples[plot] = int(np.round((n_occupied_locations[plot] / n_occupied_total) * cfg.n_samples_total))
    if not sum(n_samples.values()) == cfg.n_samples_total:
        n_samples[plot] = int(n_samples[plot] + (cfg.n_samples_total - sum(n_samples.values())))

    # generate training examples
    logger.info('getting chunks...')
    for plot_file in tqdm(os.listdir(voxelized_dir)): 
        # change cfg args according to current plot
        cfg.sample_generation.sample_generator.plot_path = osp.join(voxelized_dir, plot_file)
        cfg.sample_generation.sample_generator.features_path = osp.join(features_dir, plot_file)
        cfg.sample_generation.sample_generator.save_dir = save_dir
        obj = SampleGenerator(**cfg.sample_generation.sample_generator)
        
        # RANDOM CROP GENERATION
        occupancy_path = osp.join(occupancy_dir, plot_file)
        n_samples_plot = n_samples[plot_file.replace('.npz', '')]
        # get occupancy grid
        obj.get_occupancy_grid(occupancy_path, cfg.occupancy_res, cfg.n_points_to_calculate_occupancy, cfg.how_far_fill, cfg.min_percent_occupied_fill, ignore_for_occupancy=INSTANCE_LABEL_IGNORE_IN_RAW_DATA)
        # generate candidates for random crops
        obj.generate_candidates(cfg.n_samples_total, n_samples_plot, cfg.chunk_size)
        # check whether candidates have sufficiently high occupancy
        obj.check_occupancy(cfg.min_percent_occupied_choose)
        # save
        obj.save()




if __name__ == '__main__':
    parser = argparse.ArgumentParser('random_crop_generation')
    parser.add_argument('--config', type=str, help='path to config file for random crop generation')
    args = parser.parse_args()
    cfg = get_config(args.config)
    generate_random_crops(cfg)
