import os 
import os.path as osp
import argparse
from tree_learn.util import get_root_logger, get_config, generate_tiles


if __name__ == '__main__':
    parser = argparse.ArgumentParser('tile_generation')
    parser.add_argument('--config', type=str, help='path to config file for tile generation')
    args = parser.parse_args()
    cfg = get_config(args.config)
    base_dir = os.path.dirname(os.path.dirname(cfg.forest_path))
    logger = get_root_logger(os.path.join(base_dir, 'log_tile_generation'))
    generate_tiles(cfg.sample_generation, cfg.forest_path, logger)
