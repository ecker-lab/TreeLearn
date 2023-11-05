import os 
import os.path as osp
from tqdm import tqdm
import os.path as osp
import numpy as np
import argparse
from tree_learn.util import get_root_logger, get_config, generate_tiles


if __name__ == '__main__':
    parser = argparse.ArgumentParser('tile_generation')
    parser.add_argument('--config', type=str, help='path to config file for tile generation')
    args = parser.parse_args()
    cfg = get_config(args.config)
    logger = get_root_logger(os.path.join(cfg.base_dir, 'log_tile_generation'))
    generate_tiles(cfg.sample_generation, cfg.base_dir, cfg.plot_name, logger)
