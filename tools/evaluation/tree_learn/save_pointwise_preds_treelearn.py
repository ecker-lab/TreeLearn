import os
import argparse
import munch
import yaml
import numpy as np
import shutil
from tree_learn.dataset import TreeDataset
from tree_learn.model import TreeLearn
from tree_learn.util import load_checkpoint, ensemble, get_root_logger, get_pointwise_preds, get_config, build_dataloader


def main():
    parser = argparse.ArgumentParser('tree_learn')
    parser.add_argument('--config', type=str, help='path to config file for saving pointwise predictions')
    args = parser.parse_args()
    config = get_config(args.config)
    pointwise_data_dir = os.path.join(config.base_dir, 'pointwise_data')
    documentation_dir = os.path.join(config.base_dir, 'documentation')
    os.makedirs(pointwise_data_dir, exist_ok=True)
    os.makedirs(documentation_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(documentation_dir, os.path.basename(args.config)))
    logger = get_root_logger(os.path.join(documentation_dir, 'save_pointwise_preds_log.txt'))

    model = TreeLearn(**config.model).cuda()
    dataset = TreeDataset(**config.dataset_test, logger=logger)
    dataloader = build_dataloader(dataset, training=False, **config.dataloader)
    
    # y_greater_zero
    logger.info('getting predictions for y_greater_zero')
    load_checkpoint(config.path_checkpoint_to_use_for_y_greater_zero, logger, model)
    semantic_prediction_logits, semantic_labels, offset_predictions, offset_labels, coords, instance_labels, feats = get_pointwise_preds(model, dataloader, config.model)
    data = ensemble(coords, semantic_prediction_logits, semantic_labels, offset_predictions, 
                    offset_labels, instance_labels, feats)
    coords, semantic_prediction_logits, semantic_labels, offset_predictions, offset_labels, instance_labels, feats = data

    mask_retain = coords[:, 1] > 0
    semantic_prediction_logits_y_greater_zero, semantic_labels_y_greater_zero = semantic_prediction_logits[mask_retain], semantic_labels[mask_retain]
    offset_predictions_y_greater_zero, offset_labels_y_greater_zero = offset_predictions[mask_retain], offset_labels[mask_retain]
    coords_y_greater_zero, instance_labels_y_greater_zero, feats_y_greater_zero = coords[mask_retain], instance_labels[mask_retain], feats[mask_retain]

    # y_not_greater_zero
    logger.info('getting predictions for y_not_greater_zero')
    load_checkpoint(config.path_checkpoint_to_use_for_y_not_greater_zero, logger, model)
    semantic_prediction_logits, semantic_labels, offset_predictions, offset_labels, coords, instance_labels, feats = get_pointwise_preds(model, dataloader, config.model)
    data = ensemble(coords, semantic_prediction_logits, semantic_labels, offset_predictions, 
                    offset_labels, instance_labels, feats)
    coords, semantic_prediction_logits, semantic_labels, offset_predictions, offset_labels, instance_labels, feats = data

    mask_retain = coords[:, 1] <= 0
    semantic_prediction_logits_y_not_greater_zero, semantic_labels_y_not_greater_zero = semantic_prediction_logits[mask_retain], semantic_labels[mask_retain]
    offset_predictions_y_not_greater_zero, offset_labels_y_not_greater_zero = offset_predictions[mask_retain], offset_labels[mask_retain]
    coords_y_not_greater_zero, instance_labels_y_not_greater_zero, feats_y_not_greater_zero = coords[mask_retain], instance_labels[mask_retain], feats[mask_retain]

    # stitch results together
    logger.info('stitching predictions together and saving')
    semantic_prediction_logits = np.concatenate([semantic_prediction_logits_y_greater_zero, semantic_prediction_logits_y_not_greater_zero], 0)
    semantic_labels = np.concatenate([semantic_labels_y_greater_zero, semantic_labels_y_not_greater_zero], 0)
    offset_predictions = np.concatenate([offset_predictions_y_greater_zero, offset_predictions_y_not_greater_zero], 0)
    offset_labels = np.concatenate([offset_labels_y_greater_zero, offset_labels_y_not_greater_zero], 0)
    coords = np.concatenate([coords_y_greater_zero, coords_y_not_greater_zero], 0)
    instance_labels = np.concatenate([instance_labels_y_greater_zero, instance_labels_y_not_greater_zero], 0)
    feats = np.concatenate([feats_y_greater_zero, feats_y_not_greater_zero], 0)

    # save results
    np.save(os.path.join(pointwise_data_dir, 'semantic_prediction_logits.npy'), semantic_prediction_logits)
    np.save(os.path.join(pointwise_data_dir, 'semantic_labels.npy'), semantic_labels)
    np.save(os.path.join(pointwise_data_dir, 'offset_predictions.npy'), offset_predictions)
    np.save(os.path.join(pointwise_data_dir, 'offset_labels.npy'), offset_labels)
    np.save(os.path.join(pointwise_data_dir, 'coords.npy'), coords)
    np.save(os.path.join(pointwise_data_dir, 'instance_labels.npy'), instance_labels)
    np.save(os.path.join(pointwise_data_dir, 'feats.npy'), feats)





if __name__ == '__main__':
     main()
