import os.path as osp
import time
import torch
import tqdm
import numpy as np
import time
from collections import defaultdict
from tree_learn.util import (checkpoint_save, init_train_logger, load_checkpoint,
                            is_multiple, get_args_and_cfg, build_cosine_scheduler, build_optimizer,
                            point_wise_loss, get_eval_components, build_dataloader)
from tree_learn.model import TreeLearn
from tree_learn.dataset import TreeDataset

TREE_CLASS_IN_DATASET = 0 # semantic label for tree class in pytorch dataset
NON_TREE_CLASS_IN_DATASET = 1 # semantic label for non-tree class in pytorch dataset
TREE_CONF_THRESHOLD = 0.5 # minimum confidence for tree prediction


def train(config, epoch, model, optimizer, scheduler, scaler, train_loader, logger, writer):
    model.train()
    start = time.time()
    losses_dict = defaultdict(list)


    for i, batch in enumerate(train_loader, start=1):
        # break after a fixed number of samples have been passed
        if config.examples_per_epoch < (i * config.dataloader.train.batch_size):
            break
        
        scheduler.step(epoch)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=config.fp16):

            # forward
            loss, loss_dict = model(batch, return_loss=True)
            for key, value in loss_dict.items():
                losses_dict[key].append(value.detach().cpu().item())

        # backward
        scaler.scale(loss).backward()
        if config.grad_norm_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip, norm_type=2)
        scaler.step(optimizer)
        scaler.update()

    # log and write to tensorboard
    epoch_time = time.time() - start
    lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('train/learning_rate', lr, epoch)
    average_losses_dict = {k: sum(v) / len(v) for k, v in losses_dict.items()}
    for k, v in average_losses_dict.items():
        writer.add_scalar(f'train/{k}', v, epoch)

    log_str = f'[TRAINING] [{epoch}/{config.epochs}], time {epoch_time:.2f}s'
    for k, v in average_losses_dict.items():
        log_str += f', {k}: {v:.2f}'
    logger.info(log_str)
    checkpoint_save(epoch, model, optimizer, config.work_dir, config.save_frequency)
    

def validate(config, epoch, model, val_loader, logger, writer):  
    with torch.no_grad():
        model.eval()
        semantic_prediction_logits, offset_predictions, semantic_labels, offset_labels, coords, instance_labels = [], [], [], [], [], []
        for batch in tqdm.tqdm(val_loader):

            # forward
            output = model(batch, return_loss=False)
            offset_prediction, semantic_prediction_logit = output['offset_predictions'], output['semantic_prediction_logits']

            batch['coords'] = batch['coords'] + batch['centers']
            semantic_prediction_logits.append(semantic_prediction_logit[batch['masks_sem']])
            semantic_labels.append(batch['semantic_labels'][batch['masks_sem']])
            offset_predictions.append(offset_prediction[batch['masks_sem']])
            offset_labels.append(batch['offset_labels'][batch['masks_sem']])
            coords.append(batch['coords'][batch['masks_sem']]), 
            instance_labels.append(batch['instance_labels'][batch['masks_sem']])

    # concatenate all batches
    semantic_prediction_logits, semantic_labels = torch.cat(semantic_prediction_logits, 0), torch.cat(semantic_labels, 0)
    offset_predictions, offset_labels = torch.cat(offset_predictions, 0), torch.cat(offset_labels, 0)
    coords, instance_labels = torch.cat(coords, 0), torch.cat(instance_labels).cpu().numpy()

    # evaluate semantic and offset predictions
    pointwise_eval(semantic_prediction_logits, offset_predictions, semantic_labels, offset_labels,
                          config, epoch, writer, logger)


def pointwise_eval(semantic_prediction_logits, offset_predictions, semantic_labels, offset_labels, config, epoch, writer, logger):
    # get offset loss
    masks_sem = torch.ones_like(semantic_labels).bool()
    masks_off = semantic_labels == TREE_CLASS_IN_DATASET
    _, offset_loss = point_wise_loss(semantic_prediction_logits.float(), offset_predictions.float(), 
                                      masks_sem, masks_off, semantic_labels, offset_labels)
    
    # get semantic accuracy of classification into tree and non-tree
    semantic_prediction_logits, semantic_labels = semantic_prediction_logits.cpu().numpy(), semantic_labels.cpu().numpy()
    tree_pred_mask = torch.from_numpy(semantic_prediction_logits).float().softmax(dim=-1)[:, TREE_CLASS_IN_DATASET] >= TREE_CONF_THRESHOLD
    tree_pred_mask = tree_pred_mask.numpy()
    tree_mask = semantic_labels == TREE_CLASS_IN_DATASET
    tp, fp, tn, fn = get_eval_components(tree_pred_mask, tree_mask)
    acc = (tp + tn) / (tp + fp + fn + tn)

    # log and write to tensorboard
    logger.info(f'[VALIDATION] [{epoch}/{config.epochs}] val/semantic_acc {acc*100:.2f}, val/offset_loss {offset_loss.item():.3f}')
    writer.add_scalar(f'val/acc', acc if not np.isnan(acc) else 0, epoch)
    writer.add_scalar(f'val/Offset_MAE', offset_loss, epoch)


def main():
    args, config = get_args_and_cfg()
    logger, writer = init_train_logger(config, args)

    # training objects
    model = TreeLearn(**config.model).cuda()
    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_cosine_scheduler(config.scheduler, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)
    train_set = TreeDataset(**config.dataset_train, logger=logger)
    val_set = TreeDataset(**config.dataset_test, logger=logger)
    train_loader = build_dataloader(train_set, training=True, **config.dataloader.train)
    val_loader = build_dataloader(val_set, training=False, **config.dataloader.test)
    
    # optionally pretrain or resume
    start_epoch = 1
    if args.resume:
        logger.info(f'Resume from {args.resume}')
        start_epoch = load_checkpoint(args.resume, logger, model, optimizer=optimizer)
    elif config.pretrain:
        logger.info(f'Load pretrain from {config.pretrain}')
        load_checkpoint(config.pretrain, logger, model)

    # train and val
    logger.info('Training')
    for epoch in range(start_epoch, config.epochs + 1):
        train(config, epoch, model, optimizer, scheduler, scaler, train_loader, logger, writer)
        if is_multiple(epoch, config.validation_frequency):
            optimizer.zero_grad()
            logger.info('Validation')
            torch.cuda.empty_cache()
            validate(config, epoch, model, val_loader, logger, writer)
        writer.flush()


if __name__ == '__main__':
    main()