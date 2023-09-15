import functools
import os
import torch
from collections import OrderedDict
from timm.scheduler import CosineLRScheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader


def is_multiple(num, multiple):
    return num != 0 and num % multiple == 0


def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.
    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu


def cuda_cast(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        new_args = []
        for x in args:
            if isinstance(x, torch.Tensor):
                x = x.cuda()
            new_args.append(x)
        new_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.cuda()
            new_kwargs[k] = v
        return func(*new_args, **new_kwargs)

    return wrapper


def checkpoint_save(epoch, model, optimizer, work_dir, save_freq=16):
    if hasattr(model, 'module'):
        model = model.module
    f = os.path.join(work_dir, f'epoch_{epoch}.pth')
    checkpoint = {
        'net': weights_to_cpu(model.state_dict()),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, f)

    # remove previous checkpoints unless they are a power of 2 or a multiple of save_freq
    epoch = epoch - 1
    f = os.path.join(work_dir, f'epoch_{epoch}.pth')
    if os.path.isfile(f):
        if not is_multiple(epoch, save_freq):
            os.remove(f)


def load_checkpoint(checkpoint, logger, model, optimizer=None, strict=False):
    if hasattr(model, 'module'):
        model = model.module
    device = torch.cuda.current_device()
    state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage.cuda(device))
    # if not (torch.equal(torch.tensor(state_dict['net']['input_conv.0.weight'].shape), torch.tensor([32, 3, 3, 3, 7]))):
    #     state_dict['net']['input_conv.0.weight'] = torch.permute(state_dict['net']['input_conv.0.weight'], (3, 0, 1, 2, 4))
    src_state_dict = state_dict['net']
    target_state_dict = model.state_dict()
    skip_keys = []
    # skip mismatch size tensors in case of pretraining
    for k in src_state_dict.keys():
        if k not in target_state_dict:
            continue
        if src_state_dict[k].size() != target_state_dict[k].size():
            skip_keys.append(k)
    for k in skip_keys:
        del src_state_dict[k]
    missing_keys, unexpected_keys = model.load_state_dict(src_state_dict, strict=strict)
    if skip_keys:
        logger.info(
            f'removed keys in source state_dict due to size mismatch: {", ".join(skip_keys)}')
    if missing_keys:
        logger.info(f'missing keys in source state_dict: {", ".join(missing_keys)}')
    if unexpected_keys:
        logger.info(f'unexpected key in source state_dict: {", ".join(unexpected_keys)}')

    # load optimizer
    if optimizer is not None:
        assert 'optimizer' in state_dict
        optimizer.load_state_dict(state_dict['optimizer'])

    if 'epoch' in state_dict:
        epoch = state_dict['epoch']
    else:
        epoch = 0
    return epoch + 1


def build_optimizer(model, optim_cfg):
    assert 'type' in optim_cfg
    _optim_cfg = optim_cfg.copy()
    optim_type = _optim_cfg.pop('type')
    optim = getattr(torch.optim, optim_type)
    return optim(filter(lambda p: p.requires_grad, model.parameters()), **_optim_cfg)


def build_cosine_scheduler(cfg, optimizer):
    scheduler = CosineLRScheduler(optimizer,
                t_initial=cfg.t_initial,
                lr_min=cfg.lr_min,
                cycle_decay=cfg.cycle_decay,
                warmup_lr_init=cfg.warmup_lr_init,
                warmup_t=cfg.warmup_t,
                cycle_limit=cfg.cycle_limit,
                t_in_epochs=cfg.t_in_epochs)
    return scheduler


def build_dataloader(dataset, batch_size=1, num_workers=1, training=True):
    shuffle = training
    sampler = None
    
    if sampler is not None:
        shuffle = False
    if training:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=True,
            pin_memory=True)
    else:
        # assert batch_size == 1
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
            shuffle=False,
            sampler=sampler,
            drop_last=False,
            pin_memory=True)


@cuda_cast
def point_wise_loss(semantic_prediction_logits, offset_predictions, semantic_labels,
                    offset_labels):

    if len(semantic_prediction_logits) == 0:
        semantic_loss = 0 * semantic_labels.sum()
    else:
        semantic_loss = F.cross_entropy(
            semantic_prediction_logits, semantic_labels, reduction='sum') / len(semantic_prediction_logits)
        
    if len(offset_predictions) == 0:
        offset_loss = 0 * offset_predictions.sum()
    else:
        offset_loss = F.l1_loss(offset_predictions, offset_labels, reduction='none').sum() / len(offset_predictions)
    
    return semantic_loss, offset_loss


def get_voxel_sizes(batch, config):
    # get voxel_sizes to use in forward
    if config.mode == 'pointwise':
        voxel_sizes = torch.ones((batch['batch_size'], 3)) * config.voxel_size
    elif config.mode == 'classifier' or config.mode == "lnet":
        voxel_sizes = calculate_voxel_sizes(batch['coords'], batch['batch_ids'], config.n_voxels_in_each_direction)
    voxel_sizes = [row.tolist() for row in voxel_sizes]
    return voxel_sizes


def calculate_voxel_sizes(coords, batch_ids, n_voxels_in_each_direction):
    unique_batch_ids = torch.unique(batch_ids)
    voxel_sizes = torch.zeros((len(unique_batch_ids), 3), dtype=torch.float)

    for i in unique_batch_ids:
        instance_coords = coords[batch_ids == i]
        min_coords, _ = torch.min(instance_coords, dim=0)
        max_coords, _ = torch.max(instance_coords, dim=0)

        voxel_sizes[i] = (max_coords - min_coords + 0.001) / n_voxels_in_each_direction

    return voxel_sizes


@cuda_cast
def get_cls_loss(pred_cls_logits, target_cls, weight):
    cls_loss = F.cross_entropy(pred_cls_logits, target_cls, weight)
    return cls_loss