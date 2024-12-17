import math
import numpy as np
import torch
import os
from torch.utils.data import Dataset

INSTANCE_LABEL_IGNORE_IN_RAW_DATA = -1 # label for unlabeled in raw data
NON_TREE_CLASS_IN_RAW_DATA = 0 # label for non-trees in raw data
NON_TREE_CLASS_IN_PYTORCH_DATASET = 1 # semantic label for non-tree in pytorch dataset
TREE_CLASS_IN_PYTORCH_DATASET = 0 # semantic label for tree in pytorch dataset


class TreeDataset(Dataset):
    def __init__(self,
                 data_root,
                 inner_square_edge_length,
                 training,
                 logger,
                 data_augmentations=None):

        self.data_paths = [os.path.join(data_root, path) for path in os.listdir(data_root)]
        self.inner_square_edge_length = inner_square_edge_length
        self.logger = logger
        self.training = training
        self.data_augmentations = data_augmentations
        mode = 'train' if training else 'test'
        self.logger.info(f'Load {mode} dataset: {len(self.data_paths)} scans')


    def __len__(self):
        return len(self.data_paths)


    def __getitem__(self, index):
        # load data
        data_path = self.data_paths[index]
        data = np.load(data_path)
        
        # get entries
        xyz = data['points']
        input_feat = data['feat']

        instance_label = data['instance_label']
        semantic_label = np.empty(len(instance_label))
        semantic_label[instance_label == NON_TREE_CLASS_IN_RAW_DATA] = NON_TREE_CLASS_IN_PYTORCH_DATASET
        semantic_label[instance_label != NON_TREE_CLASS_IN_RAW_DATA] = TREE_CLASS_IN_PYTORCH_DATASET

        # get center of chunk (used for stitching tiles back together)
        if self.training:
            center = np.ones_like(xyz) # dummy value in training
        else:
            center = np.ones_like(xyz) * data['center']

        # transform data
        xyz = self.transform_train(xyz) if self.training else self.transform_test(xyz)

        # get offset
        pt_offset_label, mask_valid_offset = self.getOffset(xyz, instance_label, semantic_label)
        
        # get masks for loss calculation
        mask_inner = self.get_mask_inner(xyz)
        mask_not_ignore = self.get_mask_not_ignore(instance_label)
        mask_off = mask_inner & mask_not_ignore & (semantic_label != NON_TREE_CLASS_IN_PYTORCH_DATASET) & mask_valid_offset
        mask_sem = mask_inner & mask_not_ignore

        xyz = torch.from_numpy(xyz)
        instance_label = torch.from_numpy(instance_label)
        semantic_label = torch.from_numpy(semantic_label)
        mask_inner = torch.from_numpy(mask_inner)
        mask_off = torch.from_numpy(mask_off)
        mask_sem = torch.from_numpy(mask_sem)
        pt_offset_label = torch.from_numpy(pt_offset_label)
        input_feat = torch.from_numpy(input_feat)
        center = torch.from_numpy(center)

        return xyz, input_feat, instance_label, semantic_label, pt_offset_label, center, mask_inner, mask_off, mask_sem


    def get_mask_not_ignore(self, instance_label):
        mask_ignore = instance_label == INSTANCE_LABEL_IGNORE_IN_RAW_DATA
        mask_not_ignore = np.logical_not(mask_ignore)
        return mask_not_ignore


    def get_mask_inner(self, xyz):
        # mask of inner square
        inf_norm = np.linalg.norm(xyz[:, :-1], ord=np.inf, axis=1)
        mask_inner = inf_norm <= (self.inner_square_edge_length/2)
        return mask_inner


    def point_jitter(self, points, sigma=0.1, clip=0.2):
        jitter = np.clip(sigma * np.random.randn(points.shape[0], 3), -1 * clip, clip)
        points += jitter
        return points


    def transform_train(self, xyz, aug_prob=0.5, aug_prob_point_jitter=0.25):
        if self.data_augmentations["point_jitter"] == True:
            if np.random.random() <= aug_prob_point_jitter:
                xyz = self.point_jitter(xyz)
        xyz = self.dataAugment(xyz, data_augmentations=self.data_augmentations, prob=aug_prob)
        return xyz


    def transform_test(self, xyz):
        return xyz


    # unlike stated in the paper, we simply use the mean of the lowest 0.5m of the tree points as the tree base here
    def getOffset(self, xyz, instance_label, semantic_label):
        position = np.ones_like(xyz, dtype=np.float32)
        instances = np.unique(instance_label)
        mask_valid_offset = np.zeros_like(instance_label, dtype=bool)

        for instance in instances:
            inst_idx = np.where(instance_label == instance)
            first_idx = inst_idx[0][0]

            if semantic_label[first_idx] != NON_TREE_CLASS_IN_PYTORCH_DATASET:
                tree_points = xyz[inst_idx]
                if len(tree_points[:, 2]) > 11:
                    min_z = np.partition(tree_points[:, 2], 10)[3] # select 3rd lowest point as regualrization to avoid outliers
                else:
                    min_z = tree_points[:, 2].min()

                z_thresh_upper = min_z + 0.5
                mask_thres_upper = tree_points[:, 2] <= z_thresh_upper

                tree_points_of_interest = tree_points[mask_thres_upper]
                if len(tree_points_of_interest) > 0:
                    position_instance = np.mean(tree_points_of_interest, axis=0)
                    mask_valid_offset[inst_idx] = True
                else:
                    position_instance = np.array([0, 0, 0])

                position[inst_idx] = position_instance

        pt_offset_label = position - xyz
        return pt_offset_label, mask_valid_offset


    def dataAugment(self, xyz, data_augmentations, prob=0.6):
        jitter = data_augmentations["jitter"]
        flip = data_augmentations["flip"] 
        rot = data_augmentations["rot"]
        scale = data_augmentations["scaled"]
        m = np.eye(3)

        if scale and np.random.rand() < prob:
            scale_xy = np.random.uniform(0.8, 1.2, 2)
            scale_z = np.random.uniform(0.95, 1.05, 1)
            scale = np.concatenate([scale_xy, scale_z])
            m = m * scale
        if jitter and np.random.rand() < prob:
            m += np.random.randn(3, 3) * 0.1
        if flip and np.random.rand() < prob:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1
        if rot and np.random.rand() < prob:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0],
                              [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])

        return np.matmul(xyz, m)


    def collate_fn(self, batch):
        xyzs = []
        input_feats = []
        batch_ids = []
        instance_labels = []
        semantic_labels = []
        pt_offset_labels = []
        centers = []
        masks_inner = []
        masks_off = []
        masks_sem = []

        total_points_num = 0
        batch_id = 0


        for data in batch:
            xyz, input_feat, instance_label, semantic_label, pt_offset_label, center, mask_inner, mask_off, mask_sem = data
            total_points_num += len(xyz)

            xyzs.append(xyz)
            input_feats.append(input_feat)
            batch_ids.append(torch.ones(len(xyz))*batch_id)
            semantic_labels.append(semantic_label)
            instance_labels.append(instance_label)
            masks_inner.append(mask_inner)
            masks_off.append(mask_off)
            masks_sem.append(mask_sem)
            pt_offset_labels.append(pt_offset_label)
            centers.append(center)           
            batch_id += 1
            
        assert batch_id > 0, 'empty batch'
        if batch_id < len(batch):
            self.logger.info(f'batch is truncated from size {len(batch)} to {batch_id}')

        xyzs = torch.cat(xyzs, 0).to(torch.float32)
        input_feats = torch.cat(input_feats, 0).to(torch.float32)
        batch_ids = torch.cat(batch_ids, 0).long()
        semantic_labels = torch.cat(semantic_labels, 0).long()
        instance_labels = torch.cat(instance_labels, 0).long()
        masks_inner = torch.cat(masks_inner, 0).bool()
        masks_off = torch.cat(masks_off, 0).bool()
        masks_sem = torch.cat(masks_sem, 0).bool()
        pt_offset_labels = torch.cat(pt_offset_labels, 0).float()
        centers = torch.cat(centers, 0).float()
        
        return {
            'coords': xyzs,
            'input_feats': input_feats,
            'batch_ids': batch_ids,
            'semantic_labels': semantic_labels,
            'instance_labels': instance_labels,
            'masks_inner': masks_inner,
            'masks_off': masks_off,
            'masks_sem': masks_sem,
            'offset_labels': pt_offset_labels,
            'batch_size': batch_id,
            'centers': centers
        }
    