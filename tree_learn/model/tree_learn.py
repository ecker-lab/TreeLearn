import functools
import spconv.pytorch as spconv
import torch
import torch.nn as nn
from spconv.pytorch.utils import PointToVoxel
from .blocks import MLP, ResidualBlock, UBlock
from tree_learn.util.train import cuda_cast, point_wise_loss

LOSS_MULTIPLIER_SEMANTIC = 50 # multiply semantic loss for similar magnitude with offset loss

class TreeLearn(nn.Module):
    def __init__(self,
                 channels=32,
                 num_blocks=7,
                 kernel_size=3,
                 dim_coord=3,
                 dim_feat=1,
                 fixed_modules=[],
                 use_feats=True,
                 use_coords=False,
                 spatial_shape=None,
                 max_num_points_per_voxel=3,
                 voxel_size=0.1,
                 **kwargs):

        super().__init__()
        self.voxel_size = voxel_size
        self.fixed_modules = fixed_modules
        self.use_feats = use_feats
        self.use_coords = use_coords
        self.spatial_shape = spatial_shape
        self.max_num_points_per_voxel = max_num_points_per_voxel

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        
        # backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                dim_coord + dim_feat, channels, kernel_size=kernel_size, padding=1, bias=False, indice_key='subm1'))
        block_channels = [channels * (i + 1) for i in range(num_blocks)]
        self.unet = UBlock(block_channels, norm_fn, 2, ResidualBlock, kernel_size, indice_key_id=1)
        self.output_layer = spconv.SparseSequential(norm_fn(channels), nn.ReLU())
        
        # head
        self.semantic_linear = MLP(channels, 2, norm_fn=norm_fn, num_layers=2)
        self.offset_linear = MLP(channels, 3, norm_fn=norm_fn, num_layers=2)
        self.init_weights()

        # weight init
        for mod in fixed_modules:
            mod = getattr(self, mod)
            for param in mod.parameters():
                param.requires_grad = False


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MLP):
                m.init_weights()


    # manually set batchnorms in fixed modules to eval mode
    def train(self, mode=True):
        super().train(mode)
        for mod in self.fixed_modules:
            mod = getattr(self, mod)
            for m in mod.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()


    def forward(self, batch, return_loss):
        backbone_output, v2p_map = self.forward_backbone(**batch)
        output = self.forward_head(backbone_output, v2p_map)
        if return_loss:
            output = self.get_loss(model_output=output, **batch)
        
        return output

    @cuda_cast
    def forward_backbone(self, coords, input_feats, batch_ids, batch_size, **kwargs):
        voxel_feats, voxel_coords, v2p_map, spatial_shape = voxelize(torch.hstack([coords, input_feats]), batch_ids, batch_size, self.voxel_size, self.use_coords, self.use_feats, max_num_points_per_voxel=self.max_num_points_per_voxel)
        if self.spatial_shape is not None:
            spatial_shape = torch.tensor(self.spatial_shape, device=voxel_coords.device)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        output = self.input_conv(input)

        output = self.unet(output)
        output = self.output_layer(output)
        return output, v2p_map
    

    def forward_head(self, backbone_output, v2p_map):
        output = dict()
        backbone_feats = backbone_output.features[v2p_map]
        output['backbone_feats'] = backbone_feats
        output['semantic_prediction_logits'] = self.semantic_linear(backbone_feats)
        output['offset_predictions'] = self.offset_linear(backbone_feats)
        return output


    @cuda_cast
    def get_loss(self, model_output, semantic_labels, offset_labels, masks_off, masks_sem, **kwargs):
        loss_dict = dict()
        
        # Define variables
        semantic_prediction_logits = model_output['semantic_prediction_logits'].float()
        offset_predictions = model_output['offset_predictions'].float()
        
        # semantic and offset losses
        semantic_loss, offset_loss = point_wise_loss(
            semantic_prediction_logits,
            offset_predictions, 
            masks_sem, masks_off,
            semantic_labels, offset_labels
        )
        loss_dict['semantic_loss'] = semantic_loss * LOSS_MULTIPLIER_SEMANTIC
        loss_dict['offset_loss'] = offset_loss

        # Sum all losses
        loss = sum(_value for _value in loss_dict.values())
        return loss, loss_dict


def voxelize(feats, batch_ids, batch_size, voxel_size, use_coords, use_feats, max_num_points_per_voxel, epsilon=1):
    voxel_coords, voxel_feats, v2p_maps = [], [], []
    total_len_voxels = 0
    for i in range(batch_size):
        feats_one_element = feats[batch_ids == i]
        min_range = torch.min(feats_one_element[:, :3], dim=0).values
        max_range = torch.max(feats_one_element[:, :3], dim=0).values + epsilon
        voxelizer = PointToVoxel(
            vsize_xyz=[voxel_size, voxel_size, voxel_size], 
            coors_range_xyz=min_range.tolist() + max_range.tolist(),
            num_point_features=feats.shape[1], 
            max_num_voxels=len(feats), 
            max_num_points_per_voxel=max_num_points_per_voxel,
            device=feats.device)
        voxel_feat, voxel_coord, _, v2p_map = voxelizer.generate_voxel_with_id(feats_one_element)
        assert torch.sum(v2p_map == -1) == 0
        voxel_coord[:, [0, 2]] = voxel_coord[:, [2, 0]]
        voxel_coord = torch.cat((torch.ones((len(voxel_coord), 1), device=feats.device)*i, voxel_coord), dim=1)

        # get mean feature of voxel
        zero_rows = torch.sum(voxel_feat == 0, dim=2) == voxel_feat.shape[2]
        voxel_feat[zero_rows] = float("nan")
        voxel_feat = torch.nanmean(voxel_feat, dim=1)
        if not use_coords:
            voxel_feat[:, :3] = torch.ones_like(voxel_feat[:, :3])
        if not use_feats:
            voxel_feat[:, 3:] = torch.ones_like(voxel_feat[:, 3:])
        voxel_feat = torch.hstack([voxel_feat[:, 3:], voxel_feat[:, :3]])

        voxel_coords.append(voxel_coord)
        voxel_feats.append(voxel_feat)
        v2p_maps.append(v2p_map + total_len_voxels)
        total_len_voxels += len(voxel_coord) 
    voxel_coords = torch.cat(voxel_coords, dim=0)
    voxel_feats = torch.cat(voxel_feats, dim=0)
    v2p_maps = torch.cat(v2p_maps, dim=0)
    spatial_shape = voxel_coords.max(dim=0).values + 1

    return voxel_feats, voxel_coords, v2p_maps, spatial_shape[1:]
