"""
Basic SPformer model.
"""
import os
import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from timm.models.layers import to_4tuple

from demo.head import build_box_head
from demo.frozen_bn import FrozenBatchNorm2d

class SSP_Mlp(nn.Module):
    """ Self-attention spatial Prior Predictor module"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_4tuple(bias)
        drop_probs = to_4tuple(drop)

        self.fc1 = nn.Linear(in_features, in_features*8, bias=bias[0])
        self.act1 = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])

        # self.fc2 = nn.Linear(in_features*8, in_features*16, bias=bias[1])
        # self.act2 = act_layer()
        # self.drop2 = nn.Dropout(drop_probs[1])

        self.fc3 = nn.Linear(in_features*8, hidden_features, bias=bias[2])
        self.act3 = act_layer()
        self.drop3 = nn.Dropout(drop_probs[2])

        self.fc4 = nn.Linear(hidden_features, out_features, bias=bias[3])
        self.drop4 = nn.Dropout(drop_probs[3])

    def forward(self, x):
        """ Forward pass with input x. 
            x is the relative position matrice between feature pixels
        """
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.drop3(x)
        x = self.fc4(x)
        x = self.drop4(x)
        return x

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

class CNP_Predictor(nn.Module):
    """ Cross-attention Random Prior Predictor module"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16):
        super(CNP_Predictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        self.channel = channel
        '''convolution prior for three layers'''
        self.conv = nn.Sequential(conv(inplanes, channel),
                                        conv(channel, channel // 2),
                                        # conv(channel // 2, channel // 4),
                                        # conv(channel // 4, channel // 8),
                                        nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1))
        # self.conv_tower = conv(inplanes, channel)
        # self.conv_tower_mlp =  ATTNMLP(channel, 24, 12, 3)

        '''Predict the relation between template and search'''

    def forward(self, x):
        """ Forward pass with input x. 
            x is the cross-attention map after resized
        """
        score_map = self.conv(x)  # (B,12,H,W)
        return score_map


def vit_backbone(pretrained=False, **kwargs):
    """
    build a fully transformer-based tracker based on ViT-Base or CvT.
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """

class SPformer(nn.Module):
    """ This is the base class for SPformer """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CENTER"):
        """
        Initializes the model.

        Parameters:
            transformer: Torch module of the transformer architecture.
        """

        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        self.feat_sz_s = int(box_head.feat_sz)
        self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                template_mask=None,
                search_mask=None,
                return_last_attn=False,
                ):
        
        x, aux_dict = self.backbone(z=template, x=search,
                                    template_mask=template_mask,
                                    search_mask=search_mask,
                                    return_last_attn=return_last_attn, )

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_head(self, search_feature, gt_score_map=None):
        """
        search_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = search_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_SPformer(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''
    backbone = vit_backbone(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
    hidden_dim = backbone.embed_dim
    patch_start_index = 1

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = SPformer(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    return model

