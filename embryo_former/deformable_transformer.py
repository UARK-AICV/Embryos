# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, normal_

from misc.detr_utils.misc import  inverse_sigmoid
from embryo_former.ops.modules import MSDeformAttn
from .transformer_utils import (
    DeformableTransformerEncoder, DeformableTransformerDecoder,
    DeformableTransformerEncoderLayer, DeformableTransformerDecoderLayer
)

class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        self.no_encoder = (num_encoder_layers == 0)
        self.num_feature_levels = num_feature_levels

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.pos_trans = nn.Linear(d_model, d_model * 2)
        self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        self.reference_points = nn.Linear(d_model, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)


    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 256
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (torch.div(dim_t, 2, rounding_mode='floor')) / num_pos_feats)
        # N, L, 2
        proposals = proposals.sigmoid() * scale
        # N, L, 2, 256
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 2, 128, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def get_valid_ratio(self, mask):
        valid_ratio_L = torch.sum(~mask, 1).float() / mask.shape[1]
        return valid_ratio_L

    def prepare_encoder_inputs(self, srcs, masks, pos_embeds):
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        temporal_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            """
            lvl: (bs, )
            src: (bs, c, L )
            mask: (bs, L)
            pos_embed: (bs, d_m, L)
            """
            bs, c, L = src.shape
            temporal_shapes.append(L)
            src = src.transpose(1, 2)  # （bs, L, c）
            pos_embed = pos_embed.transpose(1, 2)  # #（bs, L, d_m）
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
            
        src_flatten = torch.cat(src_flatten, 1)  # (lvl_num, bs, wh, c)
        mask_flatten = torch.cat(mask_flatten, 1)  # (lvl_num, bs, wh)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # (lvl_num, bs, wh, d_m)
        temporal_shapes = torch.as_tensor(temporal_shapes, dtype=torch.long, device=src_flatten.device)  # (lvl_num, 2)
        level_start_index = torch.cat(
            # prod: [w0h0, w0h0+w1h1, w0h0+w1h1+w2h2, ...]
            (temporal_shapes.new_zeros((1,)), temporal_shapes.cumsum(0)[:-1]))  

        # (bs, lvl_num, 2), where 2 means (h_rate, and w_rate)， all values <= 1        
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)  

        return src_flatten, temporal_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten

    def forward_encoder(self, src_flatten, temporal_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                        mask_flatten):
        # encoder
        if self.no_encoder:
            memory = src_flatten
        else:
            memory = self.encoder(src_flatten, temporal_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                                  mask_flatten)

        return memory

    def prepare_decoder_input_query(self, memory, query_embed):
        bs, _, _ = memory.shape
        query_embed, tgt = torch.chunk(query_embed, 2, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_embed).sigmoid()
        init_reference_out = reference_points  # (bs, object_query, 2)
        return init_reference_out, tgt, reference_points, query_embed

    def prepare_decoder_input_proposal(self, gt_reference_points):
        topk_coords_unact = inverse_sigmoid(gt_reference_points)
        reference_points = gt_reference_points
        init_reference_out = reference_points
        pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
        query_embed, tgt = torch.chunk(pos_trans_out, 2, dim=2)
        return init_reference_out, tgt, reference_points, query_embed

    def forward_decoder(self, *kargs):
        hs, inter_references_out = self.decoder(*kargs)
        return hs, inter_references_out


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.transformer_ff_dim,
        dropout=args.transformer_dropout_prob,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points)
