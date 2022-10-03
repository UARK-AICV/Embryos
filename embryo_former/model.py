# ------------------------------------------------------------------------
# EmbryosFormer
# ------------------------------------------------------------------------
# Modified from PDVC(https://github.com/ttengwang/PDVC)
# ------------------------------------------------------------------------
# Modified from Deformable DETR(https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from torch import nn
import math
import copy

from misc.detr_utils import box_ops
from misc.detr_utils.misc import inverse_sigmoid
from .deformable_transformer import build_deforamble_transformer
from .criterion import SetCriterion
from .base_encoder import build_base_encoder
from .transformer_utils import (
    MLP,
    RefineTransformerDecoderLayer, RefineTransformerDecoder
)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def get_query_positional_emb(Nq: int=4, dq: int=256, Bs: int=16):
    temperature = 10000
    num_pos_feats = dq
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
    dim_t = temperature ** (2 * (torch.div(dim_t, 2, rounding_mode='floor')) / num_pos_feats)
    pos_emb = torch.ones(Bs, Nq)
    pos_emb = torch.cumsum(pos_emb, dim=1)
    # import pdb; pdb.set_trace()
    pos_emb = pos_emb[:, :, None] / dim_t
    pos_emb = torch.stack((pos_emb[:, :, 0::2].sin(), pos_emb[:, :, 1::2].cos()), dim=3).flatten(2)

    return pos_emb

class EmbryoFormer(nn.Module):
    """ This is the PDVC module that performs dense video captioning """

    def __init__(self, base_encoder, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, opt=None, translator=None):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            captioner: captioning head for generate a sentence for each event queries
            num_classes: number of foreground classes
            num_queries: number of event queries. This is the maximal number of events
                         PDVC can detect in a single video. For ActivityNet Captions, we recommend 10-30 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            opt: all configs
        """
        super().__init__()
        self.opt = opt
        self.base_encoder = base_encoder
        self.transformer = transformer

        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        self.class_head = nn.Linear(hidden_dim, num_classes)
        # self.count_head = nn.Linear(hidden_dim, opt.max_eseq_length + 1)
        self.bbox_head = MLP(hidden_dim, hidden_dim, 2, 3)
        
        self.bbox_width_head = MLP(hidden_dim, hidden_dim, 1, 3)
        self.frame_head = nn.Linear(hidden_dim, num_classes)
        self.bbox_width_softmax = nn.Softmax(dim=1)

        self.num_feature_levels = num_feature_levels
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine

        # initialization
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_head.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_head.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_head.layers[-1].bias.data, 0)

        num_pred = transformer.decoder.num_layers
        
        if with_box_refine:
            self.class_head = _get_clones(self.class_head, num_pred)
            self.bbox_head = _get_clones(self.bbox_head, num_pred)
            self.bbox_width_head = _get_clones(self.bbox_width_head, num_pred)
            nn.init.constant_(self.bbox_head[0].layers[-1].bias.data[1:], -2)
            self.transformer.decoder.bbox_head = self.bbox_head
        else:
            nn.init.constant_(self.bbox_head.layers[-1].bias.data[1:], -2)
            self.class_head = nn.ModuleList([self.class_head for _ in range(num_pred)])
            self.bbox_head = nn.ModuleList([self.bbox_head for _ in range(num_pred)])
            self.bbox_width_head = nn.ModuleList([self.bbox_width_head for _ in range(num_pred)])
            self.transformer.decoder.bbox_head = None

        self.translator = translator

        refine_n_heads = opt.refine_n_heads
        self.refine_n_layers = opt.refine_n_layers
        self.refine_dec_layer = RefineTransformerDecoderLayer(d_model=hidden_dim, dropout=0.05, n_heads=refine_n_heads)
        self.refine_dec = RefineTransformerDecoder(self.refine_dec_layer, self.refine_n_layers)
        
        self.refine_proj = MLP(num_classes, num_classes*2, num_classes, 1, activation='none') #'leaky_relu'
        
        self.refine_frame_head = nn.Linear(hidden_dim, num_classes)
        pass 

    def get_filter_rule_for_encoder(self):
        filter_rule = lambda x: 'input_proj' in x \
                                or 'transformer.encoder' in x \
                                or 'transformer.level_embed' in x \
                                or 'base_encoder' in x
        return filter_rule

    def encoder_decoder_parameters(self):
        filter_rule = self.get_filter_rule_for_encoder()
        enc_paras = []
        dec_paras = []
        for name, para in self.named_parameters():
            if filter_rule(name):
                print('enc: {}'.format(name))
                enc_paras.append(para)
            else:
                print('dec: {}'.format(name))
                dec_paras.append(para)
        return enc_paras, dec_paras

    def forward(self, dt, criterion, eval_mode=False):
        vf = dt['video_tensor']  # (N, L, C)
        mask = ~dt['video_mask']  # (N, L)
        duration = dt['video_length'][:, 1]
        N, L, C = vf.shape

        srcs, masks, pos = self.base_encoder(vf, mask, duration)
        src_flatten, temporal_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten = \
            self.transformer.prepare_encoder_inputs(srcs, masks, pos)

        memory = self.transformer.forward_encoder(src_flatten, temporal_shapes, level_start_index, valid_ratios,
                                                lvl_pos_embed_flatten, mask_flatten)
        srcs_lens = [src.shape[2] for src in srcs]
        splits = torch.split(memory, srcs_lens, dim=1)
        frame_preds = self.frame_head(splits[0]) 

        proposals_mask = None
        disable_iterative_refine = False

        query_embed = self.query_embed.weight
        proposals_mask = torch.ones(N, query_embed.shape[0], device=query_embed.device).bool()
        init_reference, tgt, reference_points, query_embed = self.transformer.prepare_decoder_input_query(memory, query_embed)
        B, Nq, dq = query_embed.shape
        query_pe = get_query_positional_emb(Nq, dq, B)
        tgt = tgt + query_pe.to(query_embed.device)

        hs, inter_references = self.transformer.forward_decoder(
                                tgt, reference_points, memory, temporal_shapes,
                                level_start_index, valid_ratios, query_embed,
                                mask_flatten, proposals_mask, disable_iterative_refine
                            )

        ### Collaborative head
        final_query_embed = hs[-1]
        B, Nq, dq = final_query_embed.shape
        query_pe = get_query_positional_emb(Nq, dq, B)
        final_query_embed = final_query_embed + query_pe.to(final_query_embed.device) # Add stage as prior for query vectors
        
        # Add width-refinement branch
        frame_query = splits[0]
        list_frame_emb, list_attn_weight = self.refine_dec(frame_query, final_query_embed)
        attn_weight = list_attn_weight[-1]
        ref_emb = list_frame_emb[-1]
        refine_frame_preds = self.refine_frame_head(ref_emb)

        others = {
            'memory': memory,
            'frame_preds': frame_preds,
            'mask_flatten': mask_flatten,
            'spatial_shapes': temporal_shapes,
            'level_start_index': level_start_index,
            'valid_ratios': valid_ratios,
            'proposals_mask': proposals_mask,
            'refine_weight_attn': attn_weight,
            'refine_frame_emb': ref_emb,
            'refine_frame_preds': refine_frame_preds,
        }

        out, loss = self.predict(dt, hs, init_reference, inter_references, others, criterion, disable_iterative_refine, eval_mode)
        return out, loss

    def predict_event_num(self, counter, hs_lid):
        hs_lid_pool = torch.max(hs_lid, dim=1, keepdim=False)[0]  # [bs, feat_dim]
        outputs_class0 = counter(hs_lid_pool)
        return outputs_class0

    
    def predict(self, dt, hs, init_reference, inter_references, others, criterion, disable_iterative_refine, eval_mode=False):
        outputs_classes = []
        outputs_coords = []
        outputs_widths = []
        outputs_centers = []

        num_pred = hs.shape[0]

        for l_id in range(num_pred):
            hs_lid = hs[l_id]
            reference = init_reference if l_id == 0 else inter_references[l_id - 1]  # [decoder_layer, batch, query_num, ...]
            outputs_class = self.class_head[l_id](hs_lid)  # [bs, num_query, N_class]
            tmp = self.bbox_head[l_id](hs_lid)  # [bs, num_query, 2]
 
            if disable_iterative_refine:
                outputs_coord = reference
            else:
                reference = inverse_sigmoid(reference)
                if reference.shape[-1] == 2:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 1
                    tmp[..., :1] += reference
                outputs_coord = tmp.sigmoid()  # [bs, num_query, 4]

            tmp_ref = inverse_sigmoid(outputs_coord)
            output_center, output_width = torch.split(tmp_ref, [1, 1], dim=-1)
            tmp_width = self.bbox_width_softmax(output_width)

            outputs_widths.append(tmp_width)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_centers.append(output_center.sigmoid())
        
        outputs_class = torch.stack(outputs_classes)  # [decoder_layer, bs, num_query, N_class]
        outputs_coord = torch.stack(outputs_coords) 
        outputs_width = torch.stack(outputs_widths)

        all_out = {
            'pred_logits': outputs_class,
            'pred_boxes': outputs_coord,
            'pred_width': outputs_width,
            'pred_center': outputs_centers,
        }
        out = {k: v[-1] for k, v in all_out.items()}
        out['pred_frames'] = others['frame_preds']
        out['query_feats'] = hs
        
        out['refine_weight_attn'] = others['refine_weight_attn']
        if eval_mode == False:
            out['pred_frames_refine'] = criterion.get_frame_refine_prediction(out, dt['video_target'], pos_weight=dt['video_pos_weight'], is_training = True)
        else:
            out['pred_frames_refine'] = criterion.get_frame_refine_prediction(out, dt['video_target'], pos_weight=dt['video_pos_weight'], is_training = False)

        if self.aux_loss:
            ks, vs = list(zip(*(all_out.items())))
            out['aux_outputs'] = [{ks[i]: vs[i][j] for i in range(len(ks))} for j in range(num_pred - 1)]
            loss, last_indices, aux_indices = criterion(out, dt['video_target'])
        else:
            loss, last_indices = criterion(out, dt['video_target'])

        return out, loss


def build(args):
    device = torch.device(args.device)
    base_encoder = build_base_encoder(args)
    transformer = build_deforamble_transformer(args)

    model = EmbryoFormer(
        base_encoder,
        transformer,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        opt=args
    )
    weight_dict = args.weight_dict

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = args.losses
    aux_losses = args.aux_losses

    criterion = SetCriterion(args.num_classes, weight_dict, losses, aux_losses, opt=args)
    criterion.to(device)

    return model, criterion

