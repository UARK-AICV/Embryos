import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, normal_

from misc.detr_utils.misc import  inverse_sigmoid
from embryo_former.ops.modules import MSDeformAttn

class RefineTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=512, dropout=0.1, n_heads=4):
        super().__init__()
        self.emb_dim = d_model
        
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Dropout(dropout), nn.LayerNorm(d_model)
        )
        
    def forward(self, frame_emb, query_emb):
        attn_output, attn_weight = self.cross_attn(frame_emb, query_emb, query_emb, need_weights=True)
        frame_emb = frame_emb + self.dropout1(attn_output) 
        frame_emb = self.norm1(frame_emb)
        frame_emb = self.ffn1(frame_emb)
        return frame_emb, attn_weight

class RefineTransformerDecoder(nn.Module):
    def __init__(self, dec_layer, n_layers) -> None:
        super().__init__()
        self.layers = _get_clones(dec_layer, n_layers)
        self.num_layers = n_layers

    def forward(self, frame_emb, query_emb):
        list_frame_emb, list_attn_weight = [], []
        out_emb = frame_emb
        for i in range(self.num_layers):
            out_emb, out_weight = self.layers[i](out_emb, query_emb)
            list_frame_emb.append(out_emb)
            list_attn_weight.append(out_weight)
        
        return list_frame_emb, list_attn_weight


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, temporal_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, temporal_shapes, level_start_index,
                              padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(temporal_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (L_) in enumerate(temporal_shapes):
            ref = torch.linspace(0.5, L_ - 0.5, L_, dtype=torch.float32, device=device)
            ref = ref.reshape(-1)[None] / (valid_ratios[:, None, lvl] * L_)
            reference_points_list.append(ref)
            
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        reference_points = reference_points[:,:,:,None]
        return reference_points

    def forward(self, src, temporal_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(temporal_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, temporal_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_temporal_shapes, level_start_index,
                src_padding_mask=None, query_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), 
            key_padding_mask=~query_mask)[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos), reference_points,
                               src, src_temporal_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)
        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_head = None

    def forward(self, tgt, reference_points, src, src_temporal_shapes, 
                src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, query_padding_mask=None, disable_iterative_refine=False):
        """_summary_

        Args:
            tgt (_type_): [Bs, N_queries, d_model]
            reference_points (_type_): [Bs, N_queries, 1]
            src (_type_): encoder memory [Bs, 188, d_model]
            src_temporal_shapes (_type_): [4]: (100, 50, 25, 13)
            src_level_start_index (_type_): _description_
            src_valid_ratios (_type_): _description_
            query_pos (_type_, optional): query_embed, shape = tgt.shape.
            src_padding_mask (_type_, optional): [Bs, 188]. Defaults to None.
            query_padding_mask (_type_, optional): proposal_mask = dt['gt_boxes_mask']. Defaults to None.
            disable_iterative_refine (bool, optional): True if two_stage. Defaults to False.

        Returns:
            _type_: _description_
        """
        
        output = tgt
        intermediate = []
        intermediate_reference_points = []
        bs = tgt.shape[0]
        prev_rep = None
        
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 2:
                stack_ratio = torch.stack([src_valid_ratios, src_valid_ratios], -1)
                reference_points_input = reference_points[:, :, None] \
                                         * torch.stack([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 1
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None, :, None]
            
            # Cross attention
            # if lid > 0 and prev_rep is not None:
            #     output = output + prev_rep
            output = layer(output, query_pos, reference_points_input, src, src_temporal_shapes, src_level_start_index,
                           src_padding_mask, query_padding_mask)
            # prev_rep = output

            # ------------------ Iterative Bounding Box refinement ------------------
            # hack implementation for iterative bounding box refinement
            if disable_iterative_refine:
                reference_points = reference_points
            else:
                if (self.bbox_head is not None):
                    tmp = self.bbox_head[lid](output)
                    # print(f'Run Iterative Bounding Box refinement in Decoding process')
                    if reference_points.shape[-1] == 2:
                        new_reference_points = tmp + inverse_sigmoid(reference_points)
                        new_reference_points = new_reference_points.sigmoid()
                    else:
                        assert reference_points.shape[-1] == 1
                        new_reference_points = tmp
                        new_reference_points[..., :1] = tmp[..., :1] + inverse_sigmoid(reference_points)
                        new_reference_points = new_reference_points.sigmoid()
                    reference_points = new_reference_points.detach()
                else:
                    reference_points = reference_points

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation="relu"):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.activation = activation

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if self.activation == 'relu':
                x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            elif self.activation == 'leaky_relu':
                x = F.leaky_relu(layer(x), negative_slope=0.1) if i < self.num_layers - 1 else layer(x)
            else:
                x = layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

