import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from common_utils.protein.constants import BBHeavyAtom
from common_utils.modules.layers import LayerNorm, AngularEncoding
from common_utils.modules.geometry import global_to_local, local_to_global, construct_3d_basis, angstrom_to_nm, pairwise_directions


class PerResidueEncoder(nn.Module):

    def __init__(self, feat_dim, args, max_aa_types=22):
        super().__init__()
        self.args = args
        self.aatype_embed = nn.Embedding(max_aa_types, feat_dim)
        self.dihed_embed = AngularEncoding()
        self.mut_embed = nn.Embedding(num_embeddings=2, embedding_dim=int(feat_dim/2), padding_idx=0)

        if self.args.input_mode == 0:
            infeat_dim = feat_dim + self.dihed_embed.get_out_dim(7) + int(feat_dim/2)
        else:
            infeat_dim = feat_dim + int(feat_dim/2)

        self.out_mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, batch):

        N, L = batch['aa'].size() # (N, L)

        # Amino acid identity features
        aa_feat = self.aatype_embed(batch['aa']) # (N, L, F)

        # Dihedral features
        if self.args.input_mode == 0:
            dihedral = torch.cat([batch['phi'][..., None], batch['psi'][..., None], batch['omega'][..., None], batch['chi']], dim=-1) # (N, L, 7)
            dihedral_mask = torch.cat([batch['phi_mask'][..., None], batch['psi_mask'][..., None], batch['omega_mask'][..., None], batch['chi_mask']], dim=-1) # (N, L, 7)
            dihedral_feat = self.dihed_embed(dihedral[..., None]) * dihedral_mask[..., None] # (N, L, 7, F)
            dihedral_feat = dihedral_feat.reshape(N, L, -1) # (N, L, 7*F)

        # Mutation features
        mut_feat = self.mut_embed(batch['mut_flag'].long()) # (N, L, F)

        # Node features
        if self.args.input_mode == 0:
            out_feat = self.out_mlp(torch.cat([aa_feat, dihedral_feat, mut_feat], dim=-1)) # (N, L, F)
            mask_residue = batch['mask_atoms'][:, :, BBHeavyAtom.CA]
            out_feat = out_feat * mask_residue[:, :, None]
        else:
            out_feat = self.out_mlp(torch.cat([aa_feat, mut_feat], dim=-1)) # (N, L, F)

        return out_feat
    

class ResiduePairEncoder(nn.Module):

    def __init__(self, feat_dim, args, max_num_atoms=5, max_aa_types=22, max_relpos=32):
        super().__init__()
        self.args = args
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types
        self.max_relpos = max_relpos

        self.aa_pair_embed = nn.Embedding(self.max_aa_types*self.max_aa_types, feat_dim)
        self.relpos_embed = nn.Embedding(2*max_relpos+1, feat_dim)

        self.aapair_to_distcoef = nn.Embedding(self.max_aa_types*self.max_aa_types, max_num_atoms*max_num_atoms)
        nn.init.zeros_(self.aapair_to_distcoef.weight)
        self.distance_embed = nn.Sequential(
            nn.Linear(max_num_atoms*max_num_atoms, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
        )

        self.dihedral_embed = AngularEncoding()

        if self.args.input_mode == 0:
            infeat_dim = feat_dim + feat_dim + feat_dim + 3 * 4
        else:
            infeat_dim = feat_dim + feat_dim

        self.out_mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, batch):

        N, L = batch['aa'].size()

        # Pair identity features
        aa_pair = batch['aa'][:,:,None] * self.max_aa_types + batch['aa'][:,None,:] # (N, L, L)
        aa_pair_feat = self.aa_pair_embed(aa_pair) # (N, L, L, F)
    
        # Relative position features
        same_chain = (batch['chain_nb'][:, :, None] == batch['chain_nb'][:, None, :]) # (N, L, L)
        relpos = torch.clamp(batch['res_nb'][:,:,None] - batch['res_nb'][:,None,:], min=-self.max_relpos, max=self.max_relpos) # (N, L, L)
        relpos_feat = self.relpos_embed(relpos + self.max_relpos) * same_chain[:,:,:,None]

        if self.args.input_mode == 0:
            # Distance features
            d = angstrom_to_nm(torch.linalg.norm(batch['pos_atoms'][:,:,None,:,None] - batch['pos_atoms'][:,None,:,None,:], dim=-1, ord=2)).reshape(N, L, L, -1) # (N, L, L, A*A)
            c = F.softplus(self.aapair_to_distcoef(aa_pair)) # (N, L, L, A*A)
            d_gauss = torch.exp(-1 * c * d**2)
            mask_atom_pair = (batch['mask_atoms'][:,:,None,:,None] * batch['mask_atoms'][:,None,:,None,:]).reshape(N, L, L, -1)
            dist_feat = self.distance_embed(d_gauss * mask_atom_pair)

            # Direction features
            direct_feat = pairwise_directions(batch['pos_atoms'])

        # Edge features
        if self.args.input_mode == 0:
            feat_all = self.out_mlp(torch.cat([aa_pair_feat, relpos_feat, dist_feat, direct_feat], dim=-1)) # (N, L, L, F)
            mask_residue = batch['mask_atoms'][:, :, BBHeavyAtom.CA] # (N, L)
            mask_pair = mask_residue[:, :, None] * mask_residue[:, None, :] # (N, L, L)
            feat_all = feat_all * mask_pair[:, :, :, None] # (N, L, L, F)
        else:
            feat_all = self.out_mlp(torch.cat([aa_pair_feat, relpos_feat], dim=-1)) # (N, L, L, F)

        return feat_all
    

def _alpha_from_logits(logits, mask, inf=1e5):
    """
    Args:
        logits: Logit matrices, (N, L, L, num_heads).
        mask:   Masks, (N, L).
    Returns:
        alpha:  Attention weights.
    """
    mask_pair = mask.unsqueeze(-1).expand_as(logits)

    logits = torch.where(mask_pair, logits, logits - inf)
    alpha = torch.softmax(logits, dim=2)  # (N, L, L, H)

    alpha = torch.where(mask_pair, alpha, torch.zeros_like(alpha))

    return alpha


def _heads(x, n_heads, n_ch):
    """
    Args:
        x:  (..., num_heads * num_channels)
    Returns:
        (..., num_heads, num_channels)
    """
    s = list(x.size())[:-1] + [n_heads, n_ch]
    return x.view(*s)


class GABlock(nn.Module):

    def __init__(self, node_feat_dim, pair_feat_dim, args, hidden_dim=32, num_points=8, num_heads=8, bias=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_points = num_points
        self.num_heads = num_heads
        self.args = args

        # Node
        self.proj_query = nn.Linear(node_feat_dim, hidden_dim * num_heads, bias=bias)
        self.proj_key = nn.Linear(node_feat_dim, hidden_dim * num_heads, bias=bias)
        self.proj_value = nn.Linear(node_feat_dim, hidden_dim * num_heads, bias=bias)

        # Pair
        self.proj_pair = nn.Linear(pair_feat_dim, num_heads, bias=bias)

        # Spatial
        self.spatial_coef = nn.Parameter(torch.full([1, 1, 1, num_heads], fill_value=np.log(np.exp(1.) - 1.)), requires_grad=True)
        self.proj_query_point = nn.Linear(node_feat_dim, num_points * num_heads * 3, bias=bias)
        self.proj_key_point = nn.Linear(node_feat_dim, num_points * num_heads * 3, bias=bias)
        self.proj_value_point = nn.Linear(node_feat_dim, num_points * num_heads * 3, bias=bias)

        # Output
        if self.args.input_mode == 1:
            self.mlp_transition_1 = nn.Linear((num_heads * hidden_dim), node_feat_dim)
        else:
            self.mlp_transition_1 = nn.Linear((num_heads * pair_feat_dim) + (num_heads * hidden_dim), node_feat_dim)
        self.layer_norm_1 = LayerNorm(node_feat_dim)
        self.mlp_transition_2 = nn.Sequential(nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
                                              nn.Linear(node_feat_dim, node_feat_dim))
        self.layer_norm_2 = LayerNorm(node_feat_dim)

    def attention_logits(self, x, z):

        query_node = _heads(self.proj_query(x), self.num_heads, self.hidden_dim)  # (N, L, H, F)
        key_node = _heads(self.proj_key(x), self.num_heads, self.hidden_dim)  # (N, L, H, F)
        logits_node = (query_node.unsqueeze(2) * key_node.unsqueeze(1) * (1 / np.sqrt(self.hidden_dim))).sum(-1)  # (N, L, L, H)
    
        if self.args.input_mode == 1:
            return logits_node
        else:
            logits_pair = self.proj_pair(z)
            return (logits_node + logits_pair) * np.sqrt(1 / 2)
    
    def node_aggregation(self, alpha, x, z):
        N, L = x.shape[:2]

        value_node = _heads(self.proj_value(x), self.num_heads, self.hidden_dim)  # (N, L, H, F)
        feat_node = alpha.unsqueeze(-1) * value_node.unsqueeze(1)  # (N, L, L, H, *) @ (N, *, L, H, F)
        feat_node = feat_node.sum(dim=2).reshape(N, L, -1)  # (N, L, H*F)

        if self.args.input_mode == 1:
            return feat_node
        else:
            feat_pair = alpha.unsqueeze(-1) * z.unsqueeze(-2)  # (N, L, L, H, *) @ (N, L, L, *, F)
            feat_pair = feat_pair.sum(dim=2).reshape(N, L, -1)  # (N, L, H*F)

            return torch.cat([feat_node, feat_pair], dim=-1)

    def forward(self, x, z):
        # Attention weights
        att_logits = self.attention_logits(x, z)
        alpha = torch.softmax(att_logits, dim=2)

        # Aggregate features
        agg_feats = self.node_aggregation(alpha, x, z)

        # Update features
        feats = self.layer_norm_1(x + self.mlp_transition_1(agg_feats))
        out_feats = self.mlp_transition_2(feats)

        if feats.shape[-1] == out_feats.shape[-1]:
            feats = self.layer_norm_2(feats + out_feats)
        else:
            feats = self.layer_norm_2(out_feats)

        return feats


class GAEncoder(nn.Module):

    def __init__(self, node_feat_dim, pair_feat_dim, num_layers, args):
        super(GAEncoder, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            if i == num_layers - 1:
                self.blocks.append(GABlock(node_feat_dim, pair_feat_dim, args, num_heads=args.n_heads))
            else:
                self.blocks.append(GABlock(node_feat_dim, pair_feat_dim, args, num_heads=args.n_heads))
        self.args = args
    
    def forward(self, res_feat, pair_feat):
        for block in self.blocks:
            res_feat = block(res_feat, pair_feat)

        return res_feat