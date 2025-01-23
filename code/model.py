import torch
import torch.nn as nn
import torch.nn.functional as F

from module import PerResidueEncoder, ResiduePairEncoder, GAEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DDG_RDE_Network(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args

        self.single_encoder = PerResidueEncoder(feat_dim=args.node_feat_dim, args=args)
        self.pair_encoder = ResiduePairEncoder(feat_dim=args.pair_feat_dim, args=args)
        self.attn_encoder = GAEncoder(args.node_feat_dim, args.pair_feat_dim, args.num_layers, args)

        self.ddg_readout = nn.Sequential(nn.Linear(args.node_feat_dim, args.node_feat_dim), 
                                         nn.ReLU(), 
                                         nn.Linear(args.node_feat_dim, args.node_feat_dim), 
                                         nn.ReLU(), 
                                         nn.Linear(args.node_feat_dim, 1)
                                         )

    def encode(self, batch):
    
        h = self.single_encoder(batch)
        z = self.pair_encoder(batch)
        h = self.attn_encoder(h, z)

        return h

    def forward(self, batch, mode='train'):

        if mode == 'test':
            batch_wt, batch_mt = {}, {}
            for k in batch[0].keys():
                if k != 'chain_id':
                    batch_wt[k] = torch.cat([t[k][0][None,...] for t in batch], dim=0)
                    batch_mt[k] = torch.cat([t[k][1][None,...] for t in batch], dim=0)
        else:
            batch_wt = {k: v for k, v in batch.items()}
            batch_mt = {k: v for k, v in batch.items()}
            batch_mt['aa'] = batch_mt['aa_mut']

        h_wt = self.encode(batch_wt)
        h_mt = self.encode(batch_mt)

        H_mt, H_wt = h_mt.max(dim=1)[0], h_wt.max(dim=1)[0]
        ddg_pred = self.ddg_readout(H_mt - H_wt).squeeze(-1)

        if mode == 'test' or mode == 'mutation':
            return ddg_pred

        if self.args.ab_mode == -1 or mode == 'val':
            loss = F.mse_loss(ddg_pred, batch['ddG'])
        else:
            train_mask, val_mask = (batch['mask'] == 1), (batch['mask'] == 0)
            train_pred_ddg = ddg_pred[train_mask]
            val_pred_ddg = ddg_pred[val_mask]

            train_gt_ddg = batch['ddG'][train_mask]
            train_kd_ddg = batch['t_ddG'][train_mask]
            val_kd_ddg = batch['t_ddG'][val_mask]

            train_gt_loss = F.mse_loss(train_pred_ddg, train_gt_ddg)
            train_kd_loss = F.mse_loss(train_pred_ddg, train_kd_ddg)
            val_kd_loss = F.mse_loss(val_pred_ddg, val_kd_ddg)
            if train_mask.sum().item() == 0:
                train_gt_loss, train_kd_loss = 0, 0
            if val_mask.sum().item() == 0:
                val_kd_loss = 0

            if self.args.ab_mode == 0:
                loss = train_gt_loss
            elif self.args.ab_mode == 1:
                loss = train_gt_loss * self.args.loss_weight + train_kd_loss * (1 - self.args.loss_weight)
            elif self.args.ab_mode == 2:
                train_val_ratio = batch['mask'].sum() / self.args.batch_size
                loss = train_gt_loss * train_val_ratio + val_kd_loss * (1 - train_val_ratio)
            elif self.args.ab_mode == 3:
                train_val_ratio = batch['mask'].sum() / self.args.batch_size
                loss = (train_gt_loss * self.args.loss_weight + train_kd_loss * (1 - self.args.loss_weight)) * train_val_ratio + val_kd_loss * (1 - train_val_ratio)

        out_dict = {
            'ddG_pred': ddg_pred,
            'ddG_true': batch['ddG'],
        }

        return loss, out_dict