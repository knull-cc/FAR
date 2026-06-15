import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from far.retriever import FARRetriever

class RetrievalTool():
    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        n_period=3,
        temperature=0.1,
        topm=20,
        with_dec=False,
        return_key=False,
        use_far=False,
        use_covariates=False,
        cov_channels=0,
        far_config=None,
        num_workers=0,
    ):
        period_num = [16, 8, 4, 2, 1]
        period_num = period_num[-1 * n_period:]
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        
        self.n_period = n_period
        self.period_num = sorted(period_num, reverse=True)
        
        self.temperature = temperature
        self.topm = topm
        
        self.with_dec = with_dec
        self.return_key = return_key
        self.num_workers = num_workers

        # ---- FAR (Future-Aligned Retrieval) configuration --------------------
        # When use_far is True, the past-similarity correlation key is replaced
        # by FAR's future-aligned embedding similarity. Everything downstream
        # (multi-grain future aggregation + fusion head) is left unchanged, so
        # toggling use_far is a clean ablation: FAR vs past-retriever, same host.
        self.use_far = use_far
        self.use_covariates = use_covariates
        self.cov_channels = cov_channels if use_covariates else 0
        far_config = far_config or {}
        self.far_config = far_config
        # Fusion-softmax temperature for the FAR path. Defaults to the same
        # value as the RAFT correlation key (0.1) so FAR's only delta over RAFT
        # is the future-aligned retrieval key, not the fusion. Exposed as a
        # knob purely for studying fusion sharpness.
        self.far_fuse_temperature = far_config.get('fuse_temperature', 0.1)
        self.x_mark_all = None
        self.far = None
        if self.use_far:
            self.far = FARRetriever(
                seq_len=seq_len,
                pred_len=pred_len,
                channels=channels,
                cov_channels=self.cov_channels,
                emb_dim=far_config.get('emb_dim', 128),
                d_model=far_config.get('d_model', 128),
                n_blocks=far_config.get('n_blocks', 3),
                dropout=far_config.get('dropout', 0.1),
                use_revin=far_config.get('use_revin', True),
                temperature=far_config.get('temperature', 0.1),
                pos_k=far_config.get('pos_k', 5),
                future_metric=far_config.get('future_metric', 'shape'),
                soft_dtw_gamma=far_config.get('soft_dtw_gamma', 0.1),
                use_hard_neg=far_config.get('use_hard_neg', False),
                hard_scale=far_config.get('hard_scale', 3.0),
                use_gating=far_config.get('use_gating', False),
                n_grains=self.n_period,
                aux_weight=far_config.get('aux_weight', 1.0),
            )
        
    def prepare_dataset(self, train_data):
        train_data_all = []
        y_data_all = []
        x_mark_all = []

        for i in range(len(train_data)):
            td = train_data[i]
            train_data_all.append(td[1])
            
            if self.with_dec:
                y_data_all.append(td[2][-(train_data.pred_len + train_data.label_len):])
            else:
                y_data_all.append(td[2][-train_data.pred_len:])

            if self.use_far and self.use_covariates:
                x_mark_all.append(td[3])
            
        self.train_data_all = torch.tensor(np.stack(train_data_all, axis=0)).float()
        self.train_data_all_mg, _ = self.decompose_mg(self.train_data_all)
        
        self.y_data_all = torch.tensor(np.stack(y_data_all, axis=0)).float()
        self.y_data_all_mg, _ = self.decompose_mg(self.y_data_all)

        if self.use_far and self.use_covariates:
            self.x_mark_all = torch.tensor(np.stack(x_mark_all, axis=0)).float()
            # The covariate (time-feature) channel count is only known once the
            # data is loaded; rebuild the FAR encoder to match it. The encoder is
            # trained separately and frozen, so rebuilding here is safe.
            detected = self.x_mark_all.shape[-1]
            if detected != self.cov_channels:
                self.cov_channels = detected
                self.far = FARRetriever(
                    seq_len=self.seq_len,
                    pred_len=self.pred_len,
                    channels=self.channels,
                    cov_channels=self.cov_channels,
                    emb_dim=self.far_config.get('emb_dim', 128),
                    d_model=self.far_config.get('d_model', 128),
                    n_blocks=self.far_config.get('n_blocks', 3),
                    dropout=self.far_config.get('dropout', 0.1),
                    use_revin=self.far_config.get('use_revin', True),
                    temperature=self.far_config.get('temperature', 0.1),
                    pos_k=self.far_config.get('pos_k', 5),
                    future_metric=self.far_config.get('future_metric', 'shape'),
                    soft_dtw_gamma=self.far_config.get('soft_dtw_gamma', 0.1),
                    use_hard_neg=self.far_config.get('use_hard_neg', False),
                    hard_scale=self.far_config.get('hard_scale', 3.0),
                    use_gating=self.far_config.get('use_gating', False),
                    n_grains=self.n_period,
                    aux_weight=self.far_config.get('aux_weight', 1.0),
                )

        self.n_train = self.train_data_all.shape[0]

    def train_far(self, device):
        """Contrastively train the FAR encoder on the KB and build the index.

        Positives/negatives are defined by FUTURE similarity; the encoder sees
        only the PAST (+ covariates). Must be called after prepare_dataset and
        before any retrieve / retrieve_all call.
        """
        assert self.use_far and self.far is not None
        cfg = self.far_config
        cov_all = self.x_mark_all if self.use_covariates else None

        print('Training FAR future-aligned encoder (per grain)')
        self.far.fit(
            past_all_mg=self.train_data_all_mg,
            future_all_mg=self.y_data_all_mg,
            cov_all=cov_all,
            device=device,
            epochs=cfg.get('epochs', 10),
            batch_size=cfg.get('batch_size', 256),
            lr=cfg.get('lr', 1e-3),
            verbose=True,
        )
        print('Encoding FAR knowledge base (per grain)')
        self.far.encode_kb(
            past_all_mg=self.train_data_all_mg,
            cov_all=cov_all,
            device=device,
            batch_size=cfg.get('encode_batch_size', 1024),
        )

    def decompose_mg(self, data_all, remove_offset=True):
        data_all = copy.deepcopy(data_all) # T, S, C

        mg = []
        for g in self.period_num:
            cur = data_all.unfold(dimension=1, size=g, step=g).mean(dim=-1)
            cur = cur.repeat_interleave(repeats=g, dim=1)
            
            mg.append(cur)
#             data_all = data_all - cur
            
        mg = torch.stack(mg, dim=0) # G, T, S, C

        if remove_offset:
            offset = []
            for i, data_p in enumerate(mg):
                cur_offset = data_p[:,-1:,:]
                mg[i] = data_p - cur_offset
                offset.append(cur_offset)
        else:
            offset = None
            
        offset = torch.stack(offset, dim=0)
            
        return mg, offset
    
    def periodic_batch_corr(self, data_all, key, in_bsz = 512):
        _, bsz, features = key.shape
        _, train_len, _ = data_all.shape
        
        bx = key - torch.mean(key, dim=2, keepdim=True)
        
        iters = math.ceil(train_len / in_bsz)
        
        sim = []
        for i in range(iters):
            start_idx = i * in_bsz
            end_idx = min((i + 1) * in_bsz, train_len)
            
            cur_data = data_all[:, start_idx:end_idx].to(key.device)
            ax = cur_data - torch.mean(cur_data, dim=2, keepdim=True)
            
            cur_sim = torch.bmm(F.normalize(bx, dim=2), F.normalize(ax, dim=2).transpose(-1, -2))
            sim.append(cur_sim)
            
        sim = torch.cat(sim, dim=2)
        
        return sim
        
    def retrieve(self, x, index, train=True, x_mark=None):
        index = index.to(x.device)
        
        bsz, seq_len, channels = x.shape
        assert seq_len == self.seq_len and channels == self.channels

        if self.use_far:
            # FAR: future-aligned embedding similarity (replaces past-corr key).
            # Decompose the query into the same multi-grain views the KB index
            # was built on, then retrieve a separate future-aligned ranking per
            # grain (mirrors RAFT's per-grain correlation key).
            x_mg, _ = self.decompose_mg(x)  # G, B, S, C
            cov = x_mark if (self.use_covariates and x_mark is not None) else None
            sim = self.far.query_similarity(x_mg, cov, device=x.device)  # G, B, T
        else:
            x_mg, mg_offset = self.decompose_mg(x) # G, B, S, C

            sim = self.periodic_batch_corr(
                self.train_data_all_mg.flatten(start_dim=2), # G, T, S * C
                x_mg.flatten(start_dim=2), # G, B, S * C
            ) # G, B, T
            
        if train:
            sliding_index = torch.arange(2 * (self.seq_len + self.pred_len) - 1).to(x.device)
            sliding_index = sliding_index.unsqueeze(dim=0).repeat(len(index), 1)
            sliding_index = sliding_index + (index - self.seq_len - self.pred_len + 1).unsqueeze(dim=1)
            
            sliding_index = torch.where(sliding_index >= 0, sliding_index, 0)
            sliding_index = torch.where(sliding_index < self.n_train, sliding_index, self.n_train - 1)

            self_mask = torch.zeros((bsz, self.n_train)).to(x.device)
            self_mask = self_mask.scatter_(1, sliding_index, 1.)
            self_mask = self_mask.unsqueeze(dim=0).repeat(self.n_period, 1, 1)
            
            sim = sim.masked_fill_(self_mask.bool(), float('-inf')) # G, B, T

        sim = sim.reshape(self.n_period * bsz, self.n_train) # G X B, T
                
        topm = torch.topk(sim, self.topm, dim=1)
        topm_index = topm.indices
        ranking_sim = torch.ones_like(sim) * float('-inf')
        
        rows = torch.arange(sim.size(0)).unsqueeze(-1).to(sim.device)
        ranking_sim[rows, topm_index] = sim[rows, topm_index]
        
        sim = sim.reshape(self.n_period, bsz, self.n_train) # G, B, T
        ranking_sim = ranking_sim.reshape(self.n_period, bsz, self.n_train) # G, B, T

        data_len, seq_len, channels = self.train_data_all.shape

        # By default FAR uses the same fusion temperature as the RAFT
        # correlation key (far_fuse_temperature defaults to self.temperature),
        # so the only delta over RAFT is the future-aligned retrieval key.
        fuse_temperature = self.far_fuse_temperature if self.use_far else self.temperature
        ranking_prob = F.softmax(ranking_sim / fuse_temperature, dim=2)
        ranking_prob = ranking_prob.detach().cpu() # G, B, T
        
        y_data_all = self.y_data_all_mg.flatten(start_dim=2) # G, T, P * C
        
        pred_from_retrieval = torch.bmm(ranking_prob, y_data_all).reshape(self.n_period, bsz, -1, channels)
        pred_from_retrieval = pred_from_retrieval.to(x.device)

        # B4 gating: expose the top-k retrieval similarities (confidence) of the
        # future-aligned ranking so the host can learn a confidence-aware gate.
        topk_vals = topm.values.reshape(self.n_period, bsz, self.topm)
        topk_sims = None
        if self.use_far and self.far_config.get('use_gating', False):
            topk_sims = topk_vals[0].detach().cpu()  # B, topm

        # Retrieval-quality diagnostic: the top-m similarity values that drive
        # the softmax fusion weights. If these are high but collapsed (tiny
        # gap), the softmax degenerates to a uniform average of the top-m
        # futures -> over-smoothed prediction -> higher MSE.
        topm_diag = topk_vals[-1].detach().cpu()  # finest grain (g=1), B, topm

        return pred_from_retrieval, topk_sims, topm_diag
    
    def retrieve_all(self, data, train=False, device=torch.device('cpu'), tag=''):
        assert self.train_data_all_mg is not None
        
        rt_loader = DataLoader(
            data,
            batch_size=1024,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False
        )
        
        retrievals = []
        topk_sims_all = []
        topm_diag_all = []
        with torch.no_grad():
            for index, batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(rt_loader):
                pred_from_retrieval, topk_sims, topm_diag = self.retrieve(
                    batch_x.float().to(device), index, train=train,
                    x_mark=batch_x_mark.float().to(device),
                )
                pred_from_retrieval = pred_from_retrieval.cpu()
                retrievals.append(pred_from_retrieval)
                if topk_sims is not None:
                    topk_sims_all.append(topk_sims)
                topm_diag_all.append(topm_diag)
                
        retrievals = torch.cat(retrievals, dim=1)

        # Print the retrieval-quality diagnostic for this split.
        diag = torch.cat(topm_diag_all, dim=0)  # n_samples, topm
        key = 'FAR' if self.use_far else 'corr'
        print(
            f"[retrieval-diag][{key}][{tag or ('train' if train else 'eval')}] "
            f"top-m sim mean={diag.mean():.4f} std={diag.std():.4f} "
            f"top1={diag[:, 0].mean():.4f} topm={diag[:, -1].mean():.4f} "
            f"gap(top1-topm)={ (diag[:, 0] - diag[:, -1]).mean():.4f}"
        )

        if len(topk_sims_all) > 0:
            topk_sims_all = torch.cat(topk_sims_all, dim=0)  # n_samples, topm
            return retrievals, topk_sims_all

        return retrievals, None
