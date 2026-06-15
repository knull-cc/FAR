import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Retrieval import RetrievalTool
from far.gating import RetrievalGate

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf

    RAFT host augmented with FAR (Future-Aligned Retrieval). When configs.use_far
    is set, the retrieval similarity is computed by a contrastively-trained,
    future-aligned encoder instead of past-waveform correlation. The fusion head
    is unchanged (FAR is a plug-in retriever, not a backbone).
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.device = torch.device(f'cuda:{configs.gpu}') if torch.cuda.is_available() else torch.device('cpu')
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        # Series decomposition block from Autoformer
#         self.decompsition = series_decomp(configs.moving_avg)
#         self.individual = individual
        self.channels = configs.enc_in

        self.linear_x = nn.Linear(self.seq_len, self.pred_len)
        
        self.n_period = configs.n_period
        self.topm = configs.topm

        # ---- FAR configuration -----------------------------------------------
        self.use_far = getattr(configs, 'use_far', False)
        self.use_gating = getattr(configs, 'far_use_gating', False) and self.use_far
        far_config = {
            'emb_dim': getattr(configs, 'far_emb_dim', 128),
            'd_model': getattr(configs, 'far_d_model', 128),
            'n_blocks': getattr(configs, 'far_n_blocks', 3),
            'dropout': getattr(configs, 'far_dropout', 0.1),
            'use_revin': bool(getattr(configs, 'far_use_revin', 1)),
            'temperature': getattr(configs, 'far_temperature', 0.1),
            'pos_k': getattr(configs, 'far_pos_k', 5),
            'future_metric': getattr(configs, 'far_future_metric', 'shape'),
            'soft_dtw_gamma': getattr(configs, 'far_soft_dtw_gamma', 0.1),
            'use_hard_neg': getattr(configs, 'far_use_hard_neg', False),
            'hard_scale': getattr(configs, 'far_hard_scale', 3.0),
            'use_gating': self.use_gating,
            'epochs': getattr(configs, 'far_epochs', 10),
            'batch_size': getattr(configs, 'far_batch_size', 256),
            'lr': getattr(configs, 'far_lr', 1e-3),
        }
        use_covariates = getattr(configs, 'far_use_covariates', False) and self.use_far
        # number of time-feature covariate channels (from data_stamp)
        cov_channels = getattr(configs, 'far_cov_channels', 0)

        self.rt = RetrievalTool(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            channels=self.channels,
            n_period=self.n_period,
            topm=self.topm,
            use_far=self.use_far,
            use_covariates=use_covariates,
            cov_channels=cov_channels,
            far_config=far_config,
        )
        
        self.period_num = self.rt.period_num[-1 * self.n_period:]
        
        module_list = [
            nn.Linear(self.pred_len // g, self.pred_len)
            for g in self.period_num
        ]
        self.retrieval_pred = nn.ModuleList(module_list)
        self.linear_pred = nn.Linear(2 * self.pred_len, self.pred_len)

        # B4: confidence-aware retrieval gate, trained jointly with the head.
        self.gate = RetrievalGate(learnable=True) if self.use_gating else None
        self.topk_sims_dict = {}

#         if self.task_name == 'classification':
#             self.projection = nn.Linear(
#                 configs.enc_in * configs.seq_len, configs.num_class)

    def prepare_dataset(self, train_data, valid_data, test_data):
        self.rt.prepare_dataset(train_data)

        # Train FAR's future-aligned encoder + build the KB embedding index.
        if self.use_far:
            self.rt.train_far(self.device)
        
        self.retrieval_dict = {}
        
        print('Doing Train Retrieval')
        train_rt, train_sims = self.rt.retrieve_all(train_data, train=True, device=self.device)

        print('Doing Valid Retrieval')
        valid_rt, valid_sims = self.rt.retrieve_all(valid_data, train=False, device=self.device)

        print('Doing Test Retrieval')
        test_rt, test_sims = self.rt.retrieve_all(test_data, train=False, device=self.device)

        del self.rt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.retrieval_dict['train'] = train_rt.detach()
        self.retrieval_dict['valid'] = valid_rt.detach()
        self.retrieval_dict['test'] = test_rt.detach()

        if self.use_gating:
            self.topk_sims_dict['train'] = train_sims
            self.topk_sims_dict['valid'] = valid_sims
            self.topk_sims_dict['test'] = test_sims

    def encoder(self, x, index, mode):
        # retrieval_dict is precomputed on CPU to avoid keeping the full
        # retrieval cache on GPU. Index it on CPU, then move only this batch.
        index_cpu = index.to('cpu', dtype=torch.long)
        device = x.device
        
        bsz, seq_len, channels = x.shape
        assert seq_len == self.seq_len and channels == self.channels
        
        x_offset = x[:, -1:, :].detach()
        x_norm = x - x_offset

        x_pred_from_x = self.linear_x(x_norm.permute(0, 2, 1)).permute(0, 2, 1) # B, P, C
        
        pred_from_retrieval = self.retrieval_dict[mode][:, index_cpu].to(device) # G, B, P, C
        
        retrieval_pred_list = []
        
        # Compress repeating dimensions
        for i, pr in enumerate(pred_from_retrieval):
            assert((bsz, self.pred_len, channels) == pr.shape)
            g = self.period_num[i]
            pr = pr.reshape(bsz, self.pred_len // g, g, channels)
            pr = pr[:, :, 0, :]
            
            pr = self.retrieval_pred[i](pr.permute(0, 2, 1)).permute(0, 2, 1)
            pr = pr.reshape(bsz, self.pred_len, self.channels)
            
            retrieval_pred_list.append(pr)

        retrieval_pred_list = torch.stack(retrieval_pred_list, dim=1)
        retrieval_pred_list = retrieval_pred_list.sum(dim=1)

        # B4: down-weight retrieval when no confidently future-aligned neighbor
        # exists (novel regime), up-weight when retrieval is confident.
        if self.use_gating and self.gate is not None:
            topk_sims = self.topk_sims_dict[mode][index_cpu].to(device)  # B, topm
            gate_w = self.gate(topk_sims)  # B, 1
            retrieval_pred_list = retrieval_pred_list * gate_w.unsqueeze(1)
        
        pred = torch.cat([x_pred_from_x, retrieval_pred_list], dim=1)
        pred = self.linear_pred(pred.permute(0, 2, 1)).permute(0, 2, 1).reshape(bsz, self.pred_len, self.channels)
        
        pred = pred + x_offset
        
        return pred

    def forecast(self, x_enc, index, mode):
        # Encoder
        return self.encoder(x_enc, index, mode)

    def imputation(self, x_enc, index, mode):
        # Encoder
        return self.encoder(x_enc, index, mode)

    def anomaly_detection(self, x_enc, index, mode):
        # Encoder
        return self.encoder(x_enc, index, mode)

    def classification(self, x_enc, index, mode):
        # Encoder
        enc_out = self.encoder(x_enc, index, mode)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, index, mode='train'):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, index, mode)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, index, mode)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, index, mode)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, index, mode)
            return dec_out  # [B, N]
        return None
