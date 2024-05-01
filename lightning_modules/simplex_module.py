import torch
# from matplotlib import pyplot as plt
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning_modules.general_module import GeneralModule
from utils.logging import get_logger
from sklearn.cluster import KMeans

import numpy as np
import os, copy

from torch import nn
import torch.nn.functional as F

from model.dna_models import CNNModel3D, CNNModel2D, MLPModel
from utils.flow_utils import DirichletConditionalFlow, simplex_proj
from collections import Counter

import scipy
def sample_cond_prob_path(hyperparams, seq, channels):
    shape = seq.shape
    batchsize = seq.shape[0]
    seq_onehot = torch.nn.functional.one_hot(seq.reshape(-1), num_classes=channels).reshape(*shape, channels)
    t = torch.from_numpy(scipy.stats.expon().rvs(size=batchsize)*hyperparams.time_scale).to(seq.device).float()
    alphas = torch.ones(*shape, channels, device=seq.device)*hyperparams.time0_scale
    alphas = alphas + t[:, None, None, None, None]*seq_onehot
    xt = torch.distributions.Dirichlet(alphas).sample()
    return xt, t+1

def sample_cond_prob_path_2d(hyperparams, seq, channels):
    shape = seq.shape
    batchsize = seq.shape[0]
    seq_onehot = torch.nn.functional.one_hot(seq.reshape(-1), num_classes=channels).reshape(*shape, channels)
    t = torch.from_numpy(scipy.stats.expon().rvs(size=batchsize)*hyperparams.time_scale).to(seq.device).float()
    alphas = torch.ones(*shape, channels, device=seq.device)*hyperparams.time0_scale
    alphas = alphas + t[:, None, None, None]*seq_onehot
    xt = torch.distributions.Dirichlet(alphas).sample()
    return xt, t+1


def RC(logits):
    assert logits.shape[-1] == 2
    B = logits.shape[0]
    RC = torch.sum(logits*torch.tensor([-1,1])[None,None,None,:], dim=-1)
    RC = torch.sum(RC.reshape(B, -1), dim=-1)
    return RC.reshape(-1,1)

class UmbrellaSampling():
    def __init__(self, RC):
        self.RC = RC
    
    def buffer2rc_trajs(self, x):
        self.rc_trajs = self.RC(x)

    def mbar(self, rc_min=None, rc_max=None, rc_bins=31, return_fes=False):
        """ Estimates free energy along reaction coordinate with binless WHAM / MBAR.

        Parameters
        ----------
        rc_min : float or None
            Minimum bin position. If None, the minimum RC value will be used.
        rc_max : float or None
            Maximum bin position. If None, the maximum RC value will be used.
        rc_bins : int or None
            Number of bins

        Returns
        -------
        bins : array
            Bin positions
        F : array
            Free energy / -log(p) for all bins

        """
        if rc_min is None:
            rc_min = np.concatenate(self.rc_trajs).min()
        if rc_max is None:
            rc_max = np.concatenate(self.rc_trajs).max()
        gmeans = torch.tensor(np.linspace(rc_min, rc_max, rc_bins))
        gstd = (rc_max - rc_min) / rc_bins
        # self.kmat = torch.exp(-(self.rc_trajs - gmeans)*(self.rc_trajs - gmeans) / (2 * gstd * gstd))
        kmat = torch.exp(-(self.rc_trajs - gmeans)*(self.rc_trajs - gmeans) / (2 * gstd * gstd))/ (torch.exp(-(self.rc_trajs - gmeans)*(self.rc_trajs - gmeans) / (2 * gstd * gstd))).sum(axis=1).reshape(-1,1)
        kmat += 1e-6
        histogram = kmat.mean(axis=0).requires_grad_(True)  
        # del norm_kmat
        FES = -torch.log(histogram)
        print("gmeans = ", gmeans)
        print("FES (kBT) = ", FES)
        F = torch.sum(FES)
        # del FES
        print("F_RC = ", F)
        if return_fes:
            return F, gmeans, FES, histogram
        else:
            return F

magn_ref = torch.tensor([-16.00, -14.00, -12.00, -10.00, -8.00, -6.00, -4.00, -2.00, 0.00, 2.00, 4.00, 6.00, 8.00, 10.00, 12.00, 14.00, 16.00,])
FES_ref = torch.tensor([16.9458,  13.8868,  12.2251,  11.3112,  10.7552,  10.4042,  10.1792,  10.0470,  9.9818,  10.0504,  10.1651,  10.3730,  10.6944,  11.1748,  11.9635,  13.5270,  16.7959])

class simplexModule(GeneralModule):
    def __init__(self, channels, num_cls, hyperparams):
        super().__init__(hyperparams)
        self.load_model(channels, num_cls, hyperparams)
        self.condflow = DirichletConditionalFlow(K=self.model.alphabet_size, alpha_spacing=0.001, alpha_max=hyperparams.alpha_max)
        self.hyperparams = hyperparams
        self.US = UmbrellaSampling(RC)

    def load_model(self, channels, num_cls, hyperparams):
        if hyperparams.model == "CNN3D":
            self.model = CNNModel3D(hyperparams, channels, num_cls)
        elif hyperparams.model == "CNN2D":
            self.model = CNNModel2D(hyperparams, channels, num_cls)
        elif hyperparams.model == "MLP":
            self.model = MLPModel(hyperparams, channels, num_cls)
        else:
            raise Exception("Unrecognized model type")


    def validation_step(self, batch, batch_idx):
        self.stage = 'val'
        loss = self.general_step(batch, batch_idx)
        self.log('val_loss', torch.tensor(self._log["val_loss"]).mean(), prog_bar=True)

    def general_step(self, batch, batch_idx=None):
        seq, cls = batch
        seq_symm = -seq+1
        seq = torch.cat([seq, seq_symm])
        cls = torch.cat([cls, cls])
        if self.hyperparams.model == "CNN3D":
            B, H, W, D = seq.shape
            xt, t = sample_cond_prob_path(self.hyperparams, seq, self.model.alphabet_size)
        elif self.hyperparams.model == "CNN2D":
            B, H, W = seq.shape
            xt, t = sample_cond_prob_path_2d(self.hyperparams, seq, self.model.alphabet_size)
        
        # self.plot_probability_path(t, xt)
        logits = self.model(xt, t, cls=cls)
        if self.hyperparams.model == "CNN3D":
            logits = (logits.permute(0,2,3,4,1)).reshape(-1, self.model.alphabet_size)
        elif self.hyperparams.model == "CNN2D":
            logits = (logits.permute(0,2,3,1)).reshape(-1, self.model.alphabet_size)
        # logits.retain_grad()
        
        
        losses = torch.nn.functional.cross_entropy(logits, seq.reshape(-1), reduction='none').reshape(B,-1)
        # norm_logits = torch.nn.functional.softmax(logits, dim=-1).reshape(B, -1)
        # log_logits_conf = torch.sum(torch.log(norm_logits), dim=-1)
        # losses = (torch.exp(log_logits_conf)*(-torch.log(cls)+log_logits_conf)).reshape(-1,1)*tempscale_list
        # losses = (cls*(-log_logits_conf)).reshape(-1,1)*tempscale_list
        # loss = losses.mean()
        # loss.backward()

        if self.hyperparams.mode == "US":
            self.US.buffer2rc_trajs(norm_logits)
            us_loss = self.US.mbar(rc_min=-16, rc_max=16, rc_bins=17)
            losses += us_loss

        
        if self.hyperparams.mode == "focal":
            norm_logits = torch.nn.functional.softmax(logits)
            fl = -(torch.pow((1-norm_logits), self.hyperparams.gamma_focal)*torch.log(norm_logits+1e-9)).sum(-1)
            losses += fl.reshape(B, -1)
        else:
            pass

        losses = losses.mean(-1)
        self.lg("loss", losses)

        if self.stage == "val":
            if self.hyperparams.model == "CNN3D":
                logits_pred, _ = self.dirichlet_flow_inference(seq)
            elif self.hyperparams.model == "CNN2D":
                logits_pred, _ = self.dirichlet_flow_inference_2d(seq)
            seq_pred = torch.argmax(logits_pred, dim=-1)
            np.save(os.path.join(os.environ["work_dir"], f"seq_val_step{self.trainer.global_step}"), seq_pred.cpu())
            np.save(os.path.join(os.environ["work_dir"], f"logits_val_step{self.trainer.global_step}"), logits_pred.cpu())
        return losses.mean()
    
    def training_step(self, batch, batch_idx):
        self.stage = "train"
        loss = self.general_step(batch)
        # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.stage = "val"
        loss = self.general_step(batch, batch_idx)


    @torch.no_grad()
    def dirichlet_flow_inference_2d(self, seq):

        B, H, W = seq.shape
        K = self.model.alphabet_size
        x0 = torch.distributions.Dirichlet(torch.ones(B, H, W, K, device=seq.device)*self.hyperparams.time0_scale).sample()
        eye = torch.eye(K).to(x0)
        xt = x0.clone()
        np.save(os.path.join(os.environ["work_dir"], f"logits_val_step{self.trainer.global_step}_inttime{1.00}"), xt.cpu())

        t_span = torch.linspace(1, self.hyperparams.alpha_max, self.hyperparams.num_integration_steps, device = self.device)
        for i, (s, t) in enumerate(zip(t_span[:-1], t_span[1:])):
            logits = self.model(xt, t=s[None].expand(B))
            flow_probs = torch.nn.functional.softmax(logits.permute(0,2,3,1)/self.hyperparams.flow_temp, -1)

            if not torch.allclose((flow_probs.reshape(B,-1,K)).sum(2), torch.ones((B, H*W), device=self.device), atol=1e-4) or not (flow_probs >= 0).all():
                print(f'WARNING: flow_probs.min(): {flow_probs.min()}. Some values of flow_probs do not lie on the simplex. There are we are {(flow_probs<0).sum()} negative values in flow_probs of shape {flow_probs.shape} that are negative. We are projecting them onto the simplex.')
                flow_probs = simplex_proj(flow_probs.reshape(B,-1)).reshape(B,H,W,K)

            c_factor = self.condflow.c_factor(xt.cpu().numpy(), s.item())
            c_factor = torch.from_numpy(c_factor).to(xt)


            # self.inf_counter += 1
            if torch.isnan(c_factor).any():
                print(f'NAN cfactor after: xt.min(): {xt.min()}, flow_probs.min(): {flow_probs.min()}')
                if self.hyperparams.allow_nan_cfactor:
                    c_factor = torch.nan_to_num(c_factor)
                    # self.nan_inf_counter += 1
                else:
                    raise RuntimeError(f'NAN cfactor after: xt.min(): {xt.min()}, flow_probs.min(): {flow_probs.min()}')

            if not (flow_probs >= 0).all(): print(f'flow_probs.min(): {flow_probs.min()}')
            cond_flows = (eye - xt.unsqueeze(-1)) * c_factor.unsqueeze(-2)
            # V=U*P: flow = conditional_flow*probability_path
            flow = (flow_probs.unsqueeze(-2) * cond_flows).sum(-1)

            xt = xt + flow * (t - s)
            if not torch.allclose((xt.reshape(B,-1,K)).sum(2), torch.ones((B, H*W), device=self.device), atol=1e-4) or not (xt >= 0).all():
                print(f'WARNING: xt.min(): {xt.min()}. Some values of xt do not lie on the simplex. There are we are {(xt<0).sum()} negative values in xt of shape {xt.shape} that are negative. We are projecting them onto the simplex.')
                xt = simplex_proj(xt.reshape(B,-1)).reshape(B,H,W,K)
            np.save(os.path.join(os.environ["work_dir"], f"logits_val_step{self.trainer.global_step}_inttime{t}"), xt.cpu())
               
        return xt, x0

    @torch.no_grad()
    def dirichlet_flow_inference(self, seq):

        B, H, W, D = seq.shape
        K = self.model.alphabet_size
        x0 = torch.distributions.Dirichlet(torch.ones(B, H, W, D, K, device=seq.device)*self.hyperparams.time0_scale).sample()
        eye = torch.eye(K).to(x0)
        xt = x0.clone()

        t_span = torch.linspace(1, self.hyperparams.alpha_max, self.hyperparams.num_integration_steps, device = self.device)
        for i, (s, t) in enumerate(zip(t_span[:-1], t_span[1:])):
            logits = self.model(xt, t=s[None].expand(B))
            flow_probs = torch.nn.functional.softmax(logits.permute(0,2,3,4,1)/self.hyperparams.flow_temp, -1)

            if not torch.allclose((flow_probs.reshape(B,-1,K)).sum(2), torch.ones((B, H*W*D), device=self.device), atol=1e-4) or not (flow_probs >= 0).all():
                print(f'WARNING: flow_probs.min(): {flow_probs.min()}. Some values of flow_probs do not lie on the simplex. There are we are {(flow_probs<0).sum()} negative values in flow_probs of shape {flow_probs.shape} that are negative. We are projecting them onto the simplex.')
                flow_probs = simplex_proj(flow_probs.reshape(B,-1)).reshape(B,H,W,D,K)

            c_factor = self.condflow.c_factor(xt.cpu().numpy(), s.item())
            c_factor = torch.from_numpy(c_factor).to(xt)


            # self.inf_counter += 1
            if torch.isnan(c_factor).any():
                print(f'NAN cfactor after: xt.min(): {xt.min()}, flow_probs.min(): {flow_probs.min()}')
                if self.hyperparams.allow_nan_cfactor:
                    c_factor = torch.nan_to_num(c_factor)
                    # self.nan_inf_counter += 1
                else:
                    raise RuntimeError(f'NAN cfactor after: xt.min(): {xt.min()}, flow_probs.min(): {flow_probs.min()}')

            if not (flow_probs >= 0).all(): print(f'flow_probs.min(): {flow_probs.min()}')
            cond_flows = (eye - xt.unsqueeze(-1)) * c_factor.unsqueeze(-2)
            # V=U*P: flow = conditional_flow*probability_path
            flow = (flow_probs.unsqueeze(-2) * cond_flows).sum(-1)

            xt = xt + flow * (t - s)
            if not torch.allclose((xt.reshape(B,-1,K)).sum(2), torch.ones((B, H*W*D), device=self.device), atol=1e-4) or not (xt >= 0).all():
                print(f'WARNING: xt.min(): {xt.min()}. Some values of xt do not lie on the simplex. There are we are {(xt<0).sum()} negative values in xt of shape {xt.shape} that are negative. We are projecting them onto the simplex.')
                xt = simplex_proj(xt.reshape(B,-1)).reshape(B,H,W,D,K)
               
        return logits, x0

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hyperparams.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
        return optimizer

    def plot_probability_path(self, t, xt):
        pass

    def lg(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        log = self._log
        log[self.stage + "_" + key].extend(data)
