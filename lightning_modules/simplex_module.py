import torch
from matplotlib import pyplot as plt
from torch import optim

from lightning_modules.general_module import GeneralModule
from utils.logging import get_logger
from sklearn.cluster import KMeans

import numpy as np
import os, copy

from torch import nn
import torch.nn.functional as F

from model.dna_models import CNNModel3D, MLPModel
from utils.flow_utils import DirichletConditionalFlow, simplex_proj

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

class simplexModule(GeneralModule):
    def __init__(self, channels, num_cls, hyperparams):
        super().__init__(hyperparams)
        self.load_model(channels, num_cls, hyperparams)
        self.condflow = DirichletConditionalFlow(K=self.model.alphabet_size, alpha_spacing=0.001, alpha_max=hyperparams.alpha_max)
        self.hyperparams = hyperparams

    def load_model(self, channels, num_cls, hyperparams):
        if hyperparams.model == "CNN":
            self.model = CNNModel3D(hyperparams, channels, num_cls)
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
        B = seq.shape[0]
        xt, t = sample_cond_prob_path(self.hyperparams, seq, self.model.alphabet_size)
        # self.plot_probability_path(t, xt)
        logits = self.model(xt, t, cls=cls)
        losses = torch.nn.functional.cross_entropy((logits.permute(0,2,3,4,1)).reshape(-1, self.model.alphabet_size), seq.reshape(-1), reduction='none').reshape(B,-1)
        losses = losses.mean(-1)
        self.lg("loss", losses)

        if self.stage == "val":
            logits_pred, _ = self.dirichlet_flow_inference(seq)
            seq_pred = torch.argmax(logits_pred, dim=1)
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
        return optimizer

    def plot_probability_path(self, t, xt):
        pass

    def lg(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        log = self._log
        log[self.stage + "_" + key].extend(data)