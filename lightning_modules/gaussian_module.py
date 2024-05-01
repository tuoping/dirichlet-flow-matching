import torch
from matplotlib import pyplot as plt
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

import scipy
def sample_cond_vector_field(hyperparams, seq, channels):
    shape = seq.shape
    batchsize = seq.shape[0]
    seq_onehot = torch.nn.functional.one_hot(seq.reshape(-1), num_classes=channels).reshape(*shape, channels)
    sample_x = torch.randn(size=seq_onehot.shape, device=seq.device)
    
    t = 1 - (torch.from_numpy(scipy.stats.expon().rvs(size=batchsize)*hyperparams.time_scale).to(seq.device).float())
    sigma_t = 1-(1-hyperparams.sigma_min)*t
    sample_x *= sigma_t[:,None,None,None,None]
    sample_x += t[:,None,None,None,None]*seq_onehot

    ut = (seq_onehot - (1-hyperparams.sigma_min)*sample_x)/sigma_t[:,None,None,None,None]
    return sample_x, t, ut.float()

def sample_cond_vector_field_2d(hyperparams, seq, channels):
    shape = seq.shape
    batchsize = seq.shape[0]
    seq_onehot = torch.nn.functional.one_hot(seq.reshape(-1), num_classes=channels).reshape(*shape, channels)
    sample_x = torch.randn(size=seq_onehot.shape, device=seq.device)
    
    t = 1 - (torch.from_numpy(scipy.stats.expon().rvs(size=batchsize)*hyperparams.time_scale).to(seq.device).float())
    sigma_t = 1-(1-hyperparams.sigma_min)*t
    sample_x *= sigma_t[:,None,None,None]
    sample_x += t[:,None,None,None]*seq_onehot

    ut = (seq_onehot - (1-hyperparams.sigma_min)*sample_x)/sigma_t[:,None,None,None]
    sample_x.requires_grad = False
    return sample_x, t, ut.float()

class gaussianModule(GeneralModule):
    def __init__(self, channels, num_cls, hyperparams):
        super().__init__(hyperparams)
        self.load_model(channels, num_cls, hyperparams)
        self.hyperparams = hyperparams

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
        B = seq.shape[0]
        if self.hyperparams.model == "CNN3D":
            xt, t, ut = sample_cond_vector_field(self.hyperparams, seq, self.model.alphabet_size)
        elif self.hyperparams.model == "CNN2D":
            xt, t, ut = sample_cond_vector_field_2d(self.hyperparams, seq, self.model.alphabet_size)

        # self.plot_probability_path(t, xt)
        ut_model = self.model(xt, t, cls=cls)
        # losses = torch.nn.functional.cross_entropy((logits.permute(0,2,3,4,1)).reshape(-1, self.model.alphabet_size), seq.reshape(-1), reduction='none').reshape(B,-1)
        if self.hyperparams.model == "CNN3D":
            losses = torch.norm((ut_model.permute(0,2,3,4,1)).reshape(B, -1, self.model.alphabet_size) - ut.reshape(B, -1, self.model.alphabet_size), dim=-1)**2/2.
        elif self.hyperparams.model == "CNN2D":
            losses = torch.norm((ut_model.permute(0,2,3,1)).reshape(B, -1, self.model.alphabet_size) - ut.reshape(B, -1, self.model.alphabet_size), dim=-1)**2/2.


        if self.hyperparams.mode == "focal":
            norm_xt = torch.nn.functional.softmax(xt, dim=-1)
            fl = ((torch.pow(norm_xt, self.hyperparams.gamma_focal).sum(-1)).reshape(B, -1))*torch.norm((ut_model.permute(0,2,3,1)).reshape(B, -1, self.model.alphabet_size) - ut.reshape(B, -1, self.model.alphabet_size), dim=-1)**2/2.
            losses += fl

        losses = losses.mean(-1)
        self.lg("loss", losses)

        if self.stage == "val":
            if self.hyperparams.model == "CNN3D":
                logits_pred = self.gaussian_flow_inference(seq)
            elif self.hyperparams.model == "CNN2D":
                logits_pred = self.gaussian_flow_inference_2d(seq)

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
    def gaussian_flow_inference_2d(self, seq):
        B, H, W = seq.shape
        K = self.model.alphabet_size
        xx = torch.normal(0, 1*self.hyperparams.time0_scale, size=(B,H,W,K), device=self.device)
        xx_t = []
        xx_t.append(xx)

        t_span = torch.linspace(0, 1, self.hyperparams.num_integration_steps, device = self.device)
        for tt in t_span:
            samples_tt = torch.ones(B, device=self.device)*tt
            u_t = self.model(xx, samples_tt)
            xx = xx + u_t.permute(0,2,3,1)*1./self.hyperparams.num_integration_steps
            xx_t.append(xx)
            np.save(os.path.join(os.environ["work_dir"], f"logits_val_step{self.trainer.global_step}_inttime{tt}"), xx.permute(0,3,1,2).cpu())
        return xx_t[-1].permute(0,3,1,2)

    @torch.no_grad()
    def gaussian_flow_inference(self, seq):
        B, H, W, D = seq.shape
        K = self.model.alphabet_size
        xx = torch.normal(0, 1*self.hyperparams.time0_scale, size=(B,H,W,D,K), device=self.device)
        xx_t = []
        xx_t.append(xx)

        t_span = torch.linspace(0, 1, self.hyperparams.num_integration_steps, device = self.device)
        for tt in t_span:
            samples_tt = torch.ones(B, device=self.device)*tt
            u_t = self.model(xx, samples_tt)
            xx = xx + u_t.permute(0,2,3,4,1)*1./self.hyperparams.num_integration_steps
            xx_t.append(xx)
        return xx_t[-1].permute(0,4,1,2,3)

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
