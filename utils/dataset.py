import copy
import pickle

import torch, esm, random, os, json
import numpy as np
from Bio import SeqIO
from collections import Counter



class EnhancerDataset(torch.utils.data.Dataset):
    def __init__(self, args, split='train'):
        all_data = pickle.load(open(f'data/the_code/General/data/Deep{"MEL2" if args.mel_enhancer else "FlyBrain"}_data.pkl', 'rb'))
        self.seqs = torch.argmax(torch.from_numpy(copy.deepcopy(all_data[f'{split}_data'])), dim=-1)
        self.clss = torch.argmax(torch.from_numpy(copy.deepcopy(all_data[f'y_{split}'])), dim=-1)
        self.num_cls = all_data[f'y_{split}'].shape[-1]
        self.alphabet_size = 4

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.clss[idx]


class TwoClassOverfitDataset(torch.utils.data.IterableDataset):
    def __init__(self, args):
        super().__init__()
        self.seq_len = args.toy_seq_len
        self.alphabet_size = args.toy_simplex_dim
        self.num_cls = 2

        if args.cls_ckpt is not None:
            distribution_dict = torch.load(os.path.join(os.path.dirname(args.cls_ckpt), 'overfit_dataset.pt'))
            self.data_class1 = distribution_dict['data_class1']
            self.data_class2 = distribution_dict['data_class2']
        else:
            self.data_class1 = torch.stack([torch.from_numpy(np.random.choice(np.arange(self.alphabet_size), size=args.toy_seq_len, replace=True)) for _ in range(args.toy_num_seq)])
            self.data_class2 = torch.stack([torch.from_numpy(np.random.choice(np.arange(self.alphabet_size), size=args.toy_seq_len, replace=True)) for _ in range(args.toy_num_seq)])
            distribution_dict = {'data_class1': self.data_class1, 'data_class2': self.data_class2}
        torch.save(distribution_dict, os.path.join(os.environ["MODEL_DIR"], 'overfit_dataset.pt'))

    def __len__(self):
        return 10000000000

    def __iter__(self):
        while True:
            if np.random.rand() < 0.5:
                yield self.data_class1[np.random.choice(np.arange(len(self.data_class1)))], torch.tensor([0])
            else:
                yield self.data_class2[np.random.choice(np.arange(len(self.data_class2)))], torch.tensor([1])

class ToyDataset(torch.utils.data.IterableDataset):
    def __init__(self, args):
        super().__init__()
        self.num_cls = args.toy_num_cls
        self.seq_len = args.toy_seq_len
        self.alphabet_size = args.toy_simplex_dim

        if args.cls_ckpt is not None:
            distribution_dict = torch.load(os.path.join(os.path.dirname(args.cls_ckpt), 'toy_distribution_dict.pt'))
            self.probs = distribution_dict['probs']
            self.class_probs = distribution_dict['class_probs']
        else:
            self.probs = torch.softmax(torch.rand((self.num_cls, self.seq_len, self.alphabet_size)), dim=2)
            self.class_probs = torch.ones(self.num_cls)
            if self.num_cls > 1:
                self.class_probs = self.class_probs * 1 / 2 / (self.num_cls - 1)
                self.class_probs[0] = 1 / 2
            assert self.class_probs.sum() == 1

            distribution_dict = {'probs': self.probs, 'class_probs': self.class_probs}
        torch.save(distribution_dict, os.path.join(os.environ["MODEL_DIR"], 'toy_distribution_dict.pt' ))

    def __len__(self):
        return 10000000000
    def __iter__(self):
        while True:
            cls = np.random.choice(a=self.num_cls,size=1,p=self.class_probs)
            seq = []
            for i in range(self.seq_len):
                seq.append(torch.multinomial(replacement=True,num_samples=1,input=self.probs[cls,i,:]))
            yield torch.tensor(seq), cls


def pbc(i,L=4):
    assert i>=-1 and i<=L
    if i-L == 0:
        return 0
    elif i == -1:
        return L-1
    else:
        return i
    

def ising_boltzman_prob(seq, J=1, kBT=4.0):
    shape = seq.shape
    spins = seq.clone().detach()
    spins[torch.where(spins==0)]=-1
    B,H,W = shape
    E = torch.zeros(B)
    for i in range(H):
        for j in range(W):
            E += -spins[:,i,j]*spins[:,pbc(i-1),j]*J
            E += -spins[:,i,j]*spins[:,pbc(i+1),j]*J
            E += -spins[:,i,j]*spins[:,i,pbc(j-1)]*J
            E += -spins[:,i,j]*spins[:,i,pbc(j+1)]*J

    E /= 2
    prob = torch.exp(-E/kBT)
    return prob, E/kBT


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

class IsingDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()
        all_data = torch.from_numpy(np.load(f"data/{args.dataset_dir}/buffer.npy"))
        print("loaded ", all_data.shape, all_data.dtype)
        self.num_cls = 1
        self.seq_len = args.toy_seq_len
        self.alphabet_size = args.toy_simplex_dim

        if args.cls_ckpt is not None:
            distribution_dict = torch.load(os.path.join(os.path.dirname(args.cls_ckpt), 'toy_distribution_dict.pt'))
            self.probs = distribution_dict['probs']
            self.class_probs = distribution_dict['class_probs']
        else:
            self.seqs = all_data.reshape(-1, *args.toy_seq_dim).to(torch.int64)
            self.seqs[torch.where(self.seqs == -1)] = 0
            # self.clss = torch.full_like(self.seqs, 0)
            prob,scaled_energy = ising_boltzman_prob(self.seqs)
            self.clss = prob
            # distribution_dict = {'probs': self.probs, 'class_probs': self.class_probs}
        # torch.save(distribution_dict, os.path.join(os.environ["MODEL_DIR"], 'toy_distribution_dict.pt' ))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.clss[idx]
    


class AlCuDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()
        all_data = torch.from_numpy(np.load(f"data/{args.dataset_dir}/buffer_atypes.npy").reshape(-1,args.toy_simplex_dim,args.toy_seq_len))
        print("loaded ", all_data.shape)
        self.num_cls = 1
        self.seq_len = args.toy_seq_len
        self.alphabet_size = args.toy_simplex_dim

        if args.cls_ckpt is not None:
            distribution_dict = torch.load(os.path.join(os.path.dirname(args.cls_ckpt), 'toy_distribution_dict.pt'))
            self.probs = distribution_dict['probs']
            self.class_probs = distribution_dict['class_probs']
        else:
            # self.seqs_T0 = torch.softmax(torch.swapaxes(all_data, 1, 2), dim=2)
            # if args.dataset_scaleTemp:
            #     print("Rescaling dataset from 620K to 420K")
            #     self.seqs = torch.pow(self.seqs_T0, 620.0/420.0)
            # else:
            #     self.seqs = self.seqs_T0
    
            self.seqs = torch.argmax(torch.swapaxes(all_data, 1, 2), dim=2).reshape(-1, *args.toy_seq_dim)
            # self.clss = torch.argmax(torch.swapaxes(all_data, 1, 2), dim=2)
            self.clss = torch.full_like(self.seqs, 0)

            # from sklearn.cluster import KMeans
            # est2 = KMeans(n_clusters=2)
            # est2.fit(self.clss)
            # counts_labels = Counter(est2.labels_)
            # counts = torch.tensor([counts_labels[k] for k in [0,1]]).reshape(self.num_cls, 1,-1)
            # self.probs = counts / counts.sum(dim=-1, keepdim=True)
            # print("probs = ", self.probs)

            # self.class_probs = torch.softmax(torch.swapaxes(all_data, 1, 2), dim=2)

            # distribution_dict = {'probs': self.probs, 'class_probs': self.class_probs}
        # torch.save(distribution_dict, os.path.join(os.environ["MODEL_DIR"], 'toy_distribution_dict.pt' ))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.clss[idx]