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


class IsingDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()
        all_data = torch.from_numpy(np.load("data/ising/buffer.npy").reshape(-1,args.toy_simplex_dim,args.toy_seq_len))
        print("loaded ", all_data.shape)
        self.num_cls = 1
        self.seq_len = args.toy_seq_len
        self.alphabet_size = args.toy_simplex_dim

        if args.cls_ckpt is not None:
            distribution_dict = torch.load(os.path.join(os.path.dirname(args.cls_ckpt), 'toy_distribution_dict.pt'))
            self.probs = distribution_dict['probs']
            self.class_probs = distribution_dict['class_probs']
        else:
            self.seqs = torch.argmax(torch.swapaxes(all_data, 1, 2), dim=2)
            self.clss = torch.argmax(torch.swapaxes(all_data, 1, 2), dim=2)

            seqs_one_hot = (torch.nn.functional.one_hot(self.seqs, num_classes=args.toy_simplex_dim)*torch.tensor([1,-1])).sum(-1)
            # seqs_one_hot = (torch.softmax(torch.swapaxes(all_data, 1, 2), dim=2)*torch.tensor([1,-1])).sum(-1)
            self.rc = [-4,-2,0,2,4]
            magn = seqs_one_hot.sum(-1).tolist()
            counts_magn = Counter(magn)
            counts = torch.tensor([counts_magn[k] for k in self.rc] ).reshape(self.num_cls, 1,-1)
            self.probs = counts / counts.sum(dim=-1, keepdim=True)
            print("probs = ", self.probs)
            # self.probs = torch.softmax(torch.swapaxes(all_data, 1, 2), dim=2)
            self.class_probs = torch.softmax(torch.swapaxes(all_data, 1, 2), dim=2)

            print("loaded ", self.probs.shape, self.class_probs.shape)
            distribution_dict = {'probs': self.probs, 'class_probs': self.class_probs}
        torch.save(distribution_dict, os.path.join(os.environ["MODEL_DIR"], 'toy_distribution_dict.pt' ))

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