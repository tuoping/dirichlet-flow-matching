
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import torch
import os,sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

# os.environ["MODEL_DIR"]="logs-local"
os.environ["MODEL_DIR"]=f"logs-gaussian-ising/latt4x4-batch1024-model4_b_rerun/"
os.environ["work_dir"]=os.path.join(os.environ["MODEL_DIR"], f"val_time0_scale_T0overTk/epoch{sys.argv[1]}")
# os.environ["work_dir"]=os.path.join(os.environ["MODEL_DIR"], f"val_baseline/epoch{sys.argv[1]}")
dataset_dir = "ising-latt4x4-T4.0"

stage = "val"
channels = 2
seq_len = 4*4
seq_dim = (4, 4)
ckpt = None
import glob
ckpt = glob.glob(os.path.join(os.environ["MODEL_DIR"], f"model-epoch={sys.argv[1]}-train_loss=*"))[0]
if stage == "train":
    batch_size = 1024
    if ckpt is not None: 
        print("Starting from ckpt:: ", ckpt)
elif stage == "val":
    batch_size = 4096
    if ckpt is None: 
        raise Exception("ERROR:: ckpt not initiated")
    print("Validating with ckpt::", ckpt)
else:
    raise Exception("Unrecognized stage")
num_workers = 2
max_steps = 100000
max_epochs = 100000
limit_train_batches = None
if stage == "train":
    limit_val_batches = 0.0
else:
    limit_val_batches = 100
grad_clip = 1.
wandb = False
check_val_every_n_epoch = None
val_check_interval = None

trainer = pl.Trainer(
    default_root_dir=os.environ["work_dir"],
    accelerator="gpu" if torch.cuda.is_available() else 'auto',
    max_steps=max_steps,
    max_epochs=max_epochs,
    num_sanity_val_steps=0,
    limit_train_batches=limit_train_batches,
    limit_val_batches=limit_val_batches,
    enable_progress_bar=not (wandb) or os.getlogin() == 'ping-tuo',
    gradient_clip_val=grad_clip,
    callbacks=[
        ModelCheckpoint(
            dirpath=os.environ["MODEL_DIR"],
            filename='model-{epoch:02d}-{train_loss:.2f}',
            save_top_k=-1,  # Save the top 3 models
            monitor='train_loss',  # Monitor validation loss
            mode='min',  # Minimize validation loss
            every_n_train_steps=5000,  # Checkpoint every 1000 training steps
        )
    ],
    check_val_every_n_epoch=check_val_every_n_epoch,
    val_check_interval=val_check_interval,
    log_every_n_steps=1,
    precision=16,
    strategy='ddp_find_unused_parameters_true'
)

class dataset_params():
    def __init__(self, toy_seq_len, toy_seq_dim, toy_simplex_dim, dataset_dir):
        self.toy_seq_len = toy_seq_len
        self.toy_seq_dim = toy_seq_dim
        self.toy_simplex_dim = toy_simplex_dim
        self.dataset_dir = dataset_dir
        self.cls_ckpt = None
        
dparams = dataset_params(seq_len, seq_dim, channels, dataset_dir)

from utils.dataset import AlCuDataset, IsingDataset
dparams.dataset_dir = dataset_dir
# train_ds = AlCuDataset(dparams)
train_ds = IsingDataset(dparams)
dparams.dataset_dir = os.path.join(dataset_dir, "val")
# val_ds = AlCuDataset(dparams)
val_ds = IsingDataset(dparams)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

class Hyperparams():
    def __init__(self, mode=None, hidden_dim=16, num_cnn_stacks=1, lr=5e-4, dropout=0.0, cls_free_guidance=False, clean_data=False, model="MLP"):
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.cls_free_guidance = cls_free_guidance
        self.clean_data = clean_data
        self.num_cnn_stacks = num_cnn_stacks
        self.lr = lr
        self.wandb = False
        self.seq_dim = seq_dim
        self.channels = channels
        self.model = model
        self.mode = mode

    def simplex_params(self, cls_expanded_simplex=False, time_scale=2, time0_scale = 1):
        self.cls_expanded_simplex = cls_expanded_simplex
        self.time_scale = time_scale
        self.alpha_max = 8
        self.num_integration_steps = 20
        self.flow_temp = 1.
        self.allow_nan_cfactor = True
        self.time0_scale = time0_scale

    def gaussian_params(self, time_scale=2, time0_scale = 1):
        self.sigma_min = 0.0001
        self.time_scale = time_scale
        self.time0_scale = time0_scale
        self.num_integration_steps = 20

hparams = Hyperparams(clean_data=True, num_cnn_stacks=3, hidden_dim=int(128), model="CNN2D")
hparams.gaussian_params()

if "time0_scale" in os.environ["work_dir"]:
    import numpy as np
    hparams.time0_scale = np.sqrt((2.0)/(float(sys.argv[2])))

from lightning_modules.gaussian_module import gaussianModule
model = gaussianModule(channels, num_cls=2, hyperparams=hparams)

if stage == "train":
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt)
else:
    trainer.validate(model, val_loader, ckpt_path=ckpt)
