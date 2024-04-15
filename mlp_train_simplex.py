
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import torch
import os,sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

# os.environ["MODEL_DIR"]="logs-local"
os.environ["MODEL_DIR"]=f"benchmarks/{sys.argv[2]}hiddendim/hidden{sys.argv[1]}"
dataset_dir = "Al-Cu"

stage = "train"
channels = 2
seq_len = 500
seq_dim = (2*5, 2*5, 5)
ckpt = None
# ckpt = os.path.join(os.environ["MODEL_DIR"], "model-epoch=498-train_loss=0.16.ckpt")
if stage == "train":
    batch_size = 128
    if ckpt is not None: 
        print("Starting from ckpt:: ", ckpt)
elif stage == "val":
    batch_size = 1000
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
    default_root_dir=os.environ["MODEL_DIR"],
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
            save_top_k=3,  # Save the top 3 models
            monitor='train_loss',  # Monitor validation loss
            mode='min',  # Minimize validation loss
            every_n_train_steps=1000,  # Checkpoint every 1000 training steps
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

from utils.dataset import AlCuDataset
dparams.dataset_dir = dataset_dir
train_ds = AlCuDataset(dparams)
dparams.dataset_dir = os.path.join(dataset_dir, "val")
val_ds = AlCuDataset(dparams)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

class Hyperparams():
    def __init__(self, hidden_dim=16, num_cnn_stacks=1, lr=5e-4, dropout=0.0, cls_free_guidance=False, clean_data=False, model="MLP"):
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

    def simplex_params(self, cls_expanded_simplex=False, mode="dirichlet", time_scale=2, time0_scale = 1):
        self.cls_expanded_simplex = cls_expanded_simplex
        self.mode = mode
        self.time_scale = time_scale
        self.alpha_max = 8
        self.num_integration_steps = 20
        self.flow_temp = 1.
        self.allow_nan_cfactor = True
        self.time0_scale = time0_scale

hparams = Hyperparams(clean_data=True, num_cnn_stacks=2, hidden_dim=int(sys.argv[1]), model=sys.argv[2])
hparams.simplex_params()
# hparams.time0_scale = (620.*620.)/(420.*420.)

from lightning_modules.simplex_module import simplexModule
model = simplexModule(channels, num_cls=2, hyperparams=hparams)

if stage == "train":
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt)
else:
    trainer.validate(model, val_loader, ckpt_path=ckpt)