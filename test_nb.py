import argparse
import datetime
import os

import torch

from mt.data import create_dataset
from mt.mvae import utils
from mt.mvae.models import Trainer
from mt.utils import str2bool
from mt.mvae.models.nb_vae import NBVAE
import numpy as np


# vary important to prevent NAN
torch.set_default_dtype(torch.float64)

model_name = 's2, s2'
fixed_curvature = True

components = utils.parse_components(model_name, fixed_curvature)

h_dim = 32
scalar_parametrization = False
device = 'cpu'

# dataset = create_dataset('mnist', 100, '/Users/jding/work/mvae/data/')
# train_loader, test_loader = dataset.create_loaders()

# ==
from mt.data.umi_data import UMIVaeDataset
import anndata
adata = anndata.read_loom('/Users/jding/work/project/scdata/hela_select.loom')

x = adata.X.todense().astype(np.double)
y = adata.obs['batch'].rank(method='dense').astype(int)

dataset = UMIVaeDataset(batch_size=128)
train_loader = dataset.create_loaders(x, y)
test_loader = train_loader

for i, data in enumerate(train_loader):
    print((data[0].shape))
import argparse
import datetime
import os

import torch

from mt.data import create_dataset
from mt.mvae import utils
from mt.mvae.models import Trainer
from mt.utils import str2bool
from mt.mvae.models.nb_vae import NBVAE
import numpy as np


# vary important to prevent NAN
torch.set_default_dtype(torch.float64)

model_name = 's2, s2'
fixed_curvature = True

components = utils.parse_components(model_name, fixed_curvature)

h_dim = 32
scalar_parametrization = False
device = 'cpu'

# dataset = create_dataset('mnist', 100, '/Users/jding/work/mvae/data/')
# train_loader, test_loader = dataset.create_loaders()

# ==
from mt.data.umi_data import UMIVaeDataset
import anndata
adata = anndata.read_loom('/Users/jding/work/project/scdata/hela_select.loom')

x = adata.X.todense().astype(np.double)
y = adata.obs['batch'].rank(method='dense').astype(int)

dataset = UMIVaeDataset(batch_size=128)
train_loader = dataset.create_loaders(x, y)
test_loader = train_loader

for i, data in enumerate(train_loader):
    print((data[0].shape))

#
import pandas as pd

mask_cyc = np.zeros(adata.n_vars)

cycle_gene = pd.read_csv('/Users/jding/work/project/scdata/cycle_gene.txt', header=None, sep='\t')
cycle_gene = cycle_gene[0][cycle_gene[0].isin(adata.var['gene_symbols'])]

mask_cyc[adata.var['gene_symbols'].isin(cycle_gene)] = 1

mask_all = np.ones(adata.n_vars)
mask_all[adata.var['gene_symbols'].isin(cycle_gene)] = 0

num_gene = len(mask_cyc)
weight = (1 + np.array((num_gene - mask_cyc.sum(), num_gene - mask_all.sum()))) / num_gene

# mask = torch.tensor([mask_cyc * weight[0], mask_all * weight[1]])
mask = torch.tensor([mask_cyc, mask_all])

# 714 cycling genes
# mask = torch.tensor([np.ones(adata.n_vars), np.ones(adata.n_vars)])
# =====
model = NBVAE(h_dim=h_dim,
              components=components,
              mask=mask,
              dataset=dataset,
              scalar_parametrization=scalar_parametrization).to(device)

cur_time = datetime.datetime.utcnow().isoformat()
chkpt_dir = f"./chkpt/vae-{'mnist'}-{model_name}-{cur_time}"

trainer = Trainer(model,
                  img_dims=[10,10],
                  chkpt_dir=chkpt_dir,
                  train_statistics=0,
                  show_embeddings=False,
                  export_embeddings=False,
                  test_every=0)

optimizer = trainer.build_optimizer(learning_rate=0.001, fixed_curvature=True)

betas = utils.linear_betas(1.0,
                           1.0,
                           end_epoch=1,
                           epochs=200)

trainer.train_stopping(optimizer=optimizer,
                       train_data=train_loader,
                       eval_data=test_loader,
                       warmup=20,
                       lookahead=10,
                       betas=betas,
                       likelihood_n=200,
                       max_epochs=500)

a = trainer.model(torch.log1p(torch.tensor(x)))
b = a[1]

bb = b.detach().numpy()
np.savetxt('/Users/jding/work/project/mvae/all_encode_v63.txt', bb)
#
# #
# import pandas as pd
#
# mask_cyc = np.zeros(adata.n_vars)
#
# cycle_gene = pd.read_csv('/Users/jding/work/scdata/cycle_gene.txt', header=None, sep='\t')
# cycle_gene = cycle_gene[0][cycle_gene[0].isin(adata.var['gene_symbols'])]
#
# mask_cyc[adata.var['gene_symbols'].isin(cycle_gene)] = 1
#
# mask_all = np.ones(adata.n_vars)
# mask_all[adata.var['gene_symbols'].isin(cycle_gene)] = 0
#
# num_gene = len(mask_cyc)
# weight = (1 + np.array((num_gene - mask_cyc.sum(), num_gene - mask_all.sum()))) / num_gene
#
# # mask = torch.tensor([mask_cyc * weight[0], mask_all * weight[1]])
# mask = torch.tensor([mask_cyc, mask_all])
#
# # 714 cycling genes
# # mask = torch.tensor([np.ones(adata.n_vars), np.ones(adata.n_vars)])
# # =====
# model = NBVAE(h_dim=h_dim,
#               components=components,
#               mask=mask,
#               dataset=dataset,
#               scalar_parametrization=scalar_parametrization).to(device)
#
# cur_time = datetime.datetime.utcnow().isoformat()
# chkpt_dir = f"./chkpt/vae-{'mnist'}-{model_name}-{cur_time}"
#
# trainer = Trainer(model,
#                   img_dims=[10,10],
#                   chkpt_dir=chkpt_dir,
#                   train_statistics=0,
#                   show_embeddings=False,
#                   export_embeddings=False,
#                   test_every=0)
#
# optimizer = trainer.build_optimizer(learning_rate=0.001, fixed_curvature=True)
#
# betas = utils.linear_betas(1.0,
#                            1.0,
#                            end_epoch=1,
#                            epochs=200)
#
# trainer.train_stopping(optimizer=optimizer,
#                        train_data=train_loader,
#                        eval_data=test_loader,
#                        warmup=20,
#                        lookahead=10,
#                        betas=betas,
#                        likelihood_n=200,
#                        max_epochs=500)
#
# a = trainer.model(torch.log1p(torch.tensor(x)))
# b = a[1]
#
# bb = b.detach().numpy()
# np.savetxt('/Users/jding/work/mvae/all_encode_v62.txt', bb)


