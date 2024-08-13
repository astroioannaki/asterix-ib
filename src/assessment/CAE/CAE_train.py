import sys
from CAE_tools import * 
import numpy as np
from tqdm import tqdm
from torch import nn, optim
import sys
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from torch.multiprocessing import Process
import torch.multiprocessing as mp


def main():
    mp.set_start_method('spawn')


    dist.init_process_group(backend='nccl')
    rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])
    torch.cuda.set_device(rank)

    filename=sys.argv[1]
    sparsity=float(sys.argv[2])
    device ='cuda'

    # Model args
    in_channels, out_channels = 1, 1
    latent_dim = 200
    depth, height, width = 50, 50, 50

    # Initialize model
    encoder = Encoder(in_channels=in_channels, out_channels=out_channels, latent_dim=latent_dim)
    decoder = Decoder(in_channels=in_channels, out_channels=out_channels, latent_dim=latent_dim)

    autoencoder = Autoencoder(encoder, decoder)
    autoencoder.to(device)

    autoencoder = DDP(autoencoder)

    batch_size =128
    epochs = 100
    workers = 4
    f = pt.vlsvfile.VlsvReader(filename)
    size = f.get_velocity_mesh_size()
    WID = f.get_WID()
    box=int(WID*size[0])
    cids=f.read(mesh="SpatialGrid",name="CellID", tag="VARIABLE")
    VDF_Data = Lazy_Vlasiator_DataSet(cids,filename,device,box,sparsity)
    # train_sampler = DistributedSampler(VDF_Data, rank=rank,device_ids=[rank])
    train_sampler = DistributedSampler(VDF_Data)
    train_loader = DataLoader(
        dataset=VDF_Data,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=True
    )

    # Initialize optimizer
    train_params = [params for params in autoencoder.parameters()]
    lr = 3e-4
    optimizer = optim.Adam(train_params, lr=lr)
    criterion = nn.MSELoss()

    # Train model
    eval_every = 1
    best_train_loss = float("inf")
    autoencoder.train()

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        total_recon_error = 0
        n_train = 0
        for batch_idx, train_tensors in enumerate(train_loader):
            optimizer.zero_grad()
            imgs = train_tensors[0].unsqueeze(0).to(rank)
            out = autoencoder(imgs)
            recon_error = criterion(out["x_recon"], imgs)
            total_recon_error += recon_error.item()
            loss = recon_error

            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            n_train += 1

            if ((batch_idx + 1) % eval_every) == 0 and global_rank == 0:
                print(f"epoch: {epoch}\nbatch_idx: {batch_idx + 1}", flush=True)
                avg_train_loss = total_recon_error / n_train
                if avg_train_loss < best_train_loss:
                    best_train_loss = avg_train_loss

                print(f"best_train_loss: {best_train_loss}")
                print(f"recon_error: {total_recon_error / n_train}\n")
                total_recon_error = 0
                n_train = 0
            if (epoch%10==0 and global_rank==0):
                torch.save(autoencoder.state_dict(), f"model_state_{epoch}.ptch")

    if global_rank==0:
        torch.save(autoencoder.state_dict(), "model_state_final.ptch")

if __name__ == "__main__":
    main()