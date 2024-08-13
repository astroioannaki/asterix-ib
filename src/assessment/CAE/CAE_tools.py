import sys
sys.path.append("../")
import vdf_extract
import pytools as pt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import multiprocessing as mp
import numpy as np
import torch
from stocaching import SharedCache


#  defining encoder
class Encoder(nn.Module):
  def __init__(self, in_channels=1, out_channels=1, latent_dim=200, act_fn=nn.ReLU()):
    super().__init__()

    self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=16, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=16),
            nn.ReLU(),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=32),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(),
            nn.Flatten()
        )

  def forward(self, x):
    x = x.view(1, 1, 50, 50, 50)
    print(x.shape)
    output = self.encoder(x)
    print("output = ", output.shape)
    return output


#  defining decoder
class Decoder(nn.Module):
  def __init__(self, in_channels=200, out_channels=1, latent_dim=200, act_fn=nn.ReLU()):
    super().__init__()
    
    # Add a linear layer to map the latent space back to the shape required by the decoder
    self.linear = nn.Sequential(
        nn.Linear(latent_dim, 128 * 6 * 6 * 6),  # Map latent_dim to a size compatible with the decoder
        nn.ReLU()
    )
    
    self.decoder = nn.Sequential(
            #nn.Unflatten(1, (1,1,50,50,50)),
            #nn.BatchNorm3d(num_features=128),
            #nn.ReLU(),
            nn.ConvTranspose3d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=32),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=16),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=16, out_channels=in_channels, kernel_size=4, stride=1, padding=0), # dimensions should be as original
            nn.BatchNorm3d(num_features=in_channels))

  def forward(self, x):
    #output = self.linear(x)
    output = x.view(-1, 128, 35, 35, 35)
    output = self.decoder(output)
    return output


#  defining autoencoder
class Autoencoder(nn.Module):
  def __init__(self, encoder, decoder):
    super().__init__()
    self.encoder = encoder
    self.encoder.to(device)

    self.decoder = decoder
    self.decoder.to(device)

  def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
  
class Lazy_Vlasiator_DataSet:
    def __init__(self, cids, filename, device,box,sparsity, cache_size=256,max_mem_gig=32):
        super().__init__()
        self.cids = cids
        self.box = box
        self.sparsity=sparsity
        self.f = pt.vlsvfile.VlsvReader(filename)
        dataset_len = len(cids) 
        data_dims = (box,box,box) 

        # initialize the cache
        self.cache = SharedCache(
            size_limit_gib=max_mem_gig,
            dataset_len=dataset_len,
            data_dims=data_dims,
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.cids)

    def __getitem__(self, idx):
        x = self.cache.get_slot(idx)
        #Lazy adds to cache
        if x is None:
            _,vdf,_ = vdf_extract.extract(self.f, idx+1, self.sparsity,restrict_box=False)
            vdf=np.array(vdf,dtype=np.float32)
            vdf = (vdf - vdf.min()) / (vdf.max() - vdf.min())
            x = torch.tensor(vdf)
            self.cache.set_slot(idx, x) 
        return x.unsqueeze(0)
    
