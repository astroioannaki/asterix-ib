import sys
from CAE_tools import * 
import numpy as np
import torch
from torch.utils.data import DataLoader

filename = sys.argv[1] 
model_checkpoint = sys.argv[2]  
sparsity = float(sys.argv[3]) 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model args
in_channels, out_channels = 1, 1
latent_dim = 200
depth, height, width = 50, 50, 50

# Initialize model
encoder = Encoder(in_channels=in_channels, out_channels=out_channels, latent_dim=latent_dim)
decoder = Decoder(in_channels=in_channels, out_channels=out_channels, latent_dim=latent_dim)

autoencoder = Autoencoder(encoder, decoder)
autoencoder.to(device)

ckpt=torch.load(model_checkpoint)
new_ckpt = {}
for k, v in ckpt.items():
    new_ckpt[k.replace('module.', '', 1)] = v
autoencoder.load_state_dict(new_ckpt)
autoencoder.eval()


f = pt.vlsvfile.VlsvReader(filename)
size = f.get_velocity_mesh_size()
WID = f.get_WID()
box=int(WID*size[0])
cids=f.read(mesh="SpatialGrid",name="CellID", tag="VARIABLE") 
print(cids.shape)
VDF_Data = Lazy_Vlasiator_DataSet(cids,filename,device,box,sparsity)

inference_loader = DataLoader(
    dataset=VDF_Data,
    batch_size=1,  
    num_workers=0,
    pin_memory=False
)

predictions = []
with torch.no_grad():
    for batch_idx, inference_tensors in enumerate(inference_loader):
        imgs = inference_tensors[0].unsqueeze(0).to(device)
        out = autoencoder(imgs)
        reconstructions = out["x_recon"].cpu().numpy()
        predictions.append(reconstructions)
        print(f"Inference done for batch {batch_idx + 1}")