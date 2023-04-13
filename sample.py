import os
import json
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import numpy  as np

from diffusion_lightning import DDP, obj, samples_fn

if __name__ == "__main__":

    '''parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.")
    parser.add_argument("--model_dir", type=str, default='', help="Path to model for loading.")
    parser.add_argument("--sample_dir", type=str, default='samples', help="Path to save generated samples.")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of generated samples in evaluation.")
    parser.add_argument("--n_samples", type=int, default=32, help="Number of generated samples in evaluation.")

    args = parser.parse_args()'''

    '''class Args():
        train = True
        config = "config/diffusion_celeba.json"
        ckpt_dir = "exp/celeba/"
        ckpt_freq = 20
        n_gpu = 1'''


    class Args():
        config = "config/diffusion_cifar10.json"
        model_dir = "exp/cifar10/epoch2500-fid8.ckpt"
        sample_dir = 'samples/cifar10/epoch1100/'
        batch_size = 16
        n_samples = 32


    args = Args()

    path_to_config = args.config
    with open(path_to_config, 'r') as f:
        conf = json.load(f)

    conf = obj(conf)
    denoising_diffusion_model = DDP(conf)

    denoising_diffusion_model.cuda()
    state_dict = torch.load(args.model_dir)
    denoising_diffusion_model.load_state_dict(state_dict['state_dict'])
    denoising_diffusion_model.eval()

    out = []

    for k in tqdm(range(int(args.n_samples // args.batch_size))):

        sample = samples_fn(denoising_diffusion_model.ema,
                            denoising_diffusion_model.diffusion,
                            (args.batch_size, 3, conf.dataset.resolution, conf.dataset.resolution))

        imgs = sample['samples'].cpu().view(args.batch_size, 3, conf.dataset.resolution, conf.dataset.resolution)
        out.append(imgs)

        if k<1:
            filepath = os.path.join(args.sample_dir, f'sample_imgs.png')
            save_image(imgs, filepath, normalize=True, scale_each=True, nrow=4)

    out_pt = torch.cat(out)

    out = out_pt.numpy()
    sample_path = os.path.join(args.sample_dir, f'sample_npy.npy')
    np.save(sample_path, out)