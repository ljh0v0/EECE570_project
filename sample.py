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
import lib.dataset as dataset
import numpy  as np
import torch.nn.functional as F
from tool.metrics import calculate_ssim, calculate_psnr

from diffusion_lightning import DDP, obj, samples_fn

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.")
    parser.add_argument("--model_dir", type=str, default='', help="Path to model for loading.")
    parser.add_argument("--sample_dir", type=str, default='samples', help="Path to save generated samples.")
    parser.add_argument("--exp_dir", type=str, default='exp', help="Path to experiments.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size of generated samples in evaluation.")
    parser.add_argument("--n_samples", type=int, default=32, help="Number of generated samples in evaluation.")

    args = parser.parse_args()

    '''class Args():
        train = True
        config = "config/diffusion_celeba.json"
        ckpt_dir = "exp/celeba/"
        ckpt_freq = 20
        n_gpu = 1'''


    # class Args():
    #     config = "config/diffusion_celeba.json"
    #     model_dir = "exp/celeba/fullset/epoch=19-step=193279.ckpt"
    #     exp_dir = "exp/celeba/fullset"
    #     sample_dir = 'exp/celeba/fullset/samples'
    #     batch_size = 2
    #     n_samples = 2
    #
    #
    # args = Args()

    path_to_config = args.config
    with open(path_to_config, 'r') as f:
        conf = json.load(f)

    conf = obj(conf)
    denoising_diffusion_model = DDP(conf, exp_dir=args.exp_dir)

    if not os.path.isdir(args.sample_dir):
        os.makedirs(args.sample_dir)

    train_set, valid_set = dataset.get_train_data(conf)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=conf.training.dataloader.drop_last)
    train_iterator = cycle(train_loader)

    denoising_diffusion_model.cuda()
    state_dict = torch.load(args.model_dir)
    denoising_diffusion_model.load_state_dict(state_dict['state_dict'])
    denoising_diffusion_model.eval()

    ds=[]
    out = []
    avg_psnr = 0.0
    avg_ssim = 0.0

    for k in tqdm(range(int(args.n_samples // args.batch_size) + 1)):
        batch, _ = next(train_iterator)
        batch = batch.cuda()
        cond = F.interpolate(batch, size=(16, 16), mode='area')
        sample = samples_fn(denoising_diffusion_model.ema,
                            denoising_diffusion_model.diffusion,
                            (args.batch_size, 3, conf.dataset.resolution, conf.dataset.resolution),
                            cond=cond)
        imgs = sample['samples'].cpu().view(args.batch_size, 3, conf.dataset.resolution, conf.dataset.resolution)
        out.append(imgs)
        ds.append(batch)

        if k < 1:
            filepath = os.path.join(args.sample_dir, f'sample_imgs.png')
            save_image(imgs, filepath, normalize=True, scale_each=True, nrow=4)
        batch = batch.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        imgs = imgs.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        for i in range(args.batch_size):
            psnr = calculate_psnr(imgs[i], batch[i])
            ssim = calculate_ssim(imgs[i], batch[i])
            avg_psnr += psnr
            avg_ssim += ssim

    out_pt = torch.cat(out)

    avg_psnr /= len(out_pt)
    avg_ssim /= len(out_pt)
    print('# Validation # PSNR: {:.4e}'.format(avg_psnr))
    print('# Validation # SSIM: {:.4e}'.format(avg_ssim))


    out = out_pt.cpu().numpy()
    sample_path = os.path.join(args.sample_dir, f'sample_npy.npy')
    np.save(sample_path, out)

    ds_pt = torch.cat(ds)
    ds = ds_pt.cpu().numpy()
    ds_path = os.path.join(args.sample_dir, f'dataset_npy.npy')
    np.save(ds_path, ds)
