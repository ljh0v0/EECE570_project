import os
import json
import argparse
import torch
from PIL import Image
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger

from lib.model import UNet
import lib.dataset as dataset
from lib.diffusion import GaussianDiffusion, make_beta_schedule

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)


def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def samples_fn(model, diffusion, shape, cond=None):
    samples = diffusion.p_sample_loop(model=model,
                                      shape=shape,
                                      noise_fn=torch.randn,
                                      cond=cond)
    return {
      'samples': samples
    }


def progressive_samples_fn(model, diffusion, shape, device, include_x0_pred_freq=50):
    samples, progressive_samples = diffusion.p_sample_loop_progressive(
        model=model,
        shape=shape,
        noise_fn=torch.randn,
        device=device,
        include_x0_pred_freq=include_x0_pred_freq
    )
    return {'samples': samples, 'progressive_samples': progressive_samples}


def bpd_fn(model, diffusion, x):
    total_bpd_b, terms_bpd_bt, prior_bpd_b, mse_bt = diffusion.calc_bpd_loop(model=model, x_0=x, clip_denoised=True)

    return {
      'total_bpd': total_bpd_b,
      'terms_bpd': terms_bpd_bt,
      'prior_bpd': prior_bpd_b,
      'mse': mse_bt
    }


def validate(val_loader, model, diffusion):
    model.eval()
    bpd = []
    mse = []
    with torch.no_grad():
        for i, (x, y) in enumerate(iter(val_loader)):
            x       = x
            metrics = bpd_fn(model, diffusion, x)

            bpd.append(metrics['total_bpd'].view(-1, 1))
            mse.append(metrics['mse'].view(-1, 1))

        bpd = torch.cat(bpd, dim=0).mean()
        mse = torch.cat(mse, dim=0).mean()

    return bpd, mse


class DDP(pl.LightningModule):
    def __init__(self, conf, exp_dir):
        super().__init__()

        self.conf  = conf
        self.exp_dir = exp_dir
        self.save_hyperparameters()
        self.n_timesteps = self.conf.model.schedule.n_timestep

        # Disable automatic optimization
        self.automatic_optimization = False
        self.grad_clip_val = conf.training.grad_clip_val

        self.model = UNet(self.conf.model.in_channel,
                          self.conf.model.channel,
                          channel_multiplier=self.conf.model.channel_multiplier,
                          n_res_blocks=self.conf.model.n_res_blocks,
                          attn_strides=self.conf.model.attn_strides,
                          dropout=self.conf.model.dropout,
                          fold=self.conf.model.fold,
                          img_size=self.conf.dataset.resolution,
                          )

        self.ema   = UNet(self.conf.model.in_channel,
                          self.conf.model.channel,
                          channel_multiplier=self.conf.model.channel_multiplier,
                          n_res_blocks=self.conf.model.n_res_blocks,
                          attn_strides=self.conf.model.attn_strides,
                          dropout=self.conf.model.dropout,
                          fold=self.conf.model.fold,
                          img_size=self.conf.dataset.resolution,
                          )

        self.betas = make_beta_schedule(schedule=self.conf.model.schedule.type,
                                        start=self.conf.model.schedule.beta_start,
                                        end=self.conf.model.schedule.beta_end,
                                        n_timestep=self.conf.model.schedule.n_timestep)

        self.diffusion = GaussianDiffusion(betas=self.betas,
                                           model_mean_type=self.conf.model.mean_type,
                                           model_var_type=self.conf.model.var_type,
                                           loss_type=self.conf.model.loss_type)



    def setup(self, stage):

        self.train_set, self.valid_set = dataset.get_train_data(self.conf)

    def forward(self, x):

        return self.diffusion.p_sample_loop(self.model, x.shape)

    def configure_optimizers(self):

        if self.conf.training.optimizer.type == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.conf.training.optimizer.lr)
        else:
            raise NotImplementedError

        # Define the LR scheduler (As in Ho et al.)
        if self.conf.training.n_anneal_steps == 0:
            lr_lambda = lambda step: 1.0
        else:
            lr_lambda = lambda step: min(step / self.conf.training.n_anneal_steps, 1.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "strict": False,
            },
        }

    def training_step(self, batch, batch_nb):
        # Optimizers
        optim = self.optimizers()
        lr_sched = self.lr_schedulers()

        img, _ = batch
        time   = (torch.rand(img.shape[0]) * self.n_timesteps).type(torch.int64).to(img.device)
        cond = F.interpolate(img, size=(16, 16), mode='area')
        loss   = self.diffusion.training_losses(self.model, img, time, cond=cond).mean()

        # Clip gradients and Optimize
        optim.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip_val
        )
        optim.step()

        # Scheduler step
        lr_sched.step()

        accumulate(self.ema, self.model.module if isinstance(self.model, nn.DataParallel) else self.model, 0.9999)

        if self.global_step % self.conf.training.log_loss_every_steps == 0:
            self.logger.log_metrics({"train_loss": loss}, step=self.global_step)

        if self.global_step % self.conf.training.retain_checkpoint_every_steps == 0:
            filename = f"checkpoint_{self.global_step}.ckpt"
            ckpt_path = os.path.join(self.exp_dir, "retain-checkpoint", filename)
            self.trainer.save_checkpoint(ckpt_path)
        if self.global_step % self.conf.training.sample_train_imgs == 0:
            sample = samples_fn(self.model, self.diffusion, img.shape, cond=cond)

            cond_upsample = F.interpolate(cond, img.shape[-2:], mode='bilinear', align_corners=True)
            grid = make_grid(torch.cat((img, cond_upsample, sample['samples']), dim=0),
                             normalize=True,
                             pad_value=0.5,
                             nrow=8)
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            self.logger.experiment.log_image(im, name='train-img-step' + str(self.global_step),
                                             step=self.global_step)

        return {'loss': loss}

    def train_dataloader(self):

        train_loader = DataLoader(self.train_set,
                                  batch_size=self.conf.training.dataloader.batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=self.conf.training.dataloader.drop_last)

        return train_loader

    def validation_step(self, batch, batch_nb):

        img, _ = batch
        time   = (torch.rand(img.shape[0]) * self.n_timesteps).type(torch.int64).to(img.device)
        cond = F.interpolate(img, size=(16, 16), mode='area')
        loss   = self.diffusion.training_losses(self.ema, img, time, cond=cond).mean()
        if batch_nb == 0:
            sample = samples_fn(self.ema, self.diffusion, img.shape, cond=cond)

            cond_upsample = F.interpolate(cond, img.shape[-2:], mode='bilinear', align_corners=True)
            grid = make_grid(torch.cat((img, cond_upsample, sample['samples']), dim=0),
                                 normalize=True,
                                 pad_value=0.5,
                                 nrow=8)
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            self.logger.experiment.log_image(im, name='val-img-step' + str(self.global_step),
                                             step=self.global_step)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):

        avg_loss         = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.logger.log_metrics({"val_loss": avg_loss}, step=self.global_step)
        
        return {'val_loss': avg_loss}

    def val_dataloader(self):
        valid_loader = DataLoader(self.valid_set,
                                  batch_size=self.conf.validation.dataloader.batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  drop_last=self.conf.validation.dataloader.drop_last)

        return valid_loader


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", default=False, help="Training or evaluation?")
    parser.add_argument("--config", type=str, required=True, help="Path to config.")

    # Training specific args
    parser.add_argument("--exp_dir", type=str, default='ckpts', help="Path to folder to save checkpoints.")
    parser.add_argument("--exp_name", type=str, default='test', help="name of experiment for comet log.")
    parser.add_argument("--resume", type=str, default=None, help="Path to model for loading.")
    parser.add_argument("--ckpt_freq", type=int, default=20, help="Frequency of saving the model (in epoch).")
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of available GPUs.")

    # Eval specific args
    parser.add_argument("--model_dir", type=str, default='final/cifar10.ckpt', help="Path to model for loading.")
    parser.add_argument("--sample_dir", type=str, default='samples', help="Path to save generated samples.")
    parser.add_argument("--prog_sample_freq", type=int, default=200, help="Progressive sample frequency.")
    parser.add_argument("--n_samples", type=int, default=20, help="Number of generated samples in evaluation.")

    args = parser.parse_args()

    # class Args():
    #     train = True
    #     config = "config/diffusion_cifar10.json"
    #     model_dir = None
    #     exp_dir = "exp/cifa10/"
    #     ckpt_freq = 20
    #     n_gpu = 1


    '''class Args():
        train = False
        config = "config/diffusion_celeba.json"
        model_dir = "exp/celeba/celeba.ckpt"
        sample_dir = 'samples/celeba-128'
        prog_sample_freq = 200
        n_samples = 20'''

    #args = Args()

    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)

    path_to_config = args.config
    with open(path_to_config, 'r') as f:
        conf = json.load(f)
    conf = obj(conf)
    denoising_diffusion_model = DDP(conf, args.exp_dir)

    if args.train:
        checkpoint_callback = ModelCheckpoint(dirpath=args.exp_dir,
                                              verbose=False,
                                              save_last=True,
                                              save_weights_only=False,
                                              every_n_epochs=args.ckpt_freq,
                                              save_on_train_epoch_end=True
                                              )

        comet_logger = CometLogger(
            api_key="nGRMV8S1NSghQEh2WmxFb3ZnA",
            save_dir="logs/",  # Optional
            project_name="EECE570",  # Optional
            experiment_name=args.exp_name,  # Optional
        )

        trainer = pl.Trainer(fast_dev_run=False,
                             gpus=args.n_gpu,
                             max_steps=conf.training.n_iter,
                             precision=conf.model.precision,
                             #gradient_clip_val=1.,
                             enable_progress_bar=True,
                             enable_checkpointing=True,
                             check_val_every_n_epoch=conf.training.eval_every_epoch,
                             callbacks=[checkpoint_callback],
                             logger=comet_logger
                             )

        trainer.fit(denoising_diffusion_model, ckpt_path=args.resume)

    else:
        
        denoising_diffusion_model.cuda()
        state_dict = torch.load(args.model_dir)
        denoising_diffusion_model.load_state_dict(state_dict['state_dict'])
        denoising_diffusion_model.eval()

        sample = progressive_samples_fn(denoising_diffusion_model.ema,
                                        denoising_diffusion_model.diffusion,
                                        (args.n_samples, 3, conf.dataset.resolution, conf.dataset.resolution),
                                        device='cuda',
                                        include_x0_pred_freq=args.prog_sample_freq)

        if not os.path.exists(args.sample_dir):
            os.mkdir(args.sample_dir)

        for i in range(args.n_samples):

            img = sample['samples'][i]
            plt.imsave(os.path.join(args.sample_dir, f'sample_{i}.png'), img.cpu().numpy().transpose(1, 2, 0))

            img = sample['progressive_samples'][i]
            img = make_grid(img, nrow=args.prog_sample_freq)
            plt.imsave(os.path.join(args.sample_dir, f'prog_sample_{i}.png'), img.cpu().numpy().transpose(1, 2, 0))

