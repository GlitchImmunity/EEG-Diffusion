from re import X
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from models.Encoder import *

# Note: Please add your models here for easy testing
class VAE(pl.LightningModule):
  def __init__(self, latent_dims = 10, lr=2e-4):
    super().__init__()
    self.latent_dims = latent_dims
    self.lr = lr
    self.accuracy = Accuracy('multiclass', num_classes=4)
    self.reduce_lr_on_plateau = None
    self.opt = None
    self.val_loss = None
    # Estimation of kld weight based on https://arxiv.org/pdf/1312.6114.pdf
    self.kld_weight = 1/256

    # Build Encoder:
    # In: 22 x 256
    self.conv1 = nn.Conv1d(22, 25, 11, 2, 5)
    self.swish1 = nn.Hardswish()
    self.bn1 = nn.BatchNorm1d(25)
    
    # In: 25 x 128
    self.conv2 = nn.Conv1d(25, 50, 11, 2, 5)
    self.swish2 = nn.Hardswish()
    self.bn2 = nn.BatchNorm1d(50)
    
    # In: 50 x 64
    self.conv3 = nn.Conv1d(50, 100, 11, 2, 5)
    self.swish3 = nn.Hardswish()
    self.bn3 = nn.BatchNorm1d(100)
    
    # In: 100 x 32
    self.conv4 = nn.Conv1d(100, 200, 11, 2, 5)
    self.swish4 = nn.Hardswish()
    self.bn4 = nn.BatchNorm1d(200)

    # In: 200 x 16
    self.conv5 = nn.Conv1d(200, 200, 11, 2, 5)
    self.swish5 = nn.Hardswish()
    self.bn5 = nn.BatchNorm1d(200)

    # In: 200 x 8
    self.conv6 = nn.Conv1d(200, 200, 11, 2, 5)
    self.swish6 = nn.Hardswish()
    self.bn6 = nn.BatchNorm1d(200)

    # Linear Layers
    self.mean = nn.Linear(200*4, self.latent_dims)
    self.var = nn.Linear(200*4, self.latent_dims)

    # Build Decoder:
    self.dec_linear = nn.Linear(self.latent_dims, 200*4)
    # In: 200 x 4
    self.dec_conv6 = nn.ConvTranspose1d(200, 200, 11, 2, 5, 1)
    self.dec_swish6 = nn.Hardswish()
    self.dec_bn6 = nn.BatchNorm1d(200)

    # In: 200 x 8
    self.dec_conv5 = nn.ConvTranspose1d(200, 200, 11, 2, 5, 1)
    self.dec_swish5 = nn.Hardswish()
    self.dec_bn5 = nn.BatchNorm1d(200)

    # In: 200 x 16
    self.dec_conv4 = nn.ConvTranspose1d(200, 100, 11, 2, 5, 1)
    self.dec_swish4 = nn.Hardswish()
    self.dec_bn4 = nn.BatchNorm1d(100)

    # In: 100 x 32
    self.dec_conv3 = nn.ConvTranspose1d(100, 50, 11, 2, 5, 1)
    self.dec_swish3 = nn.Hardswish()
    self.dec_bn3 = nn.BatchNorm1d(50)

    # In: 50 x 64
    self.dec_conv2 = nn.ConvTranspose1d(50, 25, 11, 2, 5, 1)
    self.dec_swish2 = nn.Hardswish()
    self.dec_bn2 = nn.BatchNorm1d(25)
    
    # In: 25 x 128
    self.dec_conv1 = nn.ConvTranspose1d(25, 22, 11, 2, 5, 1)
    self.dec_swish1 = nn.Hardswish()
    self.dec_bn1 = nn.BatchNorm1d(22)

    self.final_conv = nn.Conv1d(22, 22, 11, 1, 5)
    self.final_tanh = nn.Tanh()


  def forward(self, x, subject):

    out = self.conv1(x)
    out = self.swish1(out)
    out = self.bn1(out)

    out = self.conv2(out)
    out = self.swish2(out)
    out = self.bn2(out)
    
    out = self.conv3(out)
    out = self.swish3(out)
    out = self.bn3(out)
    
    out = self.conv4(out)
    out = self.swish4(out)
    out = self.bn4(out)

    out = self.conv5(out)
    out = self.swish5(out)
    out = self.bn5(out)

    out = self.conv6(out)
    out = self.swish6(out)
    out = self.bn6(out)
    
    out = torch.flatten(out, start_dim=1)

    latent_mean = self.mean(out)
    latent_var = self.var(out)
    z = torch.randn_like(latent_mean) * torch.exp(0.5 * latent_var) + latent_mean
    dec_out = self.dec_linear(z)
    dec_out = dec_out.view(-1, 200, 4)

    dec_out = self.dec_conv6(dec_out)
    dec_out = self.dec_swish6(dec_out)
    dec_out = self.dec_bn6(dec_out)

    dec_out = self.dec_conv5(dec_out)
    dec_out = self.dec_swish5(dec_out)
    dec_out = self.dec_bn5(dec_out)

    dec_out = self.dec_conv4(dec_out)
    dec_out = self.dec_swish4(dec_out)
    dec_out = self.dec_bn4(dec_out)

    dec_out = self.dec_conv3(dec_out)
    dec_out = self.dec_swish3(dec_out)
    dec_out = self.dec_bn3(dec_out)

    dec_out = self.dec_conv2(dec_out)
    dec_out = self.dec_swish2(dec_out)
    dec_out = self.dec_bn2(dec_out)

    dec_out = self.dec_conv1(dec_out)
    dec_out = self.dec_swish1(dec_out)
    dec_out = self.dec_bn1(dec_out)

    dec_out = self.final_conv(dec_out)
    dec_out = 100*self.final_tanh(dec_out)

    return dec_out, latent_mean, latent_var

  def configure_optimizers(self): 
      self.opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
      self.reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt,
            mode='min',
            factor=0.1,
            patience=100,
            verbose=True,
            min_lr=1e-8,
        )
      return [self.opt], {"optimizer": self.opt, "scheduler": self.reduce_lr_on_plateau, "monitor": "val_loss"}

  def training_step(self, train_batch, batch_idx): 
      x, y, subject = train_batch
      dec_out, latent_mean, latent_var = self.forward(x, subject) 
      reconstruct_loss = F.mse_loss(dec_out, x)
      kld = torch.mean(-.5*torch.sum(1+latent_var-latent_var.exp()-latent_mean**2,dim = 1), dim = 0)
      total_loss = reconstruct_loss + self.kld_weight * kld
      print(f"KLD:{kld}")
      print(f"REC:{reconstruct_loss}")
      print(f"TOT:{total_loss}")
      print(x.max())
      print(x.min())
      print(dec_out.max())
      print(dec_out.min())
      return total_loss 

  def validation_step(self, valid_batch, batch_idx): 
      x, y, subject = valid_batch
      dec_out, latent_mean, latent_var = self.forward(x, subject) 
      reconstruct_loss = F.mse_loss(dec_out, x)
      kld = torch.mean(-.5*torch.sum(1+latent_var-latent_var.exp()-latent_mean**2,dim = 1), dim = 0)
      total_loss = reconstruct_loss + self.kld_weight * kld
      self.log("val_loss", total_loss, prog_bar=True)
      
  def test_step(self, batch, batch_idx): 
      x, y, subject = batch
      dec_out, latent_mean, latent_var = self.forward(x, subject) 
      reconstruct_loss = F.mse_loss(dec_out, x)
      kld = torch.mean(-.5*torch.sum(1+latent_var-latent_var.exp()-latent_mean**2,dim = 1), dim = 0)
      total_loss = reconstruct_loss + self.kld_weight * kld
      self.log("test_loss", total_loss, prog_bar=True)
