import math
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from models.Encoder import *


class CNN_no_aug(pl.LightningModule):
  def __init__(self, lr=2e-4):
    super().__init__()
    self.lr = lr
    self.accuracy = Accuracy('multiclass', num_classes=4)
    self.reduce_lr_on_plateau = None
    self.opt = None
    self.val_loss = None

    # In: 22 x 1000
    self.conv1 = nn.Conv1d(22, 25, 10, 1, 'same')
    self.swish1 = nn.Hardswish()
    self.maxpool1 = nn.MaxPool1d(4, 4, 1)
    self.bn1 = nn.BatchNorm1d(25)
    self.dropout1 = nn.Dropout(0.4)
    
    # In: 25 x 250
    self.conv2 = nn.Conv1d(25, 50, 10, 1, 'same')
    self.swish2 = nn.Hardswish()
    self.maxpool2 = nn.MaxPool1d(3, 3, 1)
    self.bn2 = nn.BatchNorm1d(50)
    self.dropout2 = nn.Dropout(0.4)
    
    # In: 50 x 84
    self.conv3 = nn.Conv1d(50, 100, 11, 1, 5)
    self.swish3 = nn.Hardswish()
    self.maxpool3 = nn.MaxPool1d(3, 3, 1)
    self.bn3 = nn.BatchNorm1d(100)
    self.dropout3 = nn.Dropout(0.4)
    
    # In: 100 x 28
    self.conv4 = nn.Conv1d(100, 200, 11, 1, 5)
    self.swish4 = nn.Hardswish()
    self.maxpool4 = nn.MaxPool1d(3, 3, 1)
    self.bn4 = nn.BatchNorm1d(200)
    self.dropout4 = nn.Dropout(0.4)
    
    # In: 200 x 10
    self.conv5 = nn.Conv1d(200, 400, 11, 1, 5)
    self.swish5 = nn.Hardswish()
    self.maxpool5 = nn.MaxPool1d(3, 3, 1)
    self.bn5 = nn.BatchNorm1d(400)
    self.dropout5 = nn.Dropout(0.4)

    # Linear Layers
    self.l1 = nn.Linear(400*4, 4)

  def forward(self, x, onehot=None):
    out = self.conv1(x)
    out = self.swish1(out)
    out = self.maxpool1(out)
    out = self.bn1(out)
    out = self.dropout1(out)

    out = self.conv2(out)
    out = self.swish2(out)
    out = self.maxpool2(out)
    out = self.bn2(out)
    out = self.dropout2(out)
    
    out = self.conv3(out)
    out = self.swish3(out)
    out = self.maxpool3(out)
    out = self.bn3(out)
    out = self.dropout3(out)
    
    out = self.conv4(out)
    out = self.swish4(out)
    out = self.maxpool4(out)
    out = self.bn4(out)
    out = self.dropout4(out)
    
    out = self.conv5(out)
    out = self.swish5(out)
    out = self.maxpool5(out)
    out = self.bn5(out)
    out = self.dropout5(out)
    
    out = torch.flatten(out, start_dim=1)
    out = self.l1(out)

    return out

  def configure_optimizers(self): 
      self.opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
      self.reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True,
            min_lr=1e-8,
        )
      return [self.opt], {"optimizer": self.opt, "scheduler": self.reduce_lr_on_plateau, "monitor": "val_loss"}

  def training_step(self, train_batch, batch_idx): 
      x, y, subject = train_batch
      yhat = self.forward(x, subject) 
      loss = nn.functional.cross_entropy(yhat, y)
      return loss 

  def validation_step(self, valid_batch, batch_idx): 
      x, y, subject = valid_batch
      y_hat = self.forward(x, subject)
      loss = nn.functional.cross_entropy(y_hat, y)
      self.val_loss = loss
      self.log("val_loss", loss, prog_bar=True)
      self.log("val_accuracy", self.accuracy(torch.argmax(y,axis=1), torch.argmax(y_hat,axis=1)), prog_bar=True)
      
  def test_step(self, batch, batch_idx): 
      x, y, subject = batch
      y_hat = self.forward(x, subject)
      loss = nn.functional.cross_entropy(y_hat, y)
      self.val_loss = loss
      self.log("test_loss", loss, prog_bar=True)
      self.log("test_accuracy", self.accuracy(torch.argmax(y,axis=1), torch.argmax(y_hat,axis=1)), prog_bar=True)



class CNN(pl.LightningModule):
  def __init__(self, naive=True, end = False, lr=2e-4, time_window=256):
    super().__init__()
    self.naive = naive
    self.end = end
    self.lr = lr
    self.encoder = None
    self.accuracy = Accuracy('multiclass', num_classes=4)
    self.reduce_lr_on_plateau = None
    self.opt = None
    self.val_loss = None
    self.time_window=time_window

    offset = 0
    neurons = 0
    if not naive and end:
      neurons = 36
      self.encoder = OneHotEncoder(18, 36)
    elif not naive and not end:
      offset = 1
      self.encoder = OneHotEncoder(50, 256)

    # In: 22 x 256
    self.conv1 = nn.Conv1d(22+offset, 25, 10, 1, "same")
    self.swish1 = nn.ELU()
    self.maxpool1 = nn.MaxPool1d(3, 3, 1)
    self.bn1 = nn.BatchNorm1d(25)
    self.dropout1 = nn.Dropout(0.5)
    
    # In: 32 x 86
    self.conv2 = nn.Conv1d(25, 50, 10, 1, "same")
    self.swish2 = nn.ELU()
    self.maxpool2 = nn.MaxPool1d(3, 3, 1)
    self.bn2 = nn.BatchNorm1d(50)
    self.dropout2 = nn.Dropout(0.5)
    
    # In: 64 x 29
    self.conv3 = nn.Conv1d(50, 100, 11, 1, 5)
    self.swish3 = nn.ELU()
    self.maxpool3 = nn.MaxPool1d(3, 3, 1)
    self.bn3 = nn.BatchNorm1d(100)
    self.dropout3 = nn.Dropout(0.4)
    
    # In: 200 x 10
    self.conv4 = nn.Conv1d(100, 200, 11, 1, 5)
    self.swish4 = nn.ELU()
    self.maxpool4 = nn.MaxPool1d(3, 3, 1)
    self.bn4 = nn.BatchNorm1d(200)
    self.dropout4 = nn.Dropout(0.4)
        
    # Determine end temporal:
    temporal = math.ceil(time_window/3)
    temporal = math.ceil(temporal/3)
    temporal = math.ceil(temporal/3)
    temporal = math.ceil(temporal/3)

    # Linear Layers
    self.l1 = nn.Linear(200*int(temporal)+neurons, 4)

  def forward(self, x, onehot=None):
    x_mod = x
    subject_vector = None
    if not self.naive and not self.end:
      subject_vector = self.encoder(onehot)
      subject_vector = subject_vector[:, None, :]
      x_mod = torch.concat((x, subject_vector), axis=1)
    elif not self.naive and self.end:
      subject_vector = self.encoder(onehot)

    out = self.conv1(x_mod)
    out = self.swish1(out)
    out = self.maxpool1(out)
    out = self.bn1(out)
    out = self.dropout1(out)

    out = self.conv2(out)
    out = self.swish2(out)
    out = self.maxpool2(out)
    out = self.bn2(out)
    out = self.dropout2(out)
    
    out = self.conv3(out)
    out = self.swish3(out)
    out = self.maxpool3(out)
    out = self.bn3(out)
    out = self.dropout3(out)
    
    out = self.conv4(out)
    out = self.swish4(out)
    out = self.maxpool4(out)
    out = self.bn4(out)
    out = self.dropout4(out)
    
    out = torch.flatten(out, start_dim=1)
    if not self.naive and self.end:
      out = torch.concat((out, subject_vector), axis=1)
    out = self.l1(out)

    return out

  def configure_optimizers(self): 
      self.opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
      self.reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt,
            mode='min',
            factor=0.75,
            patience=8,
            verbose=True,
            min_lr=1e-8,
        )
      return [self.opt], {"optimizer": self.opt, "scheduler": self.reduce_lr_on_plateau, "monitor": "val_loss"}

  def training_step(self, train_batch, batch_idx): 
      x, y, subject = train_batch
      yhat = self.forward(x, subject) 
      loss = nn.functional.cross_entropy(yhat, y)
      return loss 

  def validation_step(self, valid_batch, batch_idx): 
      x, y, subject = valid_batch
      y_hat = self.forward(x, subject)
      loss = nn.functional.cross_entropy(y_hat, y)
      self.val_loss = loss
      self.log("val_loss", loss, prog_bar=True)
      self.log("val_accuracy", self.accuracy(torch.argmax(y,axis=1), torch.argmax(y_hat,axis=1)), prog_bar=True)
      
  def test_step(self, batch, batch_idx): 
      x, y, subject = batch
      y_hat = self.forward(x, subject)
      loss = nn.functional.cross_entropy(y_hat, y)
      self.val_loss = loss
      self.log("test_loss", loss, prog_bar=True)
      self.log("test_accuracy", self.accuracy(torch.argmax(y,axis=1), torch.argmax(y_hat,axis=1)), prog_bar=True)



class CNN_FFT(pl.LightningModule):
  def __init__(self, naive=True, end = False, lr=2e-4, time_window=256):
    super().__init__()
    self.naive = naive
    self.end = end
    self.lr = lr
    self.encoder = None
    self.accuracy = Accuracy('multiclass', num_classes=4)
    self.reduce_lr_on_plateau = None
    self.opt = None
    self.val_loss = None
    self.time_window=time_window

    offset = 0
    neurons = 0
    if not naive and end:
      neurons = 36
      self.encoder = OneHotEncoder(18, 36)
    elif not naive and not end:
      offset = 1
      self.encoder = OneHotEncoder(50, 256)

    # In: 22 x 256
    self.conv1 = nn.Conv1d(22+offset, 32, 10, 1, "same")
    self.swish1 = nn.ELU()
    self.maxpool1 = nn.MaxPool1d(3, 3, 1)
    self.bn1 = nn.BatchNorm1d(32)
    self.dropout1 = nn.Dropout(0.5)
    
    # In: 32 x 86
    self.conv2 = nn.Conv1d(32, 64, 10, 1, "same")
    self.swish2 = nn.ELU()
    self.maxpool2 = nn.MaxPool1d(3, 3, 1)
    self.bn2 = nn.BatchNorm1d(64)
    self.dropout2 = nn.Dropout(0.5)
    
    # In: 64 x 29
    self.conv3 = nn.Conv1d(64, 100, 11, 1, 5)
    self.swish3 = nn.ELU()
    self.maxpool3 = nn.MaxPool1d(3, 3, 1)
    self.bn3 = nn.BatchNorm1d(100)
    self.dropout3 = nn.Dropout(0.4)
    
    # In: 200 x 10
    self.conv4 = nn.Conv1d(100, 200, 11, 1, 5)
    self.swish4 = nn.ELU()
    self.maxpool4 = nn.MaxPool1d(3, 3, 1)
    self.bn4 = nn.BatchNorm1d(200)
    self.dropout4 = nn.Dropout(0.4)
        
    # Determine end temporal:
    temporal = math.ceil(time_window/3)
    temporal = math.ceil(temporal/3)
    temporal = math.ceil(temporal/3)
    temporal = math.ceil(temporal/3)

    # Linear Layers
    self.l1 = nn.Linear(200*int(temporal)+neurons, 4)

  def forward(self, x, onehot=None):
    x_mod = x
    subject_vector = None
    if not self.naive and not self.end:
      subject_vector = self.encoder(onehot)
      subject_vector = subject_vector[:, None, :]
      x_mod = torch.concat((x, subject_vector), axis=1)
    elif not self.naive and self.end:
      subject_vector = self.encoder(onehot)

    # x_mod = torch.fft.rfft(x, axis=2)
    # x_mod[:, :, 80:]=0
    # x_mod = torch.fft.irfft(x_mod, axis=2)

    out = self.conv1(x_mod)
    out = self.swish1(out)
    out = self.maxpool1(out)
    out = self.bn1(out)
    out = self.dropout1(out)

    out = self.conv2(out)
    out = self.swish2(out)
    out = self.maxpool2(out)
    out = self.bn2(out)
    out = self.dropout2(out)
    
    out = self.conv3(out)
    out = self.swish3(out)
    out = self.maxpool3(out)
    out = self.bn3(out)
    out = self.dropout3(out)
    
    out = self.conv4(out)
    out = self.swish4(out)
    out = self.maxpool4(out)
    out = self.bn4(out)
    out = self.dropout4(out)
    
    out = torch.flatten(out, start_dim=1)
    if not self.naive and self.end:
      out = torch.concat((out, subject_vector), axis=1)
    out = self.l1(out)

    return out

  def configure_optimizers(self): 
      self.opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
      return [self.opt]

  def training_step(self, train_batch, batch_idx): 
      x, y, subject = train_batch
      yhat = self.forward(x, subject) 
      loss = nn.functional.cross_entropy(yhat, y)
      return loss 

  def validation_step(self, valid_batch, batch_idx): 
      x, y, subject = valid_batch
      y_hat = self.forward(x, subject)
      loss = nn.functional.cross_entropy(y_hat, y)
      self.val_loss = loss
      self.log("val_loss", loss, prog_bar=True)
      self.log("val_accuracy", self.accuracy(torch.argmax(y,axis=1), torch.argmax(y_hat,axis=1)), prog_bar=True)
      
  def test_step(self, batch, batch_idx): 
      x, y, subject = batch
      y_hat = self.forward(x, subject)
      loss = nn.functional.cross_entropy(y_hat, y)
      self.val_loss = loss
      self.log("test_loss", loss, prog_bar=True)
      self.log("test_accuracy", self.accuracy(torch.argmax(y,axis=1), torch.argmax(y_hat,axis=1)), prog_bar=True)




class CNN_LSTM(pl.LightningModule):
  def __init__(self, naive=True, end = False, lr=2e-4, time_window=256):
    super().__init__()
    self.naive = naive
    self.end = end
    self.lr = lr
    self.encoder = None
    self.accuracy = Accuracy('multiclass', num_classes=4)
    self.reduce_lr_on_plateau = None
    self.opt = None
    self.val_loss = None
    self.time_window=time_window

    offset = 0
    neurons = 0
    if not naive and end:
      neurons = 36
      self.encoder = OneHotEncoder(18, 36)
    elif not naive and not end:
      offset = 1
      self.encoder = OneHotEncoder(50, 256)

    # In: 22 x 256
    self.conv1 = nn.Conv1d(22+offset, 32, 11, 1, 5)
    self.swish1 = nn.SiLU()
    self.maxpool1 = nn.MaxPool1d(3, 3, 1)
    self.bn1 = nn.BatchNorm1d(32)
    self.dropout1 = nn.Dropout(0.25)
    
    # In: 25 x 86
    self.conv2 = nn.Conv1d(32, 50, 11, 1, 5)
    self.swish2 = nn.SiLU()
    self.maxpool2 = nn.MaxPool1d(3, 3, 1)
    self.bn2 = nn.BatchNorm1d(50)
    self.dropout2 = nn.Dropout(0.25)
    
    # In: 50 x 29
    self.conv3 = nn.Conv1d(50, 100, 11, 1, 5)
    self.swish3 = nn.SiLU()
    self.maxpool3 = nn.MaxPool1d(3, 3, 1)
    self.bn3 = nn.BatchNorm1d(100)
    self.dropout3 = nn.Dropout(0.25)
    
    # Determine end temporal:
    temporal = math.ceil(time_window/3)
    temporal = math.ceil(temporal/3)
    temporal = math.ceil(temporal/3)
    
    # In: 200 x 10
    self.lstm1 = nn.LSTM(100, 128, 1, bidirectional=True, batch_first=True)
    self.lstm2 = nn.LSTM(256, 64, 1, bidirectional=True, batch_first=True)

    # Linear Layers
    self.l1 = nn.Linear(64*int(temporal)*2+neurons, 4)

  def forward(self, x, onehot=None):
    x_mod = x
    subject_vector = None
    if not self.naive and not self.end:
      subject_vector = self.encoder(onehot)
      subject_vector = subject_vector[:, None, :]
      x_mod = torch.concat((x, subject_vector), axis=1)
    elif not self.naive and self.end:
      subject_vector = self.encoder(onehot)

    out = self.conv1(x_mod)
    out = self.swish1(out)
    out = self.maxpool1(out)
    out = self.bn1(out)
    out = self.dropout1(out)

    out = self.conv2(out)
    out = self.swish2(out)
    out = self.maxpool2(out)
    out = self.bn2(out)
    out = self.dropout2(out)
    
    out = self.conv3(out)
    out = self.swish3(out)
    out = self.maxpool3(out)
    out = self.bn3(out)
    out = self.dropout3(out)
    
    out = torch.transpose(out, 1, 2)
    out, (h,c) = self.lstm1(out)
    out, (h,c) = self.lstm2(out)
    
    out = torch.flatten(out, start_dim=1)
    if not self.naive and self.end:
      out = torch.concat((out, subject_vector), axis=1)
    out = self.l1(out)

    return out

  def configure_optimizers(self): 
      self.opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
      self.reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True,
            min_lr=1e-8,
        )
      return [self.opt], {"optimizer": self.opt, "scheduler": self.reduce_lr_on_plateau, "monitor": "val_loss"}

  def training_step(self, train_batch, batch_idx): 
      x, y, subject = train_batch
      yhat = self.forward(x, subject) 
      loss = nn.functional.cross_entropy(yhat, y)
      return loss 

  def validation_step(self, valid_batch, batch_idx): 
      x, y, subject = valid_batch
      y_hat = self.forward(x, subject)
      loss = nn.functional.cross_entropy(y_hat, y)
      self.val_loss = loss
      self.log("val_loss", loss, prog_bar=True)
      self.log("val_accuracy", self.accuracy(torch.argmax(y,axis=1), torch.argmax(y_hat,axis=1)), prog_bar=True)
      
  def test_step(self, batch, batch_idx): 
      x, y, subject = batch
      y_hat = self.forward(x, subject)
      loss = nn.functional.cross_entropy(y_hat, y)
      self.val_loss = loss
      self.log("test_loss", loss, prog_bar=True)
      self.log("test_accuracy", self.accuracy(torch.argmax(y,axis=1), torch.argmax(y_hat,axis=1)), prog_bar=True)
      


class CNN_MHA(pl.LightningModule):
  def __init__(self, naive=True, end = False, lr=2e-4, time_window=256):
    super().__init__()
    self.naive = naive
    self.end = end
    self.lr = lr
    self.encoder = None
    self.accuracy = Accuracy('multiclass', num_classes=4)
    self.reduce_lr_on_plateau = None
    self.opt = None
    self.val_loss = None
    self.time_window=time_window

    # In: 22 x 256
    self.conv1 = nn.Conv1d(22, 25, 11, 1, 5)
    self.swish1 = nn.SiLU()
    self.maxpool1 = nn.MaxPool1d(3, 3, 1)
    self.bn1 = nn.BatchNorm1d(25)
    self.dropout1 = nn.Dropout(0.25)
    
    # In: 25 x 86
    self.conv2 = nn.Conv1d(25, 50, 11, 1, 5)
    self.swish2 = nn.SiLU()
    self.maxpool2 = nn.MaxPool1d(3, 3, 1)
    self.bn2 = nn.BatchNorm1d(50)
    self.dropout2 = nn.Dropout(0.25)
    
    # In: 50 x 29
    self.conv3 = nn.Conv1d(50, 100, 11, 1, 5)
    self.swish3 = nn.SiLU()
    self.maxpool3 = nn.MaxPool1d(3, 3, 1)
    self.bn3 = nn.BatchNorm1d(100)
    self.dropout3 = nn.Dropout(0.25)

    # In: 100 x 10
    self.conv4 = nn.Conv1d(100, 200, 11, 1, 5)
    self.swish4 = nn.SiLU()
    self.maxpool4 = nn.MaxPool1d(3, 3, 1)
    self.bn4 = nn.BatchNorm1d(200)
    self.dropout4 = nn.Dropout(0.25)
    
    # Determine end temporal:
    temporal = math.ceil(time_window/3)
    temporal = math.ceil(temporal/3)
    temporal = math.ceil(temporal/3)
    temporal = math.ceil(temporal/3)
    
    self.mha = nn.MultiheadAttention(temporal, 4, batch_first=True)

    # Linear Layers
    self.l1 = nn.Linear(200*int(temporal), 4)

  def forward(self, x, onehot=None):
    x_mod = x

    out = self.conv1(x_mod)
    out = self.swish1(out)
    out = self.maxpool1(out)
    out = self.bn1(out)
    out = self.dropout1(out)

    out = self.conv2(out)
    out = self.swish2(out)
    out = self.maxpool2(out)
    out = self.bn2(out)
    out = self.dropout2(out)
    
    out = self.conv3(out)
    out = self.swish3(out)
    out = self.maxpool3(out)
    out = self.bn3(out)
    out = self.dropout3(out)

    out = self.conv4(out)
    out = self.swish4(out)
    out = self.maxpool4(out)
    out = self.bn4(out)
    out = self.dropout4(out)

    out, _ = self.mha(out, out, out)
    
    out = torch.flatten(out, start_dim=1)
    out = self.l1(out)

    return out

  def configure_optimizers(self): 
      self.opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
      self.reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True,
            min_lr=1e-8,
        )
      return [self.opt], {"optimizer": self.opt, "scheduler": self.reduce_lr_on_plateau, "monitor": "val_loss"}

  def training_step(self, train_batch, batch_idx): 
      x, y, subject = train_batch
      yhat = self.forward(x, subject) 
      loss = nn.functional.cross_entropy(yhat, y)
      return loss 

  def validation_step(self, valid_batch, batch_idx): 
      x, y, subject = valid_batch
      y_hat = self.forward(x, subject)
      loss = nn.functional.cross_entropy(y_hat, y)
      self.val_loss = loss
      self.log("val_loss", loss, prog_bar=True)
      self.log("val_accuracy", self.accuracy(torch.argmax(y,axis=1), torch.argmax(y_hat,axis=1)), prog_bar=True)
      
  def test_step(self, batch, batch_idx): 
      x, y, subject = batch
      y_hat = self.forward(x, subject)
      loss = nn.functional.cross_entropy(y_hat, y)
      self.val_loss = loss
      self.log("test_loss", loss, prog_bar=True)
      self.log("test_accuracy", self.accuracy(torch.argmax(y,axis=1), torch.argmax(y_hat,axis=1)), prog_bar=True)
      

