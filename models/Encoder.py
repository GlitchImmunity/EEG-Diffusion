import torch
import torch.nn as nn
from torchmetrics import Accuracy
import pytorch_lightning as pl

class OneHotEncoder(pl.LightningModule):
  def __init__(self, hidden1=32, hidden2= 64):
    super().__init__()
    self.l1 = nn.Linear(9, hidden1)
    self.bn1 = nn.BatchNorm1d(hidden1)
    self.relu1 = nn.ReLU()
    
    self.l2 = nn.Linear(hidden1, hidden2)
    self.bn2 = nn.BatchNorm1d(hidden2)
    self.relu2 = nn.ReLU()

  def forward(self, x):
    out = self.l1(x)
    out = self.bn1(out)
    out = self.relu1(out)
    out = self.l2(out)
    out = self.bn2(out)
    out = self.relu2(out)
    return out

class PositionalEncoder(pl.LightningModule):
  pass