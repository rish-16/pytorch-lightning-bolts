import os
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

class FastGradientMethod:
    def __init__(self, classifier, epsilon=0.1):
        self.classifier = classifier
        self.epsilon = epsilon
        
    def generate_adversarial(self, x_test):
        pass
        
class LitNet(LightningModule):
    def __init__(self):
        super().__init__()
        
        self.l1 = nn.Linear(28 * 28, 128)
        self.l2 = nn.Linear(128, 256)
        self.l3 = nn.Linear(256, 10)
        
    def forward(self, x):
        batch_size, channels, width, height = x.size()
        
        x = x.view(batch_size, -1)
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        
        x = F.log_softmax(x, dim=1)
        return x
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss
        
    def train_dataloader(self):
        T = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (.3801, ))])
        
        dataset = MNIST(os.getcwd() + "/bin/", train=True, download=True, transform=T)
        mnist_train, _ = random_split(dataset, [55000, 5000])
        return DataLoader(mnist_train, batch_size=64)
        
    def val_dataloader(self):
        T = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (.3801, ))])
        
        dataset = MNIST(os.getcwd() + "/bin/", train=True, download=True, transform=T)
        _, mnist_val = random_split(dataset, [55000, 5000])
        return DataLoader(mnist_val, batch_size=64)
        
    def test_dataloader(self):
        T = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (.3801, ))])
        
        mnist_test = MNIST(os.getcwd() + "/bin/", train=False, download=True, transform=T)
        return DataLoader(mnist_test, batch_size=64)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
        
net = LitNet()
trainer = Trainer()
trainer.fit(net)