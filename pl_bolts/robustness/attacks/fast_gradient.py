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
        self.reg = 1e-12
        
    def fgsm(self, input_img, label, iters, targeted=False, alpha=0.02, clamp=((-2.118, -2.036, -1.804), (2.249, 2.429, 2.64)), use_cuda=False):
        device = torch.device('cuda' if use_cuda else 'cpu')
        self.classifier.to(device)
        self.classifier.eval()
        
        crit = nn.CrossEntropyLoss().to(device)
        input_img = input_img.to(device)
        img_var = input_img.clone().requires_grad(True).to(device)
        label_var = torch.LongTensor([label]).to(device)
        
        for _ in iters:
            img_var.grad = None
            pred = self.classifier(img_var)
            
            loss = crit(pred, label_var) + self.reg * F.mse_loss(img_var, input_img)
            loss.backward()
            
            noise = alpha * torch.sign(img_var.grad.data)
            
            if targeted:
                img_var.data = img_var.data - noise
            else:
                img_var.data = img_var.data + noise
                
            if clamp[0] is not None and clamp[1] is not None:
                assert len(clamp[0]) == len(clamp[1])
                for ch in range(len(clamp[0])):
                    img_var.data[:, ch, :, :].clamp_(clamp[0][ch], clamp[1][ch])
                    
            return img_var.cpu().detach()
        
    def generate_adversarial(self, iters=1, targeted=False):
        x_test = self.classifier.test_dataloader()
        x_adv = []
        for i in range(len(x_test)):
            cur_img = x_test[i]
            cur_label = y_test[i]
            res = self.fgsm(cur_img, 
                            cur_label, 
                            iters=iters, 
                            argeted=targeted, 
                            alpha=self.classifier.lr, 
                            use_cuda=False)
            x_adv.append(res)
            
        return x_adv
        
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

