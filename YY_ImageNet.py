#@save
import collections
import hashlib
import inspect
import math
import os
import random
import re
import shutil
import sys
import tarfile
import time
import zipfile
from collections import defaultdict
import pandas as pd
import requests
from IPython import display
from torch.cuda.amp import autocast, GradScaler
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
from torch.nn import functional as F
d2l = sys.modules[__name__]


import torch
from torch import nn
from torchsummary import summary

import numpy as np

import time
import torchvision
from torchvision import transforms

def add_to_class(Class): #@save
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper
    
    
def cpu(): #@save
    return torch.device('cpu')
def gpu(i=0): #@save
    return torch.device(f'cuda:{i}')
cpu(), gpu(), gpu(1)

def num_gpus(): #@save
    return torch.cuda.device_count()
num_gpus()

def try_gpu(i=0): #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()
    
class HyperParameters: #@save
    def save_hyperparameters(self, ignore=[]):
    """Save function arguments into class attributes."""
    
    #f_back: frame caller
    #frame: table of local variablies to the frame's function
    frame = inspect.currentframe().f_back
    _, _, _, local_vars = inspect.getargvalues(frame)
    #Pega as variáveis locais da função que chamou esta.
    
    self.hparams = {k:v for k, v in local_vars.items()
        if k not in set(ignore+['self']) and not k.startswith('_')}
    for k, v in self.hparams.items():
        setattr(self, k, v)


class Module(nn.Module, d2l.HyperParameters): #@save
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        
    def loss(self, y_hat, y):
        raise NotImplementedError
        
    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)
    
    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        return l
    
    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        return l
        
    def configure_optimizers(self, weight_decay=1e-4):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, amsgrad=True, weight_decay=weight_decay)
        #return torch.optim.RMSprop(self.parameters(), lr=self.lr, weight_decay=weight_decay)
        
    def apply_init(self, inputs, init=None):
        self.forward(*inputs)
        if init is not None:
            self.net.apply(init)
            

class DataModule(d2l.HyperParameters): #@save
    def __init__(self, root='../data', num_workers=0):
        self.save_hyperparameters()
        
    def get_dataloader(self, train):
        raise NotImplementedError
        
    def train_dataloader(self):
        return self.get_dataloader(train=True)
    
    def val_dataloader(self):
        return self.get_dataloader(train=False)
        

class Trainer(d2l.HyperParameters): #@save
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'
        self.epoch = None
        
    def prepare_batch(self, batch):
        return batch
        
    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)
    def prepare_model(self, model):
        model.trainer = self
        self.model = model
    
    def fit(self, model, data, lr_update=0.8, lr_reduct_rule=-1, lr_limit = 0.01, lr_uu = 2, weight_decay=0.001, warmup=0):
        self.prepare_data(data)
        self.prepare_model(model)
        #self.optim = model.configure_optimizers(weight_decay)
        #self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        
        self.warmup = warmup
        
        self.ema_start = self.max_epochs//4
        self.ema_model = None
        
        






        if self.sched == None:
            self.sched = torch.optim.lr_scheduler.OneCycleLR(self.optim, self.model.lr, epochs=self.max_epochs-self.warmup, 
                                                                 steps_per_epoch=len(self.train_dataloader))
        
        


        
        last_loss = 1
        last_losses = []
        
        lr_copy = self.model.lr
        if self.epoch==None:
            self.epoch=0
        
        while(self.epoch<self.max_epochs):
            val_loss, acc, loss = self.fit_epoch()
            #return
            actual_loss = loss
            
            last_losses.append(actual_loss)
            
            
            for g in self.optim.param_groups:
                g['weight_decay'] = g['lr']*1.56
            
            """Save Best Checkpoint"""
            if acc.item() > self.best_acc:
                self.best_acc = acc.detach().item()
                torch.save({
                        'epoch': self.epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optim.state_dict(),
                        'scheduler_state_dict': self.sched.state_dict(),
                        'loss': loss,
                        'best_acc': self.best_acc
                        }, './imagenet_checkpoints/best'+str(self.best_acc))
            

            torch.save({
                    'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optim.state_dict(),
                    'scheduler_state_dict': self.sched.state_dict(),
                    'loss': loss,
                    'best_acc': self.best_acc
                    }, './imagenet_checkpoints/checkpoint'+str(self.epoch))
            
            

            """Print Current Statistics"""
            print('Epoch:', self.epoch)
            print('Loss:', loss)
            print('Val loss:', val_loss)
            print('Accuracy: %{}'.format(acc.item()))
            print('Learning rate:', self.optim.param_groups[0]['lr'])
            print('')

            last_loss = actual_loss
            self.epoch+=1
            
        print("Val loss, acc:", val_loss, acc)
        print('Lr:', lr_copy)
        
        
        
    def fit_epoch(self):
        #Barckward
        self.model.train()
        all_losses = 0
        
        count = 0
        for batch in self.train_dataloader:
            with autocast():
                loss = self.model.training_step(self.prepare_batch(batch))
            all_losses+=torch.clone(loss).detach()
            
            for param in model.parameters():
                param.grad = None
                
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:
                    #self.clip_gradients(self.gradient_clip_val, self.model)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                self.optim.step()
            if self.sched: self.sched.step()
            
            if self.train_batch_idx == self.ema_start and self.ema_model==None:
                ema_avg = (lambda averaged_model_parameter, model_parameter, num_averaged:0.1 * averaged_model_parameter + 0.9 * model_parameter)
                self.ema_model = torch.optim.swa_utils.AveragedModel(self.model, avg_fn=ema_avg)
            if self.train_batch_idx>self.ema_start:
                self.ema_model.update_parameters(self.model)
                
            self.train_batch_idx += 1
            count+=1
            
        
            #return 0,0,0 
        
        
        if self.val_dataloader is None:
            return
        self.model.eval()
        
        all_losses/=count
        
        acc = 0
        val_loss = 0
        count = 0
        for batch in self.val_dataloader:
            with torch.no_grad():
                final_val_error = self.model.validation_step(self.prepare_batch(batch))
                val_loss += final_val_error[0]
                acc += final_val_error[1]
            count += 1
            self.val_batch_idx += 1
        val_loss/=count
        acc/=count
        
        
        return val_loss, acc, all_losses


@d2l.add_to_class(d2l.Trainer) #@save
def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
    self.save_hyperparameters()
    self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]

@d2l.add_to_class(d2l.Trainer) #@save
def prepare_batch(self, batch):
    if self.gpus:
        batch = [a.to(self.gpus[0]) for a in batch]
    return batch

@d2l.add_to_class(d2l.Trainer) #@save
def prepare_model(self, model):
    model.trainer = self
    if self.gpus:
        model.to(self.gpus[0])
    self.model = model

#@d2l.add_to_class(d2l.Trainer) #@save
#def clip_gradients(self, grad_clip_val, model):
#    params = [p for p in model.parameters() if p.requires_grad]
#    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) if p!=None else 0 for p in params))
#    if norm > grad_clip_val:
#        for param in params:
#            param.grad[:] *= grad_clip_val / norm


class Classifier(d2l.Module): #@save
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        l = self.loss(Y_hat, batch[-1])
        acc = self.accuracy(Y_hat, batch[-1])
        return l, acc
        
    def accuracy(self, Y_hat, Y, averaged=True):
        """Compute the number of correct predictions."""
        Y_hat = Y_hat.view((-1, Y_hat.shape[-1]))
        preds = Y_hat.argmax(axis=1).type(Y.dtype)
        compare = (preds == Y.view(-1)).type(torch.float32)
        return compare.mean() if averaged else compare
    
    def layer_summary(self, X_shape):
        X = torch.randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)
            
    def loss(self, Y_hat, Y, averaged=True):
        Y_hat = Y_hat.view((-1, Y_hat.shape[-1]))
        Y = Y.view((-1,))
        return F.cross_entropy(
                Y_hat, Y, reduction='mean' if averaged else 'none')


# Data transforms (normalization & data augmentation)
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
train_tfms = transforms.Compose([
                         transforms.Resize((224, 224)),
                         transforms.ColorJitter(brightness=0.01),
                         transforms.RandomCrop(224, padding=4, padding_mode='reflect'), 
                         transforms.RandomHorizontalFlip(), 
                         transforms.ToTensor(), 
                         transforms.Normalize(*stats,inplace=True)])
val_tfms = transforms.Compose([
                         transforms.Resize((224, 224)),
                         transforms.ToTensor(),
                         transforms.Normalize(*stats)])
                         

class DT_ImageNet(d2l.DataModule): #@save
    def __init__(self, train_tfms, val_tfms, batch_size=64, resize=(224, 224)):
        super().__init__()
        self.save_hyperparameters()
        
        self.train = torchvision.datasets.ImageFolder(
                                    root='/home/augusto/ImageNet/ImageNet/ILSVRC/Data/CLS-LOC/train', transform=train_tfms)
        self.val = torchvision.datasets.ImageFolder(
                                    root='/home/augusto/ImageNet/ImageNet/new_val', transform=val_tfms)
        
    def get_dataloader(self, train):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
                num_workers=4, pin_memory=True)
                                
def init_cnn(module): #@save
    """Initialize weights for CNNs."""
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)

class Residual(nn.Module): #@save
    """The Residual block of ResNet."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1,
                    stride=strides)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1,
                        stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        
    def forward(self, X):
        Y = F.gelu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.gelu(X + Y)
        
class SE(nn.Module):
    def __init__(self, in_dim, hidden_dim, reduction = 4):
        super().__init__()
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                  nn.Conv2d(in_dim, in_dim//reduction,1),
                                  nn.GELU(),
                                  nn.Conv2d(in_dim//reduction, in_dim,1),
                                  nn.Hardsigmoid())
                      
    def forward(self, x):
        return x * self.gate(x.mean((2,3), keepdim=True))

class MBConv(nn.Module):
    def __init__(self,in_dim,out_dim,kernel_size=3,stride_size=1,expand_rate = 4,se_rate = 0.25, dropout = 0.):
        super().__init__()
        hidden_dim = int(expand_rate * out_dim)
        self.bn = nn.BatchNorm2d(in_dim)
        self.expand_conv = nn.Sequential(nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
                                         nn.BatchNorm2d(hidden_dim),
                                         nn.GELU())
        self.dw_conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride_size, kernel_size//2, groups=hidden_dim, bias=False),
                                     nn.BatchNorm2d(hidden_dim),
                                     nn.GELU())
        
        self.dropout = nn.Dropout(p=dropout)
        
        self.se = SE(hidden_dim,max(1,int(out_dim*se_rate)))
        self.out_conv = nn.Sequential(nn.Conv2d(hidden_dim, out_dim, 1, bias=False),
                                      nn.BatchNorm2d(out_dim))
        if stride_size > 1:
            self.proj = nn.Sequential(nn.MaxPool2d(kernel_size, stride_size, kernel_size//2),
                                      nn.Conv2d(in_dim, out_dim, 1, bias=False)) 
        elif in_dim!=out_dim:
            self.proj = nn.Conv2d(in_dim, out_dim, 1)
        else: 
            self.proj = nn.Identity()

    def forward(self, x):
        out = self.bn(x)
        out = self.expand_conv(x)
        out = self.dw_conv(out)
        out = self.dropout(out)
        out = self.se(out)
        out = self.out_conv(out)
        return out + self.proj(x)


class YingYangBlock(nn.Module):
    def __init__(self, depth=1, num_channels=32, cnn_dropout=0., inc=2, repeat_depth=1):
        super().__init__()
        
        blk=[]
        aux_depth=depth
        aux_num_channels = num_channels
        for i in range(repeat_depth):
            blk.append(self.ying_stage(aux_depth, aux_num_channels, cnn_dropout, inc))
            aux_num_channels+=inc*aux_depth
            #aux_depth*=2
        self.ying = nn.Sequential(*blk)
        
        blk=[]
        aux_depth=depth
        aux_num_channels = num_channels
        for i in range(repeat_depth):
            blk.append(self.yang_stage(aux_depth, aux_num_channels, cnn_dropout, inc))
            aux_num_channels+=inc*aux_depth
            #aux_depth*=2
        
        self.yang = nn.Sequential(*blk)
        
        self.ying.apply(d2l.init_cnn)
        self.yang.apply(d2l.init_cnn)
        
    def ying_stage(self, depth, num_channels, cnn_dropout, inc):
        blk = []
        aux_num_channels = num_channels
        blk.append(Residual(num_channels, use_1x1conv=True))
        
        aux_num_channels+=inc
        
        #blk.append(self.reduct_channel(num_channels))
        
        for i in range(depth-1):
            blk.append(MBConv(aux_num_channels-inc, aux_num_channels, dropout=cnn_dropout))
            aux_num_channels+=inc
            
        blk.append(MBConv(aux_num_channels-inc, aux_num_channels, dropout=cnn_dropout, stride_size = 2))
        
        #blk.append(self.reduction_stage(num_channels, stride = 2))
        
        return nn.Sequential(*blk)

    def yang_stage(self, depth, num_channels, cnn_dropout, inc):
        blk = []
        aux_num_channels = num_channels
        blk.append(Residual(num_channels, use_1x1conv=True))
        
        aux_num_channels+=inc
        #blk.append(self.reduction_stage(num_channels))
        
        
        blk.append(MBConv(aux_num_channels-inc, aux_num_channels,  dropout=cnn_dropout, stride_size = 2))
        
        aux_num_channels+=inc
        
        for i in range(depth-1):
            blk.append(MBConv(aux_num_channels-inc, aux_num_channels, dropout=cnn_dropout))
            aux_num_channels+=inc
                
        
        return nn.Sequential(*blk)
    
    def forward(self, X):
        Y = self.ying(torch.unsqueeze(X[:, 0, :, :], 1))
            
        Y1 = self.yang(X)
        
        #Y = torch.cat((Y, Y1), dim=1)
        #Y = Y1*(1-Y)
        
        return Y1*(1-Y)+Y1-Y
        
class YingYangNet(d2l.Classifier):
    def __init__(self, yingyang_arch, reduction_arch,  stride = 2, lr = 0.1,
                 num_classes = 10, dropout = 0):
        super(YingYangNet, self).__init__()
        self.save_hyperparameters()
        
        self.yingyang = YingYangBlock(*yingyang_arch)
        
        yy_channels, yy_depth, cnn_dropout, inc, _ = yingyang_arch
        
        reduction_depth, reduction_channels, reduction_repeat_depth = self.reduction_arch
        
        
        aux_num_channels = yy_channels+inc*yy_depth
        aux_num_channels = reduction_channels
        reduct_blk = []
        for i in range(reduction_repeat_depth):
            reduct_blk.append(self.reduction_stage(
                                reduction_depth, aux_num_channels, cnn_dropout, stride=2, inc=inc))
            aux_num_channels+=inc
        self.reduction = nn.Sequential(*reduct_blk)
            
        self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                nn.LazyLinear(num_classes//2),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(num_classes//2, num_classes))
        
        
        #_, num_hiddens, _  = yingyang_arch
        #self.head = nn.Sequential(
        #        nn.Conv2d(num_hiddens, num_hiddens, kernel_size=1, bias=False),
        #        nn.BatchNorm2d(num_hiddens, eps=0.001, momentum=0.01),
        #        nn.Hardswish(),
        #        nn.AdaptiveAvgPool2d(output_size=1),
        #        nn.Flatten(),
        #        nn.LazyLinear(num_hiddens),
        #        nn.Hardswish(),
        #        nn.Dropout(p=dropout, inplace=True),
        #        nn.LazyLinear(num_classes))
        
        
        self.yingyang.apply(d2l.init_cnn)
        self.reduction.apply(d2l.init_cnn)
        self.head.apply(d2l.init_cnn)
        
    def reduction_stage(self, depth, num_channels, cnn_dropout=0., stride=2, inc=2):
        blk = []
        
        blk.append(Residual(num_channels, use_1x1conv=True, strides=stride))
        
        aux_num_channels = num_channels+inc
        
        blk.append(MBConv(num_channels, aux_num_channels, dropout=cnn_dropout))
        
        aux_num_channels+=inc
        
        for i in range(depth-1):
            blk.append(MBConv(aux_num_channels-inc, aux_num_channels, dropout=cnn_dropout))
            aux_num_channels+=inc
            
        
        #kernel_size = 3
        
        #blk.append(ReductionBottleNeck(num_channels, in_ratio=2, expand_ratio=5, stride=stride))
        
        
        #for i in range(depth-1):
        #    blk.append(ReductionBottleNeck(num_channels, in_ratio=1, expand_ratio=4, stride=1))
            
        return nn.Sequential(*blk)
    
    def forward(self, X):
        
        
        Y = self.yingyang(X)
        
        Y = self.reduction(Y)
        

        return self.head(Y)


class YingYangNet16(YingYangNet):
    def __init__(self, lr=0.1, inc=2, num_classes=10):
        
        #YY Residual (3, 2), (16, 16): 40 epochs, 84%
        
        repeat_depth = 1
        reduction_depth = 4
        cnn_dropout = 0
        
        depths, channels = (3, 2), (16, 64)
        
        super().__init__(
            ((depths[0], channels[0], cnn_dropout, inc, repeat_depth)),
            ((depths[1], channels[1],  reduction_depth)),
             stride=2, lr=lr, num_classes=num_classes,
             dropout = 0)
             
             
#lr = 3e-3
lr = 18e-5
print('Starting Model...')
model = YingYangNet16(lr=lr, num_classes=1000)
model.to(try_gpu())
trainer = d2l.Trainer(max_epochs=300, num_gpus=1, gradient_clip_val=1)
data = d2l.DT_ImageNet(train_tfms, val_tfms, batch_size=32)

trainer.optim = model.configure_optimizers(weight_decay=lr*1.56)
trainer.epoch = None
trainer.sched = None
trainer.best_acc = 0


checkpoint = torch.load('./imagenet_checkpoints/checkpoint126')
model.load_state_dict(checkpoint['model_state_dict'])
trainer.optim.load_state_dict(checkpoint['optimizer_state_dict'])
trainer.epoch = checkpoint['epoch']+1
#trainer.sched = checkpoint['scheduler_state_dict']

trainer.sched = torch.optim.lr_scheduler.OneCycleLR(trainer.optim, lr, 300, steps_per_epoch=len(data.train_dataloader()))
trainer.sched.load_state_dict(checkpoint['scheduler_state_dict'])
trainer.loss = checkpoint['loss']
trainer.best_acc = checkpoint['best_acc']




print('Starting Training.')
trainer.fit(model, data, 
             lr_update=0.1, lr_reduct_rule=-1, lr_limit = 1e-10, weight_decay=lr*1.56)
#summary(model, (3,224,224))
