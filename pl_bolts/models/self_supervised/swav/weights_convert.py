import torch
from collections import OrderedDict

swav_imagenet = torch.load('swav_imagenet.pth.tar')

new_state_dict = OrderedDict()

for key in swav_imagenet.keys():
    if 'prototype' in key:
        continue
    new_state_dict[key.replace('module.', 'model.')] = swav_imagenet[key]

stl10_save = torch.load("epoch=96.ckpt")
stl10_save['state_dict'] = new_state_dict

torch.save(stl10_save, 'swav_imagenet.ckpt')