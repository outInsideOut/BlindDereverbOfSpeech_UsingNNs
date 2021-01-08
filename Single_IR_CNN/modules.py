# https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/audio_classifier_tutorial.ipynb#scrollTo=RxDwF22xSVBz
# https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html

import torch
from torch import nn

class Net(nn.Module):
    def __init__(self, inLayers, outLayers):
        super(Net, self).__init__()
        # architecture
        self.lyr1 = nn.Conv2d(in_channels=inLayers, out_channels=6, kernel_size=10)
        self.lyr2 = nn.Conv2d(in_channels=6, out_channels=1, kernel_size=5)
        self.lyr3 = nn.Upsample(size=20000)
        # Functions
        self.relu = nn.ReLU()
       
    def forward(self, x):
        out = self.lyr1(x)
        out = self.relu(out)
        # print(f'layer 1: {out.shape}')
        out = self.lyr2(out)
        out = self.relu(out)
        # print(f'layer 2: {out.shape}')
        out = self.lyr3(out)
        out = out.reshape(-1, 20000)
        # print(f'reshape {out.shape}')

        return out

