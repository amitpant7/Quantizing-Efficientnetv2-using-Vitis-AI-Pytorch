import torch
import torch.nn as nn

class myModel(nn.Module):
    def __init__(self, layers, input):
        super(myModel, self).__init__()
        self.first = nn.Conv2d(in_channels= input[1], out_channels= input[1], kernel_size= 1)
        self.blocks = nn.ModuleList(layers)

    def forward(self, x):
        output = []
        y = self.first(x)
        for i, _ in enumerate(self.blocks):
             y1 = self.blocks[i](y)
             output.append(y1)
        return output



class MyLargeModel(nn.Module):
    def __init__(self, model1, model2, index = 0):
        super(MyLargeModel, self).__init__()
        self.block1 = model1
        self.block2 = model2
        self.index = index

    def forward(self, x):
        y1 = self.block1(x)
        y = self.block2(y1[self.index])  # as models reutun list as defined 
        return y
    
