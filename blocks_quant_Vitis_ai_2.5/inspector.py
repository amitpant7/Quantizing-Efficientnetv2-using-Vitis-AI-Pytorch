import torch 
import torch.nn as nn
from blocks import *

class myModel(nn.Module):
    def __init__(self, layers):
        super(myModel, self).__init__()

        self.blocks = nn.ModuleList(layers)

    def forward(self, x):
        output = []
        for i, _ in enumerate(self.blocks):
            y = self.blocks[i](x)
            output.append(y)
        return output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('./float/b3_48_28_28.pth')
model = model.to(device)


from pytorch_nndct.apis import Inspector
inspector = Inspector("DPUCZDX8G_ISA1_B4096") 


input = torch.randn(16, 48, 28, 28)
inspector.inspect(model, (input,), device=device, output_dir="inspect", image_format="png")
