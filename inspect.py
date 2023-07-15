import torch 
from torchvision.models import efficientnet_v2_s
import torch.nn as nn
from pytorch_nndct.apis import Inspector
inspector = Inspector("DPUCAHX8L_ISA0_SP")
model = efficientnet_v2_s() 
num_ftrs = model.classifier[1].in_features model.classifier[1] = nn.Linear(num_ftrs, 10)        model.load_state_dict(torch.load('float/efficientnetv2.pth'))model = model.to(device)
model.eval()

input = torch.randn([16, 3, 224, 224])
inspector.inspect(model, input)
print(inspector)