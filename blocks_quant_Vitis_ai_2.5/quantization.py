import torch 

from pytorch_nndct.apis import torch_quantizer, dump_xmodel

quant_mode = input('Enter Quant mode:')
device = input('Enter device')
deploy = input('Deploy or not ? :')


batch_size = 8
input = torch.randn([batch_size, 3, 224, 224])

model = torch.load('resnet.pth')

def evaluator(model):
    model.eval()
    model.to(device)
    
    for i in range(200):
        image = torch.randn([batch_size, 3, 224, 224])
        out = model(image)

quantizer = torch_quantizer(quant_mode, model, (input))
quant_model = quantizer.quant_model

#forward passing network
evaluator(quant_model)

if quant_mode == 'calib':
    quantizer.export_quant_config()
if deploy:
    quantizer.export_torch_script()
    quantizer.export_onnx_model()
    quantizer.export_xmodel(deploy_check=False)