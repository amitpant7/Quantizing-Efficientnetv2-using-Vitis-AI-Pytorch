import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('float/ResidualBlock-in_96x14x14-out_136-k_7-e_4-s_1-act_h_swish-use_se_True.pth')
model = model.to(device)


from pytorch_nndct.apis import Inspector
inspector = Inspector("DPUCZDX8G_ISA1_B2304") 


input = torch.randn(1, 96, 14, 14)
inspector.inspect(model, (input,), device=device, output_dir="inspect", image_format="svg")
