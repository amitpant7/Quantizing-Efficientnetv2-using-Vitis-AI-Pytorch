# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class ResidualBlock(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.module_0 = py_nndct.nn.Module('nndct_const') #ResidualBlock::933
        self.module_1 = py_nndct.nn.Module('nndct_const') #ResidualBlock::937
        self.module_2 = py_nndct.nn.Module('nndct_const') #ResidualBlock::941
        self.module_3 = py_nndct.nn.Module('nndct_const') #ResidualBlock::935
        self.module_4 = py_nndct.nn.Module('nndct_const') #ResidualBlock::939
        self.module_5 = py_nndct.nn.Module('nndct_const') #ResidualBlock::943
        self.module_6 = py_nndct.nn.Input() #ResidualBlock::input_0
        self.module_7 = py_nndct.nn.Conv2d(in_channels=136, out_channels=544, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Conv2d[conv]/input.3
        self.module_8 = py_nndct.nn.Add() #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Hswish[act]/934
        self.module_9 = py_nndct.nn.Module('nndct_relu6',inplace=True) #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Hswish[act]/791
        self.module_10 = py_nndct.nn.Module('nndct_elemwise_mul') #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Hswish[act]/792
        self.module_11 = py_nndct.nn.Module('nndct_elemwise_div') #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Hswish[act]/input.5
        self.module_12 = py_nndct.nn.Conv2d(in_channels=544, out_channels=544, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=544, bias=True) #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[depth_conv]/Conv2d[conv]/input.7
        self.module_13 = py_nndct.nn.Add() #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[depth_conv]/Hswish[act]/938
        self.module_14 = py_nndct.nn.Module('nndct_relu6',inplace=True) #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[depth_conv]/Hswish[act]/823
        self.module_15 = py_nndct.nn.Module('nndct_elemwise_mul') #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[depth_conv]/Hswish[act]/824
        self.module_16 = py_nndct.nn.Module('nndct_elemwise_div') #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[depth_conv]/Hswish[act]/940
        self.module_17 = py_nndct.nn.Mean() #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[depth_conv]/SEModule[se]/831
        self.module_18 = py_nndct.nn.Mean() #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[depth_conv]/SEModule[se]/input.9
        self.module_19 = py_nndct.nn.Conv2d(in_channels=544, out_channels=136, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[depth_conv]/SEModule[se]/Sequential[fc]/Conv2d[reduce]/input.11
        self.module_20 = py_nndct.nn.ReLU(inplace=True) #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[depth_conv]/SEModule[se]/Sequential[fc]/ReLU[relu]/input.13
        self.module_21 = py_nndct.nn.Conv2d(in_channels=136, out_channels=544, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[depth_conv]/SEModule[se]/Sequential[fc]/Conv2d[expand]/875
        self.module_22 = py_nndct.nn.Add() #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[depth_conv]/SEModule[se]/Sequential[fc]/Hsigmoid[h_sigmoid]/942
        self.module_23 = py_nndct.nn.Module('nndct_relu6',inplace=True) #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[depth_conv]/SEModule[se]/Sequential[fc]/Hsigmoid[h_sigmoid]/879
        self.module_24 = py_nndct.nn.Module('nndct_elemwise_div') #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[depth_conv]/SEModule[se]/Sequential[fc]/Hsigmoid[h_sigmoid]/944
        self.module_25 = py_nndct.nn.Module('nndct_elemwise_mul') #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[depth_conv]/SEModule[se]/input.15
        self.module_26 = py_nndct.nn.Conv2d(in_channels=544, out_channels=136, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[point_linear]/Conv2d[conv]/input
        self.module_27 = py_nndct.nn.Add() #ResidualBlock::ResidualBlock/910

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(data=3.0, dtype=torch.float, device='cpu')
        output_module_1 = self.module_1(data=3.0, dtype=torch.float, device='cpu')
        output_module_2 = self.module_2(data=3.0, dtype=torch.float, device='cpu')
        output_module_3 = self.module_3(data=6.0, dtype=torch.float, device='cpu')
        output_module_4 = self.module_4(data=6.0, dtype=torch.float, device='cpu')
        output_module_5 = self.module_5(data=6.0, dtype=torch.float, device='cpu')
        output_module_6 = self.module_6(input=args[0])
        output_module_7 = self.module_7(output_module_6)
        output_module_8 = self.module_8(input=output_module_7, other=output_module_0, alpha=1)
        output_module_8 = self.module_9(output_module_8)
        output_module_10 = self.module_10(input=output_module_7, other=output_module_8)
        output_module_10 = self.module_11(input=output_module_10, other=output_module_3)
        output_module_10 = self.module_12(output_module_10)
        output_module_13 = self.module_13(input=output_module_10, other=output_module_1, alpha=1)
        output_module_13 = self.module_14(output_module_13)
        output_module_15 = self.module_15(input=output_module_10, other=output_module_13)
        output_module_15 = self.module_16(input=output_module_15, other=output_module_4)
        output_module_17 = self.module_17(input=output_module_15, dim=3, keepdim=True)
        output_module_17 = self.module_18(input=output_module_17, dim=2, keepdim=True)
        output_module_17 = self.module_19(output_module_17)
        output_module_17 = self.module_20(output_module_17)
        output_module_17 = self.module_21(output_module_17)
        output_module_17 = self.module_22(input=output_module_17, other=output_module_2, alpha=1)
        output_module_17 = self.module_23(output_module_17)
        output_module_17 = self.module_24(input=output_module_17, other=output_module_5)
        output_module_25 = self.module_25(input=output_module_15, other=output_module_17)
        output_module_25 = self.module_26(output_module_25)
        output_module_25 = self.module_27(input=output_module_25, other=output_module_6, alpha=1)
        return output_module_25
