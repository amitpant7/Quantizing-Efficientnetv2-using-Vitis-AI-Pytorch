# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class ResidualBlock(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.module_0 = py_nndct.nn.Input() #ResidualBlock::input_0(ResidualBlock::nndct_input_0)
        self.module_1 = py_nndct.nn.Conv2d(in_channels=136, out_channels=544, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Conv2d[conv]/ret.3(ResidualBlock::nndct_conv2d_1)
        self.module_2 = py_nndct.nn.Hardswish(inplace=True) #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Hardswish[act]/1293(ResidualBlock::nndct_hswish_2)
        self.module_3 = py_nndct.nn.Conv2d(in_channels=544, out_channels=544, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=544, bias=True) #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[depth_conv]/Conv2d[conv]/ret.7(ResidualBlock::nndct_depthwise_conv2d_3)
        self.module_4 = py_nndct.nn.Hardswish(inplace=True) #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[depth_conv]/Hardswish[act]/1321(ResidualBlock::nndct_hswish_4)
        self.module_5 = py_nndct.nn.Module('aten::mean') #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[depth_conv]/SEModule[se]/1327(ResidualBlock::aten_mean_5)
        self.module_6 = py_nndct.nn.Module('aten::mean') #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[depth_conv]/SEModule[se]/1334(ResidualBlock::aten_mean_6)
        self.module_7 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[depth_conv]/SEModule[se]/SqueezeExcitation[fc]/AdaptiveAvgPool2d[avgpool]/1384(ResidualBlock::nndct_adaptive_avg_pool2d_7)
        self.module_8 = py_nndct.nn.Conv2d(in_channels=544, out_channels=136, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[depth_conv]/SEModule[se]/SqueezeExcitation[fc]/Conv2d[fc1]/ret.11(ResidualBlock::nndct_conv2d_8)
        self.module_9 = py_nndct.nn.ReLU(inplace=False) #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[depth_conv]/SEModule[se]/SqueezeExcitation[fc]/ReLU[activation]/ret.13(ResidualBlock::nndct_relu_9)
        self.module_10 = py_nndct.nn.Conv2d(in_channels=136, out_channels=544, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[depth_conv]/SEModule[se]/SqueezeExcitation[fc]/Conv2d[fc2]/ret.15(ResidualBlock::nndct_conv2d_10)
        self.module_11 = py_nndct.nn.Hardsigmoid(inplace=False) #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[depth_conv]/SEModule[se]/SqueezeExcitation[fc]/Hardsigmoid[scale_activation]/ret.17(ResidualBlock::nndct_hsigmoid_11)
        self.module_12 = py_nndct.nn.Module('nndct_elemwise_mul') #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[depth_conv]/SEModule[se]/SqueezeExcitation[fc]/ret.19(ResidualBlock::nndct_elemwise_mul_12)
        self.module_13 = py_nndct.nn.Module('nndct_elemwise_mul') #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[depth_conv]/SEModule[se]/ret.21(ResidualBlock::nndct_elemwise_mul_13)
        self.module_14 = py_nndct.nn.Conv2d(in_channels=544, out_channels=136, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResidualBlock::ResidualBlock/MBConvLayer[conv]/Sequential[point_linear]/Conv2d[conv]/ret.23(ResidualBlock::nndct_conv2d_14)
        self.module_15 = py_nndct.nn.Add() #ResidualBlock::ResidualBlock/ret(ResidualBlock::nndct_elemwise_add_15)

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_1 = self.module_1(output_module_0)
        output_module_1 = self.module_2(output_module_1)
        output_module_1 = self.module_3(output_module_1)
        output_module_1 = self.module_4(output_module_1)
        output_module_5 = self.module_5({'self': output_module_1,'dim': [3],'keepdim': True,'dtype': None})
        output_module_5 = self.module_6({'self': output_module_5,'dim': [2],'keepdim': True,'dtype': None})
        output_module_7 = self.module_7(output_module_5)
        output_module_7 = self.module_8(output_module_7)
        output_module_7 = self.module_9(output_module_7)
        output_module_7 = self.module_10(output_module_7)
        output_module_7 = self.module_11(output_module_7)
        output_module_7 = self.module_12(input=output_module_7, other=output_module_5)
        output_module_13 = self.module_13(input=output_module_1, other=output_module_7)
        output_module_13 = self.module_14(output_module_13)
        output_module_13 = self.module_15(input=output_module_13, other=output_module_0, alpha=1)
        return output_module_13
