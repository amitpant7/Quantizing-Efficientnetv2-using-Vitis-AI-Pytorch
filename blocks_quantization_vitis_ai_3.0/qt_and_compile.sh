#!/bin/bash

mapfile files < <(ls ./float/)

for file in $(files[@])
do
    echo starting quantization of $file
    name=$file 
    in_shape = $(echo $(file) | grep -o '-in_.*-out_')
    echo in_shape
    IFS='x'
    #convert to list[]
    read -ra input_shape<<<$(in_shape) 
    echo input_shape
    #  To quantize certain block of model 
    # /opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/python code/quantization.py --device "cpu" --quant_mode calib --subset_len 100 --model_name $name --input_shape $shape
    # To deploy model and get the qunatized model in (onnx, pth, xmodel formta)
    # sudo /opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/python ./code/quantization.py --quant_mode test --subset_len 1 --batch_size=1 --deploy --model_name $name --input_shape $shape

    mv compiled_model/*.xmodel ./compiled_all_blocks/
    rm -r compiled_model/*
    rm -r quantized
    rm -r quantize_result

done