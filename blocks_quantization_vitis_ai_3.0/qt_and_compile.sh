#!/bin/bash


for file in ./float/*
do
if [ -f "$file" ]; then
    echo ------------------------starting quantization-------------------- 
    name=$(echo "$file" | sed 's|./float/||') 
    in_shape=$(echo $file | grep -o 'in_.*-out' | sed 's/in_//;s/-out//')
    IFS='x'
   
    #convert to list[]
    read -ra input_shape<<< $in_shape
    input=$(echo [${input_shape[0]}, ${input_shape[1]}, ${input_shape[2]}])

    #  To quantize certain block of model 
    sudo /opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/python code/quantization.py --device "cpu" --quant_mode calib --subset_len 10 --model_name "$name"  --input_shape $input
    echo Compilation
    sudo /opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/python ./code/quantization.py --quant_mode test --subset_len 1 --batch_size=1 --deploy --model_name "$name" --input_shape $input

    # To deploy model and get the qunatized model in (onnx, pth, xmodel formta)
    sudo -E PATH=$PATH:/opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin /opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/vai_c_xir -x ./quantize_result/*.xmodel -a arch.json -o ./compiled -n "$name"

    mv compiled_model/*.xmodel ./compiled_all_blocks/
    rm -r compiled_model/*
    rm -r quantized/*
    rm -r quantize_result/*
   break

fi
done
