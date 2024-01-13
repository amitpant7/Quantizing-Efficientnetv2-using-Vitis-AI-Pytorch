#!/bin/bash
#cleaning up 
sudo rm -r ./compiled_all_blocks/*
counter=0

for file in ./float/*
do
if [ -f "$file" ]; then 
    name=$(echo "$file" | sed 's|./float/||') 
    in_shape=$(echo "$name" | grep -o 'in_.*-out' | sed 's/in_//;s/-out//')
    IFS='x'
   
    echo -----------------------Strating Qt $name -----------------------------

    #convert to list[]
    read -ra input_shape<<< $in_shape
    input=$(echo [${input_shape[0]}, ${input_shape[1]}, ${input_shape[2]}])

    #  To quantize certain block of model 
    sudo /opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/python code/quantization.py --device "cpu" --quant_mode calib --subset_len 100 --model_name "$name"  --input_shape $input
    echo Compilation
    sudo /opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/python ./code/quantization.py --quant_mode test --subset_len 1 --batch_size=1 --deploy --model_name "$name" --input_shape $input

    deploy_name=$(echo "$name" |sed 's/.pth//')
      
# To deploy model and get the qunatized model in (onnx, pth, xmodel formta)
    sudo -E PATH=$PATH:/opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin /opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/vai_c_xir -x ./quantize_result/*.xmodel -a arch.json -o ./compiled -n "$deploy_name"
    
   sudo mkdir ./compiled_all_blocks/"$deploy_name"
   sudo mv ./compiled/* ./compiled_all_blocks/"$deploy_name"
   #sudo rm -r ./compiled/*
   sudo rm -r ./quantized/*
   sudo rm -r ./quantize_result/*
  
   ((counter++))
   echo -e "\n\n"
   echo -e "---------------------------Done Block $counter-------------------------\n\n"
  
fi
done
