
  - Main only contains the scripts to run quantization on efficientnetv2, check other branches for quantized models  
  - check the replaced branch for efficientnetV2 with the replaced activation function, hard swish.  
  - Compiled branch contains code for the compiled version of efficientnet v2 after sturctured pruning, quantization and compilation for DPU 1024, Zynq Mpsoc FPGA  

### Quantized and Compiled Model For deploying on FPGA 
- Branch : [Compiled](https://github.com/amitpant7/Quantizing-Efficientnetv2-using-Vitis-AI-Pytorch/tree/compiled)
### Creating blocks to Deploy in FPGA for Latency Table Construction 
- Branch: [qt_for_latency_blocks](https://github.com/amitpant7/Quantizing-Efficientnetv2-using-Vitis-AI-Pytorch/tree/qt_for_latency_blocks)
### Installation

   1. Environment requirement
    - pytorch, opencv, tqdm ...
    - vai_q_pytorch(Optional, required by quantization)
    - XIR Python frontend (Optional, required by quantization)

   2. Installation with GPU Docker
   - Please refer to [vitis-ai](https://github.com/Xilinx/Vitis-AI/tree/master/) for how to obtain the GPU docker image.
   
   3. Installation without GPU Docker

   - Create virtual envrionment and activate it:
   ```shell
   conda activate vitis-ai-pytorch
   ```

### Preparation
   ## Dataset
   Donload CIFAR-10 Dataset, 
   or you can set down-true inside the ```load_data``` in ```efficientnetv3_quant.py```



### Quantization
   ```
python ./code/test/efficientnetv2_quant.py --device "cpu" --quant_mode calib --subset_len 1000

```
To deploy model and export it in xmodel format:
```
sudo /opt/vitis_ai/conda/envs/vitis-ai-wego-torch/bin/python ./code/test/efficientnetv2_quant.py --quant_mode test --subset_len 1 --batch_size=1 --deploy 
```

Similarly for quantization-aware training  
Fast fintune models 2000 images, calib 2000 images  
```
export W_QUANT=1
```
```
sudo /opt/vitis_ai/conda/envs/vitis-ai-wego-torch/bin/python code/efficientnetv2_QAT.py --device "cpu" --quant_mode calib --fast_finetune --subset_len 50
```
Test
```
python code/efficientnetv2_QAT.py --device "cpu" --quant_mode test --fast_finetune
```
--deploy
 ```
python code/efficientnetv2_QAT.py --device "cpu" --quant_mode test --fast_finetune --subset_len 1 --batch_size 1 --deploy
```

