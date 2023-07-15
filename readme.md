### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Evaluation](#evaluation)
4. [Performance](#performance)
5. [Model_info](#model_info)

### Installation

   1. Environment requirement
    - pytorch, opencv, tqdm ...
    - vai_q_pytorch(Optional, required by quantization)
    - XIR Python frontend (Optional, required by quantization)

   2. Installation with GPU Docker
   - Please refer to [vitis-ai](https://github.com/Xilinx/Vitis-AI/tree/master/) for how to obtain the GPU docker image.
   - Install all the python dependencies using pip:
   ```shell
   pip install --user -r requirements.txt
   ```
   
   3. Installation without GPU Docker

   - Create virtual envrionment and activate it:
   ```shell
   conda create -n torch_inceptionv3 python=3.7
   conda activate torch_inceptionv3
   ```
   - Install all the python dependencies using pip:
   ```shell
   pip install --user -r requirements.txt
   ```
   
### Preparation

  ImageNet dataset link: [ImageNet](http://image-net.org/download-images)

  ```
  The downloaded ImageNet dataset needs to be organized as the following format:
    1) Create a folder of "data" . Put the validation data in +data/val.
    2) Data from each class for validation set needs to be put in the folder:
    +data/Imagenet/val
         +val/n01847000
         +val/n02277742
         +val/n02808304
         +...
  ```

### Evaluation
1. Evaluate float model
  ```shell
  sh run_test_float.sh
  ```
2. Evaluate quantized(INT8) model
  ```shell
  sh run_test_quantized.sh
  ```

### Performance

|Model |input size|FLOPs|Params| Float top-1/5 acc(%)| Quantized top-1/5 acc(%)|
|----|---|---|---|---|---|
|Baseline | 299*299|11.42G |27.16M | 77.5/93.6| 77.1/93.5 |
|Prune0.3| 299*299 |8.02G |19.77M | 77.5/93.6| 77.0/93.4 |
|Prune0.4| 299*299 |6.82G |16.35M | 76.8/93.1| 76.4/92.8 |
|Prune0.5| 299*299 |5.68G |13.34M | 75.7/92.1| 74.9/91.9 |
|Prune0.6| 299*299 |4.54G |10.58M | 73.9/91.1| 73.0/90.7 |


### Model_info

1.  data preprocess
  ```
  data channel order: BGR(0~255)
  resize: short side reisze to 256 and keep the aspect ratio
  center crop: 299 * 299
  ```
