Actualy i used 3.5 vitis AI instead of 3.0, the folder contains script to run quantization and also contains compiled blocks for ultra96v2 fpga. 

To compile the blocks 
```
./qt_and_compile_sh
```
To Comile for other fpga boards change arch.json file accordingly.  
Log file contains logs of compilation process.

All this was done so that each block can be deployed in FPGA to measure the latency in order to build Latency table that can be utilized to in HW_NAS process.
