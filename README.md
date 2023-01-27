# pytorch-test-ARC-A770
This is a repo for deep learning network codes tested on Intel Arc A770 GPU.

Configuration for test machine:
* Windows 11 Pro for Workstation 22H2 (22621.525)
* WSL2 with Ubuntu 22.04 LTS (5.15.79.1-microsoft-standard-WSL2)
* Intel Core i9-12900K
* 64GB DDR5-5600 U-DIMM
* Intel Arc A770 16GB Limited Edition 
* Plextor M10PGN 1TB SSD

 Suggested driver and Pytorch from [instruction](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/installation.html) by Intel are used:
 * intel-level-zero-gpu 1.3.24595.35+i538~22.04 amd64 Intel(R) Graphics Compute Runtime for oneAPI Level Zero.
 * torch==1.13.0a0
 * torchvision==0.14.1a0
 * intel_extension_for_pytorch==1.13.10+xpu
