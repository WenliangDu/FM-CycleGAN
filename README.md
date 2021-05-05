# FM-CycleGAN
Source codes of "Exploring the Potential of Unsupervised Image Synthesis for SAR-Optical Image Matching" IEEE Access

## Prerequisites
- Python 3
- Anaconda 3
- NVIDIA GPU + CUDA cuDNN

## Getting Started
### Training
```
python train.py --dataroot ../../../Datasets/SEN12FeildHalf --name FMCycleGAN --num_D 1 --netD multi --lambda_identity 0 --input_nc 1 --output_nc 1
```

## Acknowledgments
This code borrows heavily from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
