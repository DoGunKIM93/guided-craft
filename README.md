 

# Guided-Craft for TISR2024 Track2 x16

## Overview of Guided-Craft
![AIR](https://github.com/DoGunKIM93/guided-craft/assets/16958744/6e1523bc-214a-4cbc-b6aa-4288113a49c6)

## PREREQUISITES
Prerequisites for Guided-Craftt.

## OS
AIR Research Framework is supported on Ubuntu 16.04 LTS or above.

## Python
It is recommended that use Python 3.7 or greater, which can be installed either through the Anaconda package manager or the Python website.

## Pytorch
Recommended that use Pytorch 1.5.0 or above version.
Important: EDVR or some models that have dependency on Deformable Convolution Networks feature only works in Pytorch 1.5.0a0+8f84ded.

## Pull container image
At the first, pull docker container image.
docker pull nvcr.io/nvidia/pytorch:20.03-py3

## Clone
```
git clone https://github.com/DoGunKIM93/guided-craft.git
```

## Install some required packages
```
pip install fast_slic munch IQA_pytorch pillow
```

## Dataset
TISR2024 Track2 x16 Dataset
```
datasetPath: 'dataset directory path' (in Param.yaml)
```

## Pre-trained
Guided-Craft Pre-trained
```
pretrainedPath: 'Pre-trained directory path' (in Param.yaml)
```

## Train 
At Guided-Craft folder, type following command:
```
python main.py
```
## Test
At Guided-Craft folder, type following command:
```
python main.py -it
```
