# PGMamba: A Physical Model-Guided Global Mamba for Underwater Image Enhancement
## Abtract
Underwater image enhancement (UIE) aims to address image degradation caused by water absorption and scattering effects. Despite significant progress in deep learning-based UIE methods, existing approaches still face key challenges due to the neglect of physical imaging principle. Moreover, while current Mamba models achieve global modeling via multi-directional scanning, their local sequential strategy lacks sufficient global context. To this end, we propose a novel Physical Model-Guided Global Mamba (PGMamba) that combines the efficient sequential modeling capability of Mamba with underwater imaging physical model. Specifically, we first design a Spatial-Aware Global Mamba (SAGMamba) that achieves efficient long-range dependency modeling through a spatial-aware ranking strategy with global context information. Second, we develop a Physical Model-Guided Feed-Forward Network (PMGFFN) that explicitly incorporates underwater optical imaging principles into the network architecture. Extensive experimental results and comprehensive ablation studies demonstrate the outstanding performance and importance of our proposed method.
## Get Strat
### Data Preparation
First, download the UIEB and LSUI datasets, and divide each into training and testing sets. Then, use generate_depth_grad.py to generate the depth and gradient maps for the training set. 
You can get the follow sturcture:
```
UIEB
├─train
    ├─raw
    ├─raw_depth
    ├─raw_grad
    ├─reference
├─test
    ├─raw
    ├─reference
```
### Environment setup
torch=2.0.0
torchaudio=2.0.1  
torchmetrics=1.5.2    
torchvision= 0.15.1

cd pytorch-gradual-warmup-lr/
python setup.py install
cd ..

### Train
```
python main.py --mode train --data_dir your dataset path
```
### Test
```
python main.py --mode test --data_dir your dataset path --test_model weight path --save_image True --save_path result
```
## Acknowledgement
This repository is built under the help of the projects [IRNeXt](https://github.com/c-yn/IRNeXt). Thanks for their excellent work.



