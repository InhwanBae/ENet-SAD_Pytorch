# ENet-SAD_Pytorch
 Pytorch implementation of "Learning Lightweight Lane Detection CNNs by Self Attention Distillation (ICCV 2019)"

## Demo


## Train
### Requirements
* pytorch
* tensorflow (for tensorboard)
* opencv
* numpy
* scipy
* tqdm

### Datasets
* [CULane](https://xingangpan.github.io/projects/CULane.html)
* [TuSimple](https://github.com/TuSimple/tusimple-benchmark/issues/3)
* [BDD100K](http://bdd-data.berkeley.edu/)

### Training
```
python train.py --exp_dir ./experiments/exp1
```

## Performance


## Acknowledgement
This repo is built upon official implementation [ENet-SAD](https://github.com/cardwing/Codes-for-Lane-Detection) and based on [PyTorch-ENet](https://github.com/davidtvs/PyTorch-ENet), [SCNN_Pytorch](https://github.com/harryhan618/SCNN_Pytorch).
