# ENet-SAD Pytorch
 Pytorch implementation of "[Learning Lightweight Lane Detection CNNs by Self Attention Distillation (ICCV 2019)](https://arxiv.org/abs/1908.00821)"

<img src="./image/ENet-SAD_paper_model_architecture.png" alt="drawing" width="750"/>

## Demo
#### Video
![demo_gif](./image/ENet-SAD_demo.gif)

Demo trained with CULane dataset & tested with \driver_193_90frame\06051123_0635.MP4

`gpu_runtime: 0.022898435592651367 FPS: 43` on RTX 2080 TI

#### Comparison
| Category | 40k episode (before SAD)                  | 60k episode (after SAD)                  |
| -------- | ----------------------------------------- | ---------------------------------------- |
| Image    | ![img1](./image/ENet_before_SAD.png)      | ![img2](./image/ENet_after_SAD.png)      |
| Lane     | ![img3](./image/ENet_before_SAD_lane.png) | ![img4](./image/ENet_after_SAD_lane.png) |

## Train
### Requirements
* pytorch
* tensorflow (for tensorboard)
* opencv

### Datasets
* [CULane](https://xingangpan.github.io/projects/CULane.html)
* [TuSimple](https://github.com/TuSimple/tusimple-benchmark/issues/3)
* [BDD100K](http://bdd-data.berkeley.edu/)
* [VPGNet](https://github.com/SeokjuLee/VPGNet/issues/50)

You need to change the correct dataset path in `./config.py`
```
Dataset_Path = dict(
    CULane = "/workspace/CULANE_DATASET",
    Tusimple = "/workspace/TUSIMPLE_DATASET",
    BDD100K = "/workspace/BDD100K_DATASET",
    VPGNet = "/workspace/VPGNet_DATASET"
)
```

### Training
First, change some hyperparameters in `./experiments/*/cfg.json`
```
{
  "model": "enet_sad",               <- "scnn" or "enet_sad"
  "dataset": {
    "dataset_name": "CULane",        <- "CULane" or "Tusimple"
    "batch_size": 12,
    "resize_shape": [800, 288]       <- [800, 288] with CULane, [640, 368] with Tusimple, and [640, 360] with BDD100K
                                        This size is defined in the ENet-SAD paper, any size is fine if it is a multiple of 8.
  },
  ...
}
```

And then, start training with `train.py`
```
python train.py --exp_dir ./experiments/exp1
```

## Performance


## Acknowledgement
This repo is built upon official implementation [ENet-SAD](https://github.com/cardwing/Codes-for-Lane-Detection) and based on [PyTorch-ENet](https://github.com/davidtvs/PyTorch-ENet), [SCNN_Pytorch](https://github.com/harryhan618/SCNN_Pytorch).
