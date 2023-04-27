# GridNet
GridNet - A dense voting method for estimating fingerprint poses

![](./image_feature/pose_2d/gridnet4/1_1.png)

## Requirements
pytorch==1.10.0, numpy==1.21.4, scipy==1.7.3, imageio==2.13.3, opencv==3.4.2.17, PyYaml==6.0, Pillow==8.4.0

## Train
1. create your own training list in `.pkl` file
2. create a new directory (e.g., `datasets`) and place the `.pkl` file in it
3. modify parameters in [./train_gridnet.yaml](./train_gridnet.yaml) for your own project
4. run `python trainer_i2p.py -g <GPU> -y gridnet`

## Deploy
1. download the [trained model](https://cloud.tsinghua.edu.cn/f/685981cc0d2f4d48ad41/?dl=1) and unzip it
2. place the model in the folder [./logs](./logs)
3. adjust the related parameters `prefix` in the file [./deploy_gridnet.py](./deploy_gridnet.py) according to your image location
4. run `python deploy_gridnet.py -i <img_name.img_format>`
