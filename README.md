# Fingerprint 2D pose estimation
Pytorch implementation of paper "Estimating Fingerprint Pose via Dense Voting"

<img src="./image_feature/pose_2d/gridnet4/1_1.png" width="20%" height="20%">

## Requirements
- numpy, scipy, imageio, PyYaml, Pillow
- opencv==3.4.2.17
- pytorch==1.10.0

## Deploy
1. download the [trained model](https://cloud.tsinghua.edu.cn/f/685981cc0d2f4d48ad41/?dl=1) and unzip it
2. place the model in the folder [./logs](./logs)
3. adjust the related parameters `prefix` in the file [./deploy_gridnet.py](./deploy_gridnet.py) according to your image location
4. run `python deploy_gridnet.py -i <img_name.img_format>`

## Contact
If you have any questions about our work, please contact [dyj17@mails.tsinghua.edu.cn]()

## Citation
If you find this project helpful, please cite our paper:
```
@ARTICLE{10100723,
  author={Duan, Yongjie and Feng, Jianjiang and Lu, Jiwen and Zhou, Jie},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Estimating Fingerprint Pose via Dense Voting}, 
  year={2023},
  volume={18},
  number={},
  pages={2493-2507},
  doi={10.1109/TIFS.2023.3266625}}
```