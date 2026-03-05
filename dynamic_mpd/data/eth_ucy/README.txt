ETH/UCY Pedestrian Trajectory Dataset
============================================================

数据来源: Social GAN (Stanford)

数据集:
- eth.txt: ETH dataset
- hotel.txt: Hotel dataset
- univ.txt: University students
- zara1.txt: Zara1 dataset
- zara2.txt: Zara2 dataset

数据格式:
frame_id  ped_id  x  y

常用分割 (Leave-one-out):
- Train: 4个数据集, Test: 1个数据集
- 例如: Train=[eth,hotel,univ,zara1], Test=[zara2]
