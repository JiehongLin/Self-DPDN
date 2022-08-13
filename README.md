# Self-Supervised Deep Prior Deformation Network
Code for "Category-Level 6D Object Pose and Size Estimation using Self-Supervised Deep Prior Deformation Networks". ECCV2022.

Created by [Jiehong Lin](https://jiehonglin.github.io/), Zewei Wei, Changxing Ding, and [Kui Jia](http://kuijia.site/).

![image](https://github.com/JiehongLin/Self-DPDN/blob/main/pic/overview.jpg)

## Requirements
The code has been tested with
- python 3.6.5
- pytorch 1.3.0
- CUDA 10.2

Some dependent packages：
- [gorilla](https://github.com/Gorilla-Lab-SCUT/gorilla-core) 
```
pip install gorilla-core==0.2.5.6
```
- [pointnet2](https://github.com/erikwijmans/Pointnet2_PyTorch)
```
cd model/pointnet2
python setup.py install
```

## Data Processing

Download the data provided by [NOCS](https://github.com/hughw19/NOCS_CVPR2019) ([camera_train](http://download.cs.stanford.edu/orion/nocs/camera_train.zip), [camera_test](http://download.cs.stanford.edu/orion/nocs/camera_val25K.zip), [camera_composed_depths](http://download.cs.stanford.edu/orion/nocs/camera_composed_depth.zip), [real_train](http://download.cs.stanford.edu/orion/nocs/real_train.zip), [real_test](http://download.cs.stanford.edu/orion/nocs/real_test.zip),
[ground truths](http://download.cs.stanford.edu/orion/nocs/gts.zip),
and [mesh models](http://download.cs.stanford.edu/orion/nocs/obj_models.zip) (also some [processed '*.pkl' files](https://drive.google.com/file/d/1Nz7cwcQWO_In4K6jKN1-5pQ0orY4UV9x/view?usp=sharing))) and segmentation results ([Link](https://drive.google.com/file/d/1hNmNRr7YRCgg-c_qdvaIzKEd2g4Kac3w/view?usp=sharing)), and unzip them in data folder as follows:

```
data
├── CAMERA
│   ├── train
│   └── val
├── camera_full_depths
│   ├── train
│   └── val
├── Real
│   ├── train
│   └── test
├── gts
│   ├── val
│   └── real_test
├── obj_models
│   ├── train
│   ├── val
│   ├── real_train
│   └── real_test
├── segmentation_results
│   ├── train_trainedwoMask
│   ├── test_trainedwoMask
│   └── test_trainedwithMask
└── mean_shapes.npy
```
Run the following scripts to prepare the dataset:

```
python data_processing.py
```
## Training DPDN under Different Settings

Train DPDN under unsupervised setting:
```
python train.py --gpus 0,1 --config config/unsupervised.yaml
```
Train DPDN under unsupervised setting (with mask labels):
```
python train.py --gpus 0,1 --config config/unsupervised_withMask.yaml
```
Train DPDN under supervised setting:
```
python train.py --gpus 0,1 --config config/supervised.yaml
```
## Evaluation
Download trained models and test results [Link]. Evaluate our models under different settings:
```
python test.py --config config/unsupervised.yaml
python test.py --config config/unsupervised_withMask.yaml
python test.py --config config/supervised.yaml
```
or directly evaluate our results on REAL275 test set:
```
python test.py --config config/unsupervised.yaml --only_eval
python test.py --config config/unsupervised_withMask.yaml --only_eval
python test.py --config config/supervised.yaml --only_eval
```

## Results
Qualitative results on REAL275 test set:

|   | IoU25 | IoU75 | 5 degree 2 cm | 5 degree 5 cm | 10 degree 2 cm | 10 degree 5 cm |
|---|---|---|---|---|---|---|
| unsupervised | 72.6 | 63.8 | 37.8 | 45.5 | 59.8 | 71.3 |
| unsupervised (with masks) | 78.2 | 64.5 | 39.4 | 44.2 | 61.4 | 70.9 |
| supervised  | 83.4 | 76.0 | 46.0 | 50.7 | 70.4 | 78.4 |


## Acknowledgements

Our implementation leverages the code from [NOCS](https://github.com/hughw19/NOCS_CVPR2019), [DualPoseNet](https://github.com/Gorilla-Lab-SCUT/DualPoseNet), and [SPD](https://github.com/mentian/object-deformnet).

## License
Our code is released under MIT License (see LICENSE file for details).

## Contact
`lin.jiehong@mail.scut.edu.cn`
