# Self-Supervised Deep Prior Deformation Network
Code for "Category-Level 6D Object Pose and Size Estimation using Self-Supervised Deep Prior Deformation Networks". ECCV2022.

[[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-20077-9_2) [[Arxiv]](https://arxiv.org/abs/2207.05444)

Created by [Jiehong Lin](https://jiehonglin.github.io/), Zewei Wei, Changxing Ding, and [Kui Jia](http://kuijia.site/).

![image](https://github.com/JiehongLin/Self-DPDN/blob/main/pic/overview.jpg)


## News
- 2023/08: Our another work named [VI-Net](https://github.com/JiehongLin/VI-Net) is accepted by ICCV2023.
- 2023/07: [Codes](https://github.com/JiehongLin/Self-DPDN/tree/main/DPDN-Pytorch1.9.0-Cuda11.2) tested with Pytorch 1.9.0 and CUDA 11.2 are also provided.

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
and [mesh models](http://download.cs.stanford.edu/orion/nocs/obj_models.zip)) and segmentation results ([Link](https://drive.google.com/file/d/1hNmNRr7YRCgg-c_qdvaIzKEd2g4Kac3w/view?usp=sharing)), and unzip them in data folder as follows:

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
Train DPDN under supervised setting:
```
python train.py --gpus 0,1 --config config/supervised.yaml
```
## Evaluation
Download trained models and test results [[Link](https://drive.google.com/file/d/1hWkbH4Z0RXQeYLxC_kOINxdTnDLtm9SX/view?usp=sharing)]. Evaluate our models under different settings:
```
python test.py --config config/unsupervised.yaml
python test.py --config config/supervised.yaml
```
or directly evaluate our results on REAL275 test set:
```
python test.py --config config/unsupervised.yaml --only_eval
python test.py --config config/supervised.yaml --only_eval
```
One can also evaluate our models under the easier unsupervised setting with mask labels for segmentation (still without pose annotations):
```
python test.py --config config/unsupervised.yaml --mask_label
```
or 
```
python test.py --config config/unsupervised.yaml --mask_label --only_eval
```
## Results
Qualitative results on REAL275 test set:

|   | IoU25 | IoU75 | 5 degree 2 cm | 5 degree 5 cm | 10 degree 2 cm | 10 degree 5 cm |
|---|---|---|---|---|---|---|
| unsupervised | 72.6 | 63.8 | 37.8 | 45.5 | 59.8 | 71.3 |
| unsupervised (with masks) | 83.0 | 70.3 | 39.4 | 45.0 | 59.8 | 72.1 |
| supervised  | 83.4 | 76.0 | 46.0 | 50.7 | 70.4 | 78.4 |


## Citation
If you find our work useful in your research, please consider citing:

    @inproceedings{lin2022category,
      title={Category-level 6D object pose and size estimation using self-supervised deep prior deformation networks},
      author={Lin, Jiehong and Wei, Zewei and Ding, Changxing and Jia, Kui},
      booktitle={European Conference on Computer Vision},
      pages={19--34},
      year={2022},
      organization={Springer}
    }



## Acknowledgements

Our implementation leverages the code from [NOCS](https://github.com/hughw19/NOCS_CVPR2019), [DualPoseNet](https://github.com/Gorilla-Lab-SCUT/DualPoseNet), and [SPD](https://github.com/mentian/object-deformnet).

## License
Our code is released under MIT License (see LICENSE file for details).

## Contact
`lin.jiehong@mail.scut.edu.cn`
