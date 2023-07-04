# DPDN with Pytorch 1.9.0 and Cuda 11.2

The training and test commands are the same as [those](https://github.com/JiehongLin/Self-DPDN) with Pytorch 1.3.0 and Cuda 10.2.

## Requirements
```
    pip install gorilla-core==0.2.6.0
    pip install opencv-python==4.5.1.48
    pip install gpustat==1.0.0

    cd model/pointnet2
    python setup.py install
```

## Trained models

Trained models could be downloaded from this [link](https://drive.google.com/file/d/1k2OkP9blIi2gBZ4Fou3XpidKWem4XQ1F/view?usp=sharing).

## Results


Qualitative results on REAL275 dataset are similar to those obtained with Pytorch 1.3.0 and Cuda 10.2.

|   |  IoU25 | IoU75 | 5 degree 2 cm | 5 degree 5 cm | 10 degree 2 cm | 10 degree 5 cm |
|---|---|---|---|---|---|---|
| unsupervised |  72.7 | 63.4 | 38.1 | 45.6 | 60.0 | 71.2 |
| unsupervised (with masks) |  82.9 | 71.0 | 38.7 | 44.5 | 62.4 | 72.0 |
| supervised  |  83.5 | 74.4 | 45.3 | 52.9 | 68.0 | 78.4 |
