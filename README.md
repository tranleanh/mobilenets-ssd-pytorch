# MobileNet-SSD and MobileNetV2-SSDLite with PyTorch

Object Detection with MobileNet-SSD, MobileNetV2-SSDLite on VOC, BDD100K Datasets

==================================> (LOADING 50%...)

## Results
1. Detection

<img src="readme_images/detection_105e.jpg" width="1200">

2. View the result on [Youtube](https://www.youtube.com/watch?v=0u3f4t-Wkv4)

## Dependencies
- Python 3.6+
- OpenCV
- PyTorch
- Pyenv (optional)

## Dataset Path (optional)
The dataset path should be structured as follow:
```bashrc
|- bdd100k -- bdd100k -- images -- 100k -- train
|                     |                 |- val
|                     |- labels
|                     |- xml -- train
|                            |- val
|
|- MobileNets-SSD -- data -- VOCdevkit -- test -- VOC2007
     (our repo)   |                    |- VOC2007
                  |- bdd_files
                  |- images
                  |- models
                  |- ...
                  |- train_ssd_BDD.py
                  |- ssd_test_img.py
                  |- ...
```
## Pre-processing
1. Convert BDD100K anotation format (.json) to VOC anotation format (.xml)
```bashrc
$ python bdd2voc.py
```
2. Remove training samples having no anotation (70000 to 69863)
```bashrc
notebook: remove_nolabel_samples_bdd.ipynb
```
## Download Pre-trained Models (VOC)
1. MobileNet-SSD
```bashrc
$ wget -P models https://storage.googleapis.com/models-hao/mobilenet-v1-ssd-mp-0_675.pth
```
2. MobileNetV2-SSDLite
```bashrc
$ wget -P models https://storage.googleapis.com/models-hao/mb2-ssd-lite-mp-0_686.pth
```
## Train
## Test
1. Test on image
```bashrc
$ python ssd_test_img.py
```
2. Test on video
```bashrc
$ python ssd_test_video.py
```

## References
- https://github.com/qfgaohao/pytorch-ssd

April 2020

Tran Le Anh
