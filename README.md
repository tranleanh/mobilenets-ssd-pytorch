# MobileNet-SSD and MobileNetV2-SSD with PyTorch

Object Detection with MobileNet-SSD, MobileNetV2-SSD on VOC, BDD100K Datasets

===> (LOADING 20%...)

## Results
1. Detection

<img src="readme_images/detection_105e.jpg" width="1200">

2. View the result on [Youtube](https://www.youtube.com/watch?v=0u3f4t-Wkv4)

## Dependencies
- Python 3.6+
- OpenCV
- PyTorch
- Pyenv (optional)

## Pre-processing
1. Convert BDD100K anotation format (.json) to VOC anotation format (.xml)
```bashrc
$ python bdd2voc.py
```
2. Remove training samples having no anotation (70000 to 69863)
```bashrc
notebook: remove_nolabel_samples_bdd.ipynb
```

## Train
## Test

## References
- https://github.com/qfgaohao/pytorch-ssd

April 2020

Tran Le Anh
