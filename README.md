# Localize to Classify and Classify to Localize: Mutual Guidance in Object Detection
By Heng Zhang, Elisa FROMONT, Sébastien LEFEVRE, Bruno AVIGNON
## Introduction
Most deep learning object detectors are based on the anchor mechanism and resort to the Intersection over Union (IoU) between predefined anchor boxes and ground truth boxes to evaluate the matching quality between anchors and objects. In this paper, we question this use of IoU and propose a new anchor matching criterion guided, during the training phase, by the optimization of both the localization and the classification tasks: the predictions related to one task are used to dynamically assign sample anchors and improve the model on the other task, and vice versa. This is the Pytorch implementation of Mutual Guidance detectors. For more details, please refer to our [ACCV paper](xxxxxxxx).
<img align="center" src="https://github.com/zhangheng19931123/MutualGuide/blob/master/doc/compare.png">
&nbsp;
&nbsp;
## Experimental results
### VOC2007 Test
| **Detector** | **Resolution** | **mAP** | **AP50** | **AP75** | **FPS** (2080Ti) | **Trained model** |
|:-------|:-----:|:-------:|:-------:|:-------:|:-------:|:-------:|
| FSSD (VGG16) | 320x320 | 54.1 | 80.1 | 58.3 |  | [Google Drive](xxxxxxxx) |
| FSSD (VGG16) + MG | 320x320 | **56.2** | **80.4** | **61.4** |  | [Google Drive](xxxxxxxx) |
| RetinaNet (VGG16) | 320x320 | 55.2 | 80.2 | 59.6 |  | [Google Drive](xxxxxxxx) |
| RetinaNet (VGG16) + MG | 320x320 | **57.7** | **81.1** | **62.9** |  | [Google Drive](xxxxxxxx) |
| RFBNet (VGG16) | 320x320 | 55.6 | 80.9 | 59.6 |  | [Google Drive](xxxxxxxx) |
| RFBNet (VGG16) + MG | 320x320 | **57.9** | **81.5** | **62.6** |  | [Google Drive](xxxxxxxx) |
| RetinaNet (VGG16) + PAFPN | 320x320 | 58.1 | 81.7 | 63.3 | | [Google Drive](xxxxxxxx) |
| RetinaNet (VGG16) + PAFPN + MG | 320x320 | **59.5** | **82.3** | **64.2** | | [Google Drive](xxxxxxxx) |
### COCO2017 Val
| **Detector** | **Resolution** | **mAP** | **AP50** | **AP75** | **FPS** (2080Ti) | **Trained model** |
|:-------|:-----:|:-------:|:-------:|:-------:|:-------:|:-------:|
| FSSD (VGG16) | 320x320 | 31.1 | 48.9 | 32.7 |  | [Google Drive](xxxxxxxx) |
| FSSD (VGG16) + MG | 320x320 | **32.0** | **49.3** | **33.9** |  | [Google Drive](xxxxxxxx) |
| RetinaNet (VGG16) | 320x320 | 32.3 | 50.3 | 34.0 |  | [Google Drive](xxxxxxxx) |
| RetinaNet (VGG16) + MG | 320x320 | **33.6** | **50.8** | **35.7** |  | [Google Drive](xxxxxxxx) |
| RFBNet (VGG16) | 320x320 | 33.4 | 51.6 | 35.1 |  | [Google Drive](xxxxxxxx) |
| RFBNet (VGG16) + MG | 320x320 | **34.6** | **52.0** | **36.8** |  | [Google Drive](xxxxxxxx) |
| RetinaNet (VGG16) + PAFPN | 320x320 | 33.9 | 51.9 | 35.7 | 175 | [Google Drive](xxxxxxxx) |
| RetinaNet (VGG16) + PAFPN + MG | 320x320 | **35.3** | **52.4** | **37.3** | 175 | [Google Drive](xxxxxxxx) |
| RetinaNet (VGG16) | 512x512 | 37.1 | 56.5 | 39.5 |  | [Google Drive](xxxxxxxx) |
| RetinaNet (VGG16) + MG | 512x512 | **38.2** | **56.6** | **41.0** |  | [Google Drive](xxxxxxxx) |
## Datasets
First download the VOC and COCO dataset, you may find the sripts in `data/scripts/` useful.
Then create a folder named `datasets` and link the downloaded datasets inside:
```Shell
$ mkdir datasets
$ ln -s /path_to_your_voc_dataset datasets/VOCdevkit
$ ln -s /path_to_your_coco_dataset datasets/coco2017
```
Finally prepare folders to save evaluation results:
```Shell
$ mkdir eval
$ mkdir eval/COCO
$ mkdir eval/VOC
```
## Training
For training with Mutual Guide:
```Shell
$ python3 main.py --version fssd --backbone vgg16 --dataset voc --size 320 --mutual_guide
                            retinanet       resnet18        coco       512
                            rfbnet
                            pafpn
```
**Remarks:**
- For training without Mutual Guide, just remove the '--mutual_guide';
- The default folder to save trained model is `weights/`.
## Evaluation
Every time you want to evaluate a trained network:
```Shell
$ python3 main.py --version fssd --backbone vgg16 --dataset voc --size 320 --trained_model path_to_saved_weights
                            retinanet       resnet18        coco       512
                            pafpn
                            rfbnet
```
It will directly print the mAP, AP50 and AP50 results on VOC2007 Test or COCO2017 Val.
## Citing Mutual Guide
Please cite our paper in your publications if it helps your research.
