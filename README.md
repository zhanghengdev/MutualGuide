# Localize to Classify and Classify to Localize: Mutual Guidance in Object Detection
By Heng Zhang, Elisa FROMONT, Sébastien LEFEVRE, Bruno AVIGNON

## Planning
- [x] Add **draw** function to plot detection results.
- [x] Add [RepVGG](https://arxiv.org/abs/2101.03697) backbone.
- [x] Add [ShuffleNetV2](https://arxiv.org/abs/1807.11164) backbone.
- [ ] Add **TensorRT** transform code for inference acceleration.
- [ ] Add custom dataset training.
## Introduction
Most deep learning object detectors are based on the anchor mechanism and resort to the Intersection over Union (IoU) between predefined anchor boxes and ground truth boxes to evaluate the matching quality between anchors and objects. In this paper, we question this use of IoU and propose a new anchor matching criterion guided, during the training phase, by the optimization of both the localization and the classification tasks: the predictions related to one task are used to dynamically assign sample anchors and improve the model on the other task, and vice versa. This is the Pytorch implementation of Mutual Guidance detectors. For more details, please refer to our [ACCV paper](https://openaccess.thecvf.com/content/ACCV2020/html/Zhang_Localize_to_Classify_and_Classify_to_Localize_Mutual_Guidance_in_ACCV_2020_paper.html).
<img align="center" src="https://github.com/zhangheng19931123/MutualGuide/blob/master/doc/compare.png">
&nbsp;
&nbsp;
## Experimental results
### VOC2007 Test
| **Backbone** | **Neck** | **MutualGuide** | **Resolution** | **mAP** | **AP50** | **AP75** | **Trained model** |
|:-----:|:-----:|:-----:|:-----:|:-------:|:-------:|:-------:|:-------:|
| ShuffleNetV2 | FPN | | 320x320 | 51.7 | 55.5 | 77.0 | [Google Drive](https://drive.google.com/file/d/1IOTIyS9hZY7-g3RP2p3OkcVmtGdmWJIc/view?usp=sharing) |
| ShuffleNetV2 | FPN | ✓ | 320x320 | **52.8** | **56.6** | **77.3** | [Google Drive](https://drive.google.com/file/d/1bFVrBPPQDymstgjwlss3AUK6WrN-iszr/view?usp=sharing) |
| VGG16 | SSD | | 320x320 | 54.1 | 80.1 | 58.3 | [Google Drive](https://drive.google.com/file/d/1IOTIyS9hZY7-g3RP2p3OkcVmtGdmWJIc/view?usp=sharing) |
| VGG16 | SSD | ✓ | 320x320 | **56.2** | **80.4** | **61.4** | [Google Drive](https://drive.google.com/file/d/1bFVrBPPQDymstgjwlss3AUK6WrN-iszr/view?usp=sharing) |
| VGG16 | FPN | | 320x320 | 55.2 | 80.2 | 59.6 | [Google Drive](https://drive.google.com/file/d/1c3bGwtFRD9GvxdyqDq1jknlZvRPpxjUi/view?usp=sharing) |
| VGG16 | FPN | ✓ | 320x320 | **57.7** | **81.1** | **62.9** | [Google Drive](https://drive.google.com/file/d/1vviR8H6xHfvY5Q4DDmZQ-lWLjEpPqrLr/view?usp=sharing) |
| VGG16 | RFB | | 320x320 | 55.6 | 80.9 | 59.6 | [Google Drive](https://drive.google.com/file/d/1MOM4pTh4TQ1l3ADFqT-BLL9RoSJK33v3/view?usp=sharing) |
| VGG16 | RFB | ✓ | 320x320 | **57.9** | **81.5** | **62.6** | [Google Drive](https://drive.google.com/file/d/1Nb6NPa4aNfz49NhGeTTfgW2vR-UVUzIz/view?usp=sharing) |
| VGG16 | PAFPN | | 320x320 | 58.1 | 81.7 | 63.3 | |
| VGG16 | PAFPN | ✓ | 320x320 | **59.5** | **82.2** | **64.1** | [Google Drive](https://drive.google.com/file/d/1CQtVaMJctxNgmbNp9N-R8KH0gCOQhhPP/view?usp=sharing) |
| REGVGG A2 | PAFPN | | 320x320 | | | | |
| REGVGG A2 | PAFPN | ✓ | 320x320 | **61.8** | **83.7** | **68.1** | |
### COCO2017 Val
| **Backbone** | **Neck** | **MutualGuide** | **Resolution** | **mAP** | **AP50** | **AP75** | **FPS** (V100) | **Trained model** |
|:-----:|:-----:|:-----:|:-----:|:-------:|:-------:|:-------:|:-------:|:-------:|
| VGG16 | SSD | | 320x320 | 31.1 | 48.9 | 32.7 | 365 | [Google Drive](https://drive.google.com/file/d/1i6frTMPX1Bi-OpTZEyRYsPQTnAPyEplb/view?usp=sharing) |
| VGG16 | SSD | ✓ | 320x320 | **32.0** | **49.3** | **33.9** | 365 | [Google Drive](https://drive.google.com/file/d/1bSOTSRMPkc6WDiL8AdKtvaZKbxtbEGFp/view?usp=sharing) |
| VGG16 | FPN | | 320x320 | 32.3 | 50.3 | 34.0 | 270 | [Google Drive](https://drive.google.com/file/d/1Gx0I1sTqgFmUtQln0NPrT4_k9x2VCIUM/view?usp=sharing) |
| VGG16 | FPN | ✓ | 320x320 | **33.6** | **50.8** | **35.7** | 270 | [Google Drive](https://drive.google.com/file/d/12Af5Pz-Zsl8oww7NjDmjWjvT0br6zFTn/view?usp=sharing) |
| VGG16 | RFB | | 320x320 | 33.4 | 51.6 | 35.1 | 115 | [Google Drive](https://drive.google.com/file/d/1KnNcYBCKA53MJ70rpRoMk-Q247FVTH4K/view?usp=sharing) |
| VGG16 | FPN | ✓ | 320x320 | **34.6** | **52.0** | **36.8** | 115 | [Google Drive](https://drive.google.com/file/d/1rZ_hKWLGASDlRKNEEdAA5V0vb6st5Sqk/view?usp=sharing) |
| VGG16 | PAFPN | | 320x320 | 33.9 | 51.9 | 35.7 | 220 | [Google Drive](https://drive.google.com/file/d/13zBaiJ7LvlvPBogKB069OPhuV6JLKZzg/view?usp=sharing) |
| VGG16 | PAFPN | ✓ | 320x320 | **35.3** | **52.4** | **37.3** | 220 | [Google Drive](https://drive.google.com/file/d/1IC18t7wnnm1Wk8q9UpkPzGy2-g68_uyY/view?usp=sharing) |
| REGVGG A2 | PAFPN | | 320x320 | | | | | |
| REGVGG A2 | PAFPN | ✓ | 320x320 | | | | | |
| VGG16 | FPN | | 512x512 | 37.1 | 56.5 | 39.5 | 250 | |
| VGG16 | FPN | ✓ | 512x512 | **38.2** | **56.6** | **41.0** | 250 | |
| VGG16 | PAFPN | | 512x512 | 38.5 | 57.6 | 41.0 | 195 | [Google Drive](https://drive.google.com/file/d/1yBllIGiix3FF5njQzV39Uhbz4dSAJrgO/view?usp=sharing) |
| VGG16 | PAFPN | ✓ | 512x512 | **39.4** | **57.5** | **42.3** | 195 | [Google Drive](https://drive.google.com/file/d/1kj0auR9w2zZeSSffFuS-MS0pX07Ro61T/view?usp=sharing) |
## Datasets
First download the VOC and COCO dataset, you may find the sripts in `data/scripts/` helpful.
Then create a folder named `datasets` and link the downloaded datasets inside:
```Shell
$ mkdir datasets
$ ln -s /path_to_your_voc_dataset datasets/VOCdevkit
$ ln -s /path_to_your_coco_dataset datasets/coco2017
```
## Training
For training with Mutual Guide:
```Shell
$ python3 main.py --neck ssd --backbone vgg16 --dataset voc --size 320 --mutual_guide
                         fpn            resnet18        coco       512
                         pafpn          repvgg
                                        shufflenet
```
**Remarks:**

- For training without MutualGuide, just remove the '--mutual_guide';
- The default folder to save trained model is `weights/`.
## Evaluation
Every time you want to evaluate a trained network:
```Shell
$ python3 main.py --neck ssd --backbone vgg16 --dataset voc --size 320 --trained_model path_to_saved_weights --draw
                         fpn            resnet18        coco       512
                         pafpn          repvgg
                                        shufflenet
```
**Remarks:**
- It will directly print the mAP, AP50 and AP50 results on VOC2007 Test or COCO2017 Val.
- Add parameter `--draw` to draw detection results. They will be saved in `draw/VOC/` or  `draw/COCO/`.
## Citing Mutual Guidance
Please cite our paper in your publications if it helps your research:

    @InProceedings{Zhang_2020_ACCV,
        author    = {Zhang, Heng and Fromont, Elisa and Lefevre, Sebastien and Avignon, Bruno},
        title     = {Localize to Classify and Classify to Localize: Mutual Guidance in Object Detection},
        booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
        month     = {November},
        year      = {2020}
    }
