# Localize to Classify and Classify to Localize: Mutual Guidance in Object Detection
By Heng Zhang, Elisa FROMONT, Sébastien LEFEVRE, Bruno AVIGNON

## Planning
- [x] Add **draw** function to plot detection results.
- [x] Add [RepVGG](https://arxiv.org/abs/2101.03697) backbone.
- [x] Add [ShuffleNetV2](https://arxiv.org/abs/1807.11164) backbone.
- [ ] Add **TensorRT** transform code for inference acceleration.
- [ ] Add **custom dataset** training.
## Introduction
Most deep learning object detectors are based on the anchor mechanism and resort to the Intersection over Union (IoU) between predefined anchor boxes and ground truth boxes to evaluate the matching quality between anchors and objects. In this paper, we question this use of IoU and propose a new anchor matching criterion guided, during the training phase, by the optimization of both the localization and the classification tasks: the predictions related to one task are used to dynamically assign sample anchors and improve the model on the other task, and vice versa. This is the Pytorch implementation of Mutual Guidance detectors. For more details, please refer to our [ACCV paper](https://openaccess.thecvf.com/content/ACCV2020/html/Zhang_Localize_to_Classify_and_Classify_to_Localize_Mutual_Guidance_in_ACCV_2020_paper.html).
<img align="center" src="https://github.com/zhangheng19931123/MutualGuide/blob/master/doc/compare.png">
&nbsp;
&nbsp;
## Experimental results
### VOC2007 Test
| **Backbone** | **Neck** | MG | **Resolution** | **mAP** | **AP50** | **AP75** | **Model** |
|:-----:|:-----:|:-----:|:-----:|:-------:|:-------:|:-------:|:-------:|
| ShuffleNetV2 | FPN | | 320x320 | 51.7 | 55.5 | 77.0 | [Download](https://drive.google.com/file/d/16uX2sQo3tOY9OmukUMDWHlmW68cqixBc/view?usp=sharing) |
| ShuffleNetV2 | FPN | ✓ | 320x320 | **52.8** | **56.6** | **77.3** | [Download](https://drive.google.com/file/d/1KK0qHQWuBmMPmAHwVw0G0wF4PbAZ3GR0/view?usp=sharing) |
| VGG16 | SSD | | 320x320 | 54.1 | 80.1 | 58.3 | [Download](https://drive.google.com/file/d/1fCfa3E9rama3SeTD5Tt7CyWapzxBAm0g/view?usp=sharing) |
| VGG16 | SSD | ✓ | 320x320 | **56.2** | **80.4** | **61.4** | [Download](https://drive.google.com/file/d/1jQLuU3yNy-09eoSfRO7p6k_6YnxsX8B1/view?usp=sharing) |
| VGG16 | FPN | | 320x320 | 55.2 | 80.2 | 59.6 | [Download](https://drive.google.com/file/d/1Cv2PNB2VnisEDZa87j_ToXnZtWtQ9s11/view?usp=sharing) |
| VGG16 | FPN | ✓ | 320x320 | **57.7** | **81.1** | **62.9** | [Download](https://drive.google.com/file/d/1clZS_Q8n6ZH7Vtaw5KZWXoL9h9X_ZZS8/view?usp=sharing) |
| VGG16 | RFB | | 320x320 | 55.6 | 80.9 | 59.6 |                                                              |
| VGG16 | RFB | ✓ | 320x320 | **57.9** | **81.5** | **62.6** |                                                              |
| VGG16 | PAFPN | | 320x320 | 58.1 | 81.7 | 63.3 | |
| VGG16 | PAFPN | ✓ | 320x320 | **59.5** | **82.2** | **64.1** | [Download](https://drive.google.com/file/d/1su13LbkbhoFjAk0xp7NxqzSjOqB01773/view?usp=sharing) |
| REGVGG A2 | PAFPN | | 320x320 | 60.0 | 83.1 | 65.1 | [Download](https://drive.google.com/file/d/15SEoXNeRr4Mv-ZEPdjC2psFh0WScjvzr/view?usp=sharing) |
| REGVGG A2 | PAFPN | ✓ | 320x320 | **61.8** | **83.7** | **68.1** | [Download](https://drive.google.com/file/d/1kv2439v33WvfWy592vnqWWAiV7IVu0k6/view?usp=sharing) |
### COCO2017 Val
| **Backbone** | **Neck** | MG | **Resolution** | **mAP** | **AP50** | **AP75** | **FPS** | **Model** |
|:-----:|:-----:|:-----:|:-----:|:-------:|:-------:|:-------:|:-------:|:-------:|
| VGG16 | SSD | | 320x320 | 31.1 | 48.9 | 32.7 | 365 | [Download](https://drive.google.com/file/d/1zWRDl9UXvUurfGaj2QnHJDRZJFH1w_1B/view?usp=sharing) |
| VGG16 | SSD | ✓ | 320x320 | **32.0** | **49.3** | **33.9** | 365 | [Download](https://drive.google.com/file/d/1Ec658sd8z7jHW9Y2dQlRhWdmgKW_JJdp/view?usp=sharing) |
| VGG16 | FPN | | 320x320 | 32.3 | 50.3 | 34.0 | 270 | [Download](https://drive.google.com/file/d/1DUMXX6_7ca1RBg1kNqfjpXaaPkKL8DBa/view?usp=sharing) |
| VGG16 | FPN | ✓ | 320x320 | **33.6** | **50.8** | **35.7** | 270 | [Download](https://drive.google.com/file/d/1Wah6c5PbUndv7lx_lPvwKd8XC9dyCJCe/view?usp=sharing) |
| VGG16 | RFB | | 320x320 | 33.4 | 51.6 | 35.1 | 115 |                                                              |
| VGG16 | RFB | ✓ | 320x320 | **34.6** | **52.0** | **36.8** | 115 |                                                              |
| VGG16 | PAFPN | | 320x320 | 33.9 | 51.9 | 35.7 | 220 | [Download](https://drive.google.com/file/d/1qHHtEj0G81ivBgK81LzEQ2Z4OJVGlf9C/view?usp=sharing) |
| VGG16 | PAFPN | ✓ | 320x320 | **35.3** | **52.4** | **37.3** | 220 | [Download](https://drive.google.com/file/d/1nmhlMGGRCNRWMhmotsi8YEujWPhUyiFU/view?usp=sharing) |
| REGVGG A2 | PAFPN | | 320x320 | | | | | |
| REGVGG A2 | PAFPN | ✓ | 320x320 | **36.2** | **53.6** | **38.5** | | [Download](https://drive.google.com/file/d/1zuopiHZyq8JFMvEFgvB4VuQlyh5Tv4ux/view?usp=sharing) |
| VGG16 | FPN | | 512x512 | 37.1 | 56.5 | 39.5 | 250 | |
| VGG16 | FPN | ✓ | 512x512 | **38.2** | **56.6** | **41.0** | 250 | |
| VGG16 | PAFPN | | 512x512 | 38.5 | 57.6 | 41.0 | 195 | [Download](https://drive.google.com/file/d/18zmuc9GCjVyCNb23Vv91LCSHjFCYLuhY/view?usp=sharing) |
| VGG16 | PAFPN | ✓ | 512x512 | **39.4** | **57.5** | **42.3** | 195 | [Download](https://drive.google.com/file/d/1Px9DLuGWIICsEw4tKYGCefLuXf09Xqg1/view?usp=sharing) |

**Remarks:**

<<<<<<< HEAD
- The inference Frame-Per-Second is measured by Pytorch 1.2.0 framework on a Tesla V100 GPU, the post-processing time (nms) time is not included.
=======
- The inference FPS is measure by Pytorch 1.20 framework on a Tesla V100 GPU, the post-processing time (nms) time is not included.
>>>>>>> f28a48da2a4222460f620bbbe1638949fc1a8f23

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
