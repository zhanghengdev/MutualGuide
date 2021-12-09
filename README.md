<img align="center" src="https://github.com/zhangheng19931123/MutualGuide/blob/master/doc/mg.svg">

## Introduction
MutualGuide is a compact object detector specially designed for embedded devices. Comparing to existing detectors, this repo contains two key features. 

Firstly, the Mutual Guidance mecanism assigns labels to the classification task based on the prediction on the localization task, and vice versa, alleviating the misalignment problem between both tasks; Secondly, the teacher-student prediction disagreements guides the knowledge transfer in a feature-based detection distillation framework, thereby reducing the performance gap between both models.

For more details, please refer to our [ACCV paper](https://openaccess.thecvf.com/content/ACCV2020/html/Zhang_Localize_to_Classify_and_Classify_to_Localize_Mutual_Guidance_in_ACCV_2020_paper.html) and [BMVC paper](https://www.bmvc2021.com/).

## Planning
- [x] Add [RepVGG](https://arxiv.org/abs/2101.03697) backbone.
- [x] Add [RegNet](https://arxiv.org/abs/2003.13678) backbone.
- [x] Add [ShuffleNetV2](https://arxiv.org/abs/1807.11164) backbone.
- [x] Add [SwinTransformer](https://arxiv.org/abs/2103.14030) backbone.
- [x] Add **TensorRT** transform code for inference acceleration.
- [x] Add **draw** function to plot detection results.
- [x] Add **custom dataset** training (annotations in `XML` format).


## Benchmark

- Without knowledge distillation:


| **Backbone** | **Resolution** | **AP<sup>val**<br>0.5:0.95 | **AP<sup>val**<br>0.5 | **AP<sup>val**<br>0.75 | **AP<sup>val**<br>small | **AP<sup>val**<br>medium | **AP<sup>val**<br>large | **Speed V100**<br>(ms) |
|:------------:|:--------------:|:--------------------------:|:---------------------:|:----------------------:|:-----------------------:|:------------------------:|:-----------------------:|:----------------------:|
| ShuffleNet-1.0(new) | 320x320      | 30.5 | 47.5 | 31.9 | 12.5 | 34.2 | 46.0 | 8 |
| ResNet-18(new) | 512x512      | 43.0 | 61.8 | 46.3 | 26.7 | 47.9 | 56.4 | 12 |
| ResNet-34(new) | 512x512      | 45.0 | 63.7 | 48.3 | 28.5 | 50.1 | 59.7 | 16 |
| RepVGG-A2(new) | 512x512      | 45.1 | 64.0 | 48.6 | 27.7 | 50.4 | 59.0 | 18 |



- With knowledge distillation:

| **Backbone** | **Resolution** | **AP<sup>val**<br>0.5:0.95 | **AP<sup>val**<br>0.5 | **AP<sup>val**<br>0.75 | **AP<sup>val**<br>small | **AP<sup>val**<br>medium | **AP<sup>val**<br>large | **Speed V100**<br>(ms) |
|:------------:|:--------------:|:--------------------------:|:---------------------:|:----------------------:|:-----------------------:|:------------------------:|:-----------------------:|:----------------------:|
| ResNet-18      | 512x512      | 42.9 | 60.7 | 46.2 | 25.4 | 48.8 | 57.2 | 12 |
| RepVGG-A1      | 512x512      | 44.0 | 62.1 | 47.3 | 27.6 | 49.9 | 57.9 | 12 |

**Remarks:**

- The precision is measured on the COCO2017 Val dataset. 
- **Sorry we forgot to synchronize the GPU time during the last run, so the runtime measurements were incorrect. The updated results should be correct.**
- The inference runtime is measured by Pytorch framework (**without** TensorRT acceleration) on a Tesla V100 GPU, and the post-processing time (e.g., NMS) is **not** included (i.e., we measure the model inference time).
- To dowload from Google cloud, go to this [link](https://drive.google.com/drive/folders/1ZNfhY1Znvg7BBZV6qCTM3DCQLjTt7mgM?usp=sharing).
- To dowload from Baidu cloud, go to this [link](https://pan.baidu.com/s/1G9KbNmbwteiE4a2yb-JiXg) (password: `dvz7`).



## Datasets

First download the VOC and COCO dataset, you may find the sripts in `data/scripts/` helpful.
Then create a folder named `datasets` and link the downloaded datasets inside:

```Shell
$ mkdir datasets
$ ln -s /path_to_your_voc_dataset datasets/VOCdevkit
$ ln -s /path_to_your_coco_dataset datasets/coco2017
```
**Remarks:**

- For training on custom dataset, first modify the dataset path `XMLroot` and categories `XML_CLASSES` in `data/xml_dataset.py`. Then apply `--dataset XML`.

## Training

For training with [Mutual Guide](https://openaccess.thecvf.com/content/ACCV2020/html/Zhang_Localize_to_Classify_and_Classify_to_Localize_Mutual_Guidance_in_ACCV_2020_paper.html):
```Shell
$ python3 train.py --neck ssd --backbone vgg16    --dataset VOC --size 320 --multi_anchor --mutual_guide
                          fpn            resnet34           COCO       512
                          pafpn          repvgg-A2          XML
                                         regnet800
                                         shufflenet-1.0
                                         swin-T
```

For knowledge distillation using [PDF-Distil](https://www.bmvc2021.com/):
```Shell
$ python3 distil.py --neck ssd --backbone vgg11    --dataset VOC --size 320 --multi_anchor --mutual_guide --kd pdf
                           fpn            resnet18           COCO       512
                           pafpn          repvgg-A1          XML
                                          regnet400
                                          shufflenet-0.5
```

**Remarks:**

- For training without MutualGuide, just remove the `--mutual_guide`;
- For training on custom dataset, convert your annotations into XML format and use the parameter `--dataset XML`. An example is given in `datasets/XML/`;
- For knowledge distillation with traditional MSE loss, just use parameter `--kd mse`;
- The default folder to save trained model is `weights/`.

## Evaluation

Every time you want to evaluate a trained network:
```Shell
$ python3 test.py --neck ssd --backbone vgg11    --dataset VOC --size 320 --trained_model path_to_saved_weights --multi_anchor --draw
                         fpn            resnet18           COCO       512
                         pafpn          repvgg-A1          XML
                                        shufflenet-0.5
```

**Remarks:**

- It will directly print the mAP, AP50 and AP50 results on VOC2007 Test or COCO2017 Val;
- Add parameter `--draw` to draw detection results. They will be saved in `draw/VOC/` or `draw/COCO/` or `draw/XML/`;
- Add `--trt` to activate TensorRT acceleration.

## Citing us

Please cite our papers in your publications if they help your research:

    @InProceedings{Zhang_2020_ACCV,
        author    = {Zhang, Heng and Fromont, Elisa and Lefevre, Sebastien and Avignon, Bruno},
        title     = {Localize to Classify and Classify to Localize: Mutual Guidance in Object Detection},
        booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
        month     = {November},
        year      = {2020}
    }

    @InProceedings{Zhang_2021_BMVC,
        author    = {Zhang, Heng and Fromont, Elisa and Lefevre, Sebastien and Avignon, Bruno},
        title     = {PDF-Distil: including Prediction Disagreements in Feature-based Distillation for object detection},
        booktitle = {Proceedings of the British Machine Vision Conference (BMVC)},
        month     = {November},
        year      = {2021}
    }

## Acknowledgement

This project contains pieces of code from the following projects: [mmdetection](https://github.com/open-mmlab/mmdetection), [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch), [rfbnet](https://github.com/ruinmessi/RFBNet) and [yolox](https://github.com/Megvii-BaseDetection/YOLOX).
