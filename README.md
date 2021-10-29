<img align="center" src="https://github.com/zhangheng19931123/MutualGuide/blob/master/doc/mg.svg">

# Introduction
MutualGuide is a compact object detector specially designed for embedded devices. Comparing to existing detectors, this repo contains two key features. 

Firstly, the Mutual Guidance mecanism assigns labels to the classification task based on the prediction on the localization task, and vice versa, alleviating the misalignment problem between both tasks; Secondly, the teacher-student prediction disagreements guides the knowledge transfer in a feature-based detection distillation framework, thereby reducing the performance gap between both models.

For more details, please refer to our [ACCV paper](https://openaccess.thecvf.com/content/ACCV2020/html/Zhang_Localize_to_Classify_and_Classify_to_Localize_Mutual_Guidance_in_ACCV_2020_paper.html) and [BMVC paper](https://www.bmvc2021.com/).

# Planning
- [x] Add [RepVGG](https://arxiv.org/abs/2101.03697) backbone.
- [x] Add [ShuffleNetV2](https://arxiv.org/abs/1807.11164) backbone.
- [x] Add **TensorRT** transform code for inference acceleration.
- [x] Add **draw** function to plot detection results.
- [x] Add **custom dataset** training (annotations in `XML` format).
- [ ] Add [Transformer](https://arxiv.org/abs/2102.12122) backbone.
- [ ] Add [BiFPN](https://arxiv.org/abs/1911.09070) neck.


# Benchmark

- Without knowledge distillation:


| **Backbone** | **Resolution** | **AP<sup>val**<br>0.5:0.95 | **AP<sup>val**<br>0.5 | **AP<sup>val**<br>0.75 | **AP<sup>val**<br>small | **AP<sup>val**<br>medium | **AP<sup>val**<br>large | **Speed V100**<br>(ms) | **Weights** |
|:------------:|:--------------:|:--------------------------:|:---------------------:|:----------------------:|:-----------------------:|:------------------------:|:-----------------------:|:----------------------:|:-----------:|
| ShuffleNet-1.0 | 320x320      | 29.3 | 45.4 | 30.5 | 12.0 | 33.2 | 43.9 | 7.0 | [Google](https://drive.google.com/file/d/1t5JuFlb6GQL2nFJkxeh4gD_XUC0uwNay/view?usp=sharing) |
| ShuffleNet-1.0 | 512x512      | 33.5 | 51.0 | 35.8 | 17.7 | 38.0 | 44.3 | 9.0 | [Google](https://drive.google.com/file/d/1USgfIl82mR_6-AH2bCe3k7xvx_ioxMk7/view?usp=sharing) |
| ResNet-34      | 512x512      | 44.1 | 62.3 | 47.6 | 26.5 | 50.2 | 58.3 | 6.9 | [Google](https://drive.google.com/file/d/1DRQ0FHo2Wfn8u4xoN62FLA_Mrjskwj5b/view?usp=sharing) |
| ResNet-18      | 512x512      | 42.0 | 60.0 | 45.3 | 25.4 | 47.1 | 56.0 | 4.4 | [Google](https://drive.google.com/file/d/1wZ_tO55nYrzb3X12CrhHby-lw6mAv1M_/view?usp=sharing) |
| RepVGG-A2      | 512x512      | 44.2 | 62.5 | 47.5 | 27.2 | 50.3 | 57.2 | 5.3 | [Google](https://drive.google.com/file/d/1fHbSDRvoDB4h-Dh2cm9zoiDYnluGw6Kh/view?usp=sharing) |
| RepVGG-A1      | 512x512      | 43.1 | 61.3 | 46.6 | 26.6 | 49.3 | 55.9 | 4.4 | [Google](https://drive.google.com/file/d/1iLppaAs7sLr9TXkD3oqmNlG1MVufYRMS/view?usp=sharing) |


- With knowledge distillation:

| **Backbone** | **Resolution** | **AP<sup>val**<br>0.5:0.95 | **AP<sup>val**<br>0.5 | **AP<sup>val**<br>0.75 | **AP<sup>val**<br>small | **AP<sup>val**<br>medium | **AP<sup>val**<br>large | **Speed V100**<br>(ms) | **Weights** |
|:------------:|:--------------:|:--------------------------:|:---------------------:|:----------------------:|:-----------------------:|:------------------------:|:-----------------------:|:----------------------:|:-----------:|
| ShuffleNet-0.5 | 320x320      |  |  |  |  |  |  |  | running |
| ShuffleNet-0.5 | 512x512      |  |  |  |  |  |  |  | running |
| ResNet-18      | 512x512      | 42.9 | 60.7 | 46.2 | 25.4 | 48.8 | 57.2 | 4.4 | [Google](https://drive.google.com/file/d/1bilD6E3tdjJI3ZD4vZ6nUU_eSsieAfm5/view?usp=sharing) |
| RepVGG-A1      | 512x512      | 44.0 | 62.1 | 47.3 | 27.6 | 49.9 | 57.9 | 4.4 | [Google](https://drive.google.com/file/d/1hsb_rxArYYCHK7_RJ37k0N_1uZRu2WmG/view?usp=sharing) |

**Remarks:**

- The precision is measured on the COCO2017 Val dataset. 
- The inference runtime is measured by Pytorch framework (**without** TensorRT acceleration) on a Tesla V100 GPU, and the post-processing time (e.g., NMS) is **not** included (i.e., we measure the model inference time).
- To dowload from Baidu cloud, go to this [link](https://pan.baidu.com/s/1G9KbNmbwteiE4a2yb-JiXg) (password: `dvz7`).

# Datasets

First download the VOC and COCO dataset, you may find the sripts in `data/scripts/` helpful.
Then create a folder named `datasets` and link the downloaded datasets inside:

```Shell
$ mkdir datasets
$ ln -s /path_to_your_voc_dataset datasets/VOCdevkit
$ ln -s /path_to_your_coco_dataset datasets/coco2017
```
**Remarks:**

- For training on custom dataset, first modify the dataset path `XMLroot` and categories `XML_CLASSES` in `data/xml_dataset.py`. Then apply `--dataset XML`.

# Training

For training with [Mutual Guide](https://openaccess.thecvf.com/content/ACCV2020/html/Zhang_Localize_to_Classify_and_Classify_to_Localize_Mutual_Guidance_in_ACCV_2020_paper.html):
```Shell
$ python3 train.py --neck ssd --backbone vgg16    --dataset VOC --size 320 --multi_level --multi_anchor --mutual_guide --pretrained
                          fpn            resnet34           COCO       512
                          pafpn          repvgg-A2          XML
                                         shufflenet-1.0
```

For knowledge distillation using [PDF-Distil](https://www.bmvc2021.com/):
```Shell
$ python3 distil.py --neck ssd --backbone vgg11    --dataset VOC --size 320 --multi_level --multi_anchor --mutual_guide --pretrained --kd pdf
                           fpn            resnet18           COCO       512
                           pafpn          repvgg-A1          XML
                                          shufflenet-0.5
```

**Remarks:**

- For training without MutualGuide, just remove the `--mutual_guide`;
- For training on custom dataset, convert your annotations into XML format and use the parameter `--dataset XML`. An example is given in `datasets/XML/`;
- For knowledge distillation with traditional MSE loss, just use parameter `--kd mse`;
- The default folder to save trained model is `weights/`.

# Evaluation

Every time you want to evaluate a trained network:
```Shell
$ python3 test.py --neck ssd --backbone vgg11    --dataset VOC --size 320 --trained_model path_to_saved_weights --multi_level --multi_anchor --pretrained --draw
                         fpn            resnet18           COCO       512
                         pafpn          repvgg-A1          XML
                                        shufflenet-0.5
```

**Remarks:**

- It will directly print the mAP, AP50 and AP50 results on VOC2007 Test or COCO2017 Val;
- Add parameter `--draw` to draw detection results. They will be saved in `draw/VOC/` or `draw/COCO/` or `draw/XML/`;
- Add `--trt` to activate TensorRT acceleration.

# Citing us

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

# Acknowledgement

This project contains pieces of code from the following projects: [mmdetection](https://github.com/open-mmlab/mmdetection), [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch), [rfbnet](https://github.com/ruinmessi/RFBNet) and [yolox](https://github.com/Megvii-BaseDetection/YOLOX).