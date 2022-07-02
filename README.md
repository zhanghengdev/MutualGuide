<img align="center" src="https://github.com/zhangheng19931123/MutualGuide/blob/master/doc/mg.svg">

## Introduction
MutualGuide is a compact object detector specially designed for edge computing devices. Comparing to existing detectors, this repo contains two key features. 

Firstly, the Mutual Guidance mecanism assigns labels to the classification task based on the prediction on the localization task, and vice versa, alleviating the misalignment problem between both tasks; Secondly, the teacher-student prediction disagreements guides the knowledge transfer in a feature-based detection distillation framework, thereby reducing the performance gap between both models.

For more details, please refer to our [ACCV paper](https://openaccess.thecvf.com/content/ACCV2020/html/Zhang_Localize_to_Classify_and_Classify_to_Localize_Mutual_Guidance_in_ACCV_2020_paper.html) and [BMVC paper](https://www.bmvc2021.com/).

## Planning
- [ ] Train medium and large models.
- [ ] Add [SIOU](https://arxiv.org/abs/2205.12740) loss.
- [x] Add [CspDarknet](https://arxiv.org/abs/2107.08430) backbone.
- [x] Add [RepVGG](https://arxiv.org/abs/2101.03697) backbone.
- [x] Add [ShuffleNetV2](https://arxiv.org/abs/1807.11164) backbone.
- [x] Add [SwinTransformer](https://arxiv.org/abs/2103.14030) backbone.
- [ ] Add **TensorRT** transform code for inference acceleration.
- [x] Add **vis** function to plot detection results.
- [x] Add **custom dataset** training (annotations in `XML` format).


## Benchmark

|  **Backbone**   | **Size** | **AP<sup>val**<br>0.5:0.95 | **AP<sup>val**<br>0.5 | **AP<sup>val**<br>0.75 | **AP<sup>val**<br>small | **AP<sup>val**<br>medium | **AP<sup>val**<br>large | Params<br>(M) | FLOPs<br>(G) | **Speed**<br>(ms) |
| :-------------: | :------: | :------------------------: | :-------------------: | :--------------------: | :---------------------: | :----------------------: | :---------------------: | :-----------: | :----------: | :---------------: |
| cspdarknet-0.75 | 640x640  |            43.0            |         61.1          |          46.2          |          24.2           |           50.0           |          59.9           |     24.32     |    24.02     |    11.4(3060)     |
| cspdarknet-0.5  | 640x640  |            40.4            |         58.4          |          43.3          |          21.0           |           46.4           |          58.0           |     17.40     |    12.67     |     6.5(3060)     |
| shufflenet-1.5  | 640x640  |            35.7            |         53.9          |          37.9          |          16.5           |           41.3           |          53.5           |     2.55      |     2.65     |     5.6(3060)     |
| shufflenet-1.0  | 640x640  |            31.8            |         49.0          |          33.1          |          13.6           |           35.8           |          48.4           |     1.50      |     1.47     |     5.4(3060)     |


**Remarks:**

- The precision is measured on the COCO2017 Val dataset. 
- The inference runtime is measured by Pytorch framework (**without** TensorRT acceleration) on a GTX 3060 GPU, and the post-processing time (e.g., NMS) is **not** included (i.e., we measure the model inference time).
- To dowload from Baidu cloud, go to this [link](https://pan.baidu.com/s/16ZZUjL22XINUXpw8lNn76w) (password: `mugu`).



## Datasets

First download the COCO2017 dataset, you may find the sripts in `data/scripts/` helpful.
Then modify the parameter `self.root` in `data/coco.py` to the path of COCO dataset:

```python
self.root = os.path.join("/home/heng/Documents/Datasets/", "COCO/")
```
**Remarks:**

- For training on custom dataset, first modify the dataset path and categories `XML_CLASSES` in `data/xml_dataset.py`. Then apply `--dataset XML`.

## Training

For training with [Mutual Guide](https://openaccess.thecvf.com/content/ACCV2020/html/Zhang_Localize_to_Classify_and_Classify_to_Localize_Mutual_Guidance_in_ACCV_2020_paper.html):
```Shell
$ python3 train.py --neck ssd --backbone vgg16    --dataset COCO
                          fpn            resnet34           VOC
                          pafpn          repvgg-A2          XML
                                         cspdarknet-0.75
                                         shufflenet-1.0
                                         swin-T
```

For knowledge distillation using [PDF-Distil](https://www.bmvc2021.com/):
```Shell
$ python3 distil.py --neck ssd --backbone vgg11    --dataset COCO  --kd pdf
                           fpn            resnet18           VOC
                           pafpn          repvgg-A1          XML
                                          cspdarknet-0.5
                                          shufflenet-0.5
```

**Remarks:**

- For training without MutualGuide, just use the `--mutual_guide False`;
- For training on custom dataset, convert your annotations into XML format and use the parameter `--dataset XML`. An example is given in `datasets/XML/`;
- For knowledge distillation with traditional MSE loss, just use parameter `--kd mse`;
- The default folder to save trained model is `weights/`.

## Evaluation

Every time you want to evaluate a trained network:
```Shell
$ python3 test.py --neck ssd --backbone vgg11    --dataset COCO --trained_model path_to_saved_weights --vis
                         fpn            resnet18           VOC
                         pafpn          repvgg-A1          XML
                                        cspdarknet-0.5
                                        shufflenet-0.5
```

**Remarks:**

- It will directly print the mAP, AP50 and AP50 results on COCO2017 Val;
- Add parameter `--vis` to draw detection results. They will be saved in `draw/VOC/` or `draw/COCO/` or `draw/XML/`;

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

This project contains pieces of code from the following projects: [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch), [rfbnet](https://github.com/ruinmessi/RFBNet), [mmdetection](https://github.com/open-mmlab/mmdetection) and [yolox](https://github.com/Megvii-BaseDetection/YOLOX).
