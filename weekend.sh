python3 train.py --multi_level --backbone resnet34 --mutual_guide --dataset VOC --multi_anchor
python3 test.py --trained_model weights/VOC_retina_pafpn_resnet34_size320_anchor24.0_MG.pth --backbone resnet34 --multi_level --dataset VOC --eval_thresh 0.05 --multi_anchor 
python3 train.py --multi_level --backbone resnet34 --mutual_guide --dataset VOC
python3 test.py --trained_model weights/VOC_fcos_pafpn_resnet34_size320_anchor24.0_MG.pth --backbone resnet34 --multi_level --dataset VOC --eval_thresh 0.05 
