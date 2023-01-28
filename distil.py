#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from models.detector import Detector
from data import (
    COCODetection,
    VOCDetection,
    XMLDetection,
    DataPrefetcher,
    detection_collate,
)
from utils import (
    Timer,
    ModelEMA,
    HintLoss,
    MultiBoxLoss,
    get_prior_box,
    tencent_trick,
    adjust_learning_rate,
)
import torch.nn.functional as F

cudnn.benchmark = True

### For Reproducibility ###
# import random
# SEED = 0
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
# torch.cuda.empty_cache()
# cudnn.benchmark = False
# cudnn.deterministic = True
# cudnn.enabled = True
### For Reproducibility ###

parser = argparse.ArgumentParser(description="PDF Distilation")
parser.add_argument("--config", type=str)
parser.add_argument("--dataset", default="COCO", type=str)
parser.add_argument("--resume_ckpt", default=None, type=str)
args = parser.parse_args()


def save_model(
    model: nn.Module,
    iteration: int,
    suffix: str,
) -> None:
    os.makedirs(args.save_folder, exist_ok=True)
    save_path = os.path.join(
        args.save_folder,
        "{}_{}_{}_kd{}_size{}_anchor{}_{}_{}.pth".format(
            args.dataset,
            args.neck,
            args.backbone,
            args.kd,
            args.image_size,
            args.anchor_size,
            "MG" if args.mutual_guide else "Retina",
            suffix,
        ),
    )
    tosave = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    print("Saving to {}".format(save_path))
    torch.save(tosave, save_path)
    return


if __name__ == "__main__":

    print("Extracting params...")
    with open(args.config, "r") as f:
        configs = yaml.safe_load(f)
        for config in configs.values():
            for key, value in config.items():
                setattr(args, key, value)
    print(args)
    
    print("Loading dataset...")
    if args.dataset == "COCO":
        train_sets = [("2017", "train")]
        dataset = COCODetection(train_sets, args.image_size)
    elif args.dataset == "VOC":
        train_sets = [("2007", "trainval"), ("2012", "trainval")]
        dataset = VOCDetection(train_sets, args.image_size)
    elif args.dataset == "XML":
        dataset = XMLDetection("train", args.image_size)
    else:
        raise NotImplementedError("ERROR: Unkown dataset {}".format(args.dataset))
    epoch_size = len(dataset) // args.batch_size
    end_iter = epoch_size * args.max_epoch

    print("Loading network...")
    model = Detector(
        args.image_size,
        dataset.num_classes,
        args.backbone,
        args.neck,
        mode="student",
    ).cuda()
    ema_model = ModelEMA(model)
    optimizer = optim.SGD(
        tencent_trick(model),
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=True,
    )
    scaler = torch.cuda.amp.GradScaler()

    if args.resume_ckpt:
        print("Resuming checkpoint from", args.resume_ckpt)
        state_dict = torch.load(args.resume_ckpt)
        model.load_state_dict(state_dict["model"], strict=True)
        optimizer.load_state_dict(state_dict["optimizer"])
        start_iter = state_dict["iteration"]
    else:
        start_iter = 0

    print("Preparing teacher...")
    if args.backbone == "resnet18":
        backbone = "resnet34"
    elif args.backbone == "repvgg-A0":
        backbone = "repvgg-A1"
    elif args.backbone == "yolox-s":
        backbone = "yolox-m"
    elif args.backbone == "shufflenet-0.5":
        backbone = "shufflenet-1.0"
    else:
        raise NotImplementedError
    teacher = Detector(
        args.image_size,
        dataset.num_classes,
        backbone,
        args.neck,
        mode="teacher",
    ).cuda()
    teacher_ckpt = os.path.join(
        args.save_folder,
        "{}_{}_{}_size{}_anchor{}_{}_Final.pth".format(
            args.dataset,
            args.neck,
            backbone,
            args.image_size,
            args.anchor_size,
            "MG" if args.mutual_guide else "Retina",
        ),
    )
    print("Loading teacher weights from", teacher_ckpt)
    state_dict = torch.load(teacher_ckpt)
    teacher.load_state_dict(state_dict["model"], strict=False)
    teacher.deploy()

    print("Preparing criterion and anchor boxes...")
    criterion_det = MultiBoxLoss(args.mutual_guide)
    criterion_kd = HintLoss(args.kd)
    priors = get_prior_box(args.anchor_size, args.image_size).cuda()

    print(
        "Distilating {}-{}-{}-{} on {} with {} images".format(
            args.kd,
            "MG" if args.mutual_guide else "Retina",
            args.neck,
            args.backbone,
            dataset.name,
            len(dataset),
        )
    )
    timer = Timer()
    for iteration in range(start_iter, end_iter):
        if iteration % epoch_size == 0:

            # save checkpoint
            save_model(ema_model.ema, iteration, "CKPT")

            # create batch iterator
            rand_loader = data.DataLoader(
                dataset,
                args.batch_size,
                shuffle=True,
                num_workers=4,
                collate_fn=detection_collate,
            )
            prefetcher = DataPrefetcher(rand_loader)
            model.train()

        # traning iteratoin
        timer.tic()
        adjust_learning_rate(
            optimizer,
            args.lr,
            iteration,
            args.warm_iter,
            end_iter,
        )
        (images, targets) = prefetcher.next()

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                out_t = teacher(images)
            out_s = model(images)
            loss_det = criterion_det(out_s, priors, targets)
            loss_kd = criterion_kd(out_t, out_s)
            loss = loss_kd + loss_det

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ema_model.update(model)
        load_time = timer.toc()

        # logging
        if iteration % 100 == 0:
            print(
                "iter {}/{}, lr {:.6f}, loss {:.2f} {:.2f}, time {:.2f}s, eta {:.2f}h".format(
                    iteration,
                    end_iter,
                    optimizer.param_groups[0]["lr"],
                    loss_det.item(),
                    loss_kd.item(),
                    load_time,
                    load_time * (end_iter - iteration) / 3600,
                )
            )
            timer.clear()

    # model saving
    save_model(ema_model.ema, iteration, "Final")
