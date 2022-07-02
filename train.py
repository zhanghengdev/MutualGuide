#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import argparse
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
    detection_collate,
    DataPrefetcher,
)
from utils import (
    Timer,
    PriorBox,
    ModelEMA,
    MultiBoxLoss,
    tencent_trick,
    adjust_learning_rate,
)

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

parser = argparse.ArgumentParser(description="Mutual Guide Training")
parser.add_argument("--dataset", default="COCO")
parser.add_argument("--neck", default="pafpn")
parser.add_argument("--backbone", default="resnet18")
parser.add_argument("--anchor_size", default=48.0, type=float)
parser.add_argument("--image_size", default=640, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--max_epoch", default=100, type=int)
parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument("--warm_iter", default=500, type=int)
parser.add_argument("--resume_ckpt", default=None)
parser.add_argument("--save_folder", default="weights/")
parser.add_argument("--mutual_guide", default=True, action="store_true")
args = parser.parse_args()
print(args)


def save_model(
    model: nn.Module,
    iteration: int,
    suffix: str,
) -> None:
    save_path = os.path.join(
        args.save_folder,
        "{}_{}_{}_size{}_anchor{}_{}_{}.pth".format(
            args.dataset,
            args.neck,
            args.backbone,
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

    print("Loading dataset...")
    if args.dataset == "COCO":
        train_sets = [("2017", "train")]
        dataset = COCODetection(train_sets, args.image_size)
        epoch_size = len(dataset) // args.batch_size
    elif args.dataset == "VOC":
        train_sets = [("2007", "trainval"), ("2012", "trainval")]
        dataset = VOCDetection(train_sets, args.image_size)
        epoch_size = len(dataset) // args.batch_size
    elif args.dataset == "XML":
        dataset = XMLDetection("train", args.image_size)
        epoch_size = len(dataset) // args.batch_size
    else:
        raise NotImplementedError("ERROR: Unkown dataset {}".format(args.dataset))
    end_iter = epoch_size * args.max_epoch

    print("Loading network...")
    model = Detector(
        args.image_size, dataset.num_classes, args.backbone, args.neck, mode="normal"
    ).cuda()
    optimizer = optim.SGD(
        tencent_trick(model), lr=args.lr, momentum=0.9, weight_decay=0.0005
    )

    if args.resume_ckpt:
        print("Resuming checkpoint from", args.resume_ckpt)
        state_dict = torch.load(args.resume_ckpt)
        model.load_state_dict(state_dict["model"], strict=True)
        optimizer.load_state_dict(state_dict["optimizer"])
        start_iter = state_dict["iteration"]
    else:
        start_iter = 0

    print("Preparing criterion and anchor boxes...")
    criterion = MultiBoxLoss(args.mutual_guide)
    priors = PriorBox(args.anchor_size, args.image_size).cuda()

    print(
        "Training {}-{}-{} on {} with {} images".format(
            "MG" if args.mutual_guide else "Retina",
            args.neck,
            args.backbone,
            dataset.name,
            len(dataset),
        )
    )
    timer = Timer()
    ema_model = ModelEMA(model)
    scaler = torch.cuda.amp.GradScaler()
    os.makedirs(args.save_folder, exist_ok=True)
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
        adjust_learning_rate(optimizer, args.lr, iteration, args.warm_iter, end_iter)
        (images, targets) = prefetcher.next()

        with torch.cuda.amp.autocast():
            out = model(images)
            loss = criterion(out, priors, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        ema_model.update(model)
        load_time = timer.toc()

        # logging
        if iteration % 100 == 0:
            print(
                "iter {}/{}, lr {:.6f}, loss {:.2f}, time {:.2f}s, eta {:.2f}h".format(
                    iteration,
                    end_iter,
                    optimizer.param_groups[0]["lr"],
                    loss.item(),
                    load_time,
                    load_time * (end_iter - iteration) / 3600,
                )
            )
            timer.clear()

    # model saving
    save_model(ema_model.ema, iteration, "Final")
