# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import ast
import glob
import json
import os
import random
import wandb
import gc
import copy

import albumentations
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from tqdm import tqdm
from scipy.spatial.distance import directed_hausdorff


# +
def seed_everything(seed):
    torch.manual_seed(seed)  # for all devices (both CPU and CUDA)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('seed is set')


def dice_score(y_true, y_pred, thres=0.5, dim=(-2, -1), empty=1, eps=0.00001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred >= thres).to(torch.float32)
    intersection = 2 * (y_true * y_pred).sum(dim=dim)
    masks_sum = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    if empty == 1:
        # when y_true[n] is empty and y_pred[n] also empty, dice[n] = 1
        dice = ((intersection + eps) / (masks_sum + eps)).mean()
    else:
        # when y_true[n] is empty and y_pred[n] also empty, dice[n] = 0
        dice = ((intersection) / (masks_sum + eps)).mean()
    return dice


def iou_score(y_true, y_pred, thres=0.5, dim=(-2, -1), empty=1, eps=0.00001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thres).to(torch.float32)
    intersection = (y_true * y_pred).sum(dim=dim)
    union = ((y_true + y_pred) > 0).sum(dim=(-2, -1))
    if empty == 1:
        # when y_true[n] is empty and y_pred[n] also empty, iou[n] = 1
        iou = ((intersection + eps) / (union + eps)).mean(dim=(1, 0))
    else:
        # when y_true[n] is empty and y_pred[n] also empty, iou[n] = 0
        iou = ((intersection) / (union + eps)).mean(dim=(1, 0))
    return iou


def compute_hausdorff(y_pred, y_true, max_dist=np.sqrt(2) * 384):
    sum_dist = 0
    for i in range(y_pred.shape[0]):
        pred = np.where(y_pred[i] > 0.5, 1, 0)
        gt = y_true[i]
        if np.all(pred == gt):
            dist = 0.0
        dist = directed_hausdorff(np.argwhere(pred), np.argwhere(gt))[0]
        if dist > max_dist:  # when gt is all 0s, may get inf.
            dist = max_dist
        sum_dist += dist

    return sum_dist / y_pred.shape[0]


def train_one_epoch(epoch, model, dataloader, loss_fn, optimizer, scheduler):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train()
    step_update_lr = (len(dataloader) // 30) * 10  # 3 times per epoch
    running_loss = 0
    epoch_loss = 0
    total_dice = 0
    # show train progress bar
    train_bar = tqdm(
        enumerate(dataloader), total=len(dataloader), desc=f"Train: epoch #{epoch + 1}"
    )
    for n, (imgs, masks) in train_bar:
        optimizer.zero_grad()

        imgs = imgs.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        y_pred = model.forward(imgs)  # logits
        loss = loss_fn(y_pred, masks)
        dice_metric = dice_score(
            masks,
            torch.nn.Sigmoid()(y_pred),
        )
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_loss = running_loss / (n + 1)
        if (n + 1) % step_update_lr == 0:
            if scheduler is not None:
                scheduler.step()
        # add loss and lr to progress bar
        current_lr = optimizer.param_groups[0]["lr"]
        total_dice += dice_metric
        memory = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        train_bar.set_postfix(
            train_loss=f"{epoch_loss:0.4f}",
            lr=f"{current_lr:0.6f}",
            current_dice=f"{dice_metric:0.4f}",
            epoch_dice=f"{(total_dice / (n + 1)):0.4f}",
            gpu_memory=f"{memory:0.2f} GB",
        )
    gc.collect()
    torch.cuda.empty_cache()
    return epoch_loss

def val_one_epoch(epoch, model, dataloader, loss_fn):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    running_loss = 0
    running_dice = 0
    running_iou = 0
    running_hd = 0
    val_metrics = {"dice": 0, "iou": 0, "loss": 0, "hd": 0}
    # show train progress bar
    val_bar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Val:   epoch #{epoch + 1}",
    )
    for n, (imgs, masks) in val_bar:

        imgs = imgs.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        with torch.no_grad():
            y_pred = model.forward(imgs)  # logits
        loss = loss_fn(y_pred, masks)
        y_pred = torch.nn.Sigmoid()(y_pred)
        dice = (
            dice_score(
                masks,
                y_pred,
            )
            .cpu()
            .detach()
            .numpy()
        )
        iou = (
            iou_score(
                masks,
                y_pred,
            )
            .cpu()
            .detach()
            .numpy()
        )
        h_dist = compute_hausdorff(
            y_pred.detach().cpu().numpy(),
            masks.detach().cpu().numpy(),
            np.sqrt(2) * masks.shape[-1]
        )

        # sum metrics
        running_loss += loss.item()
        running_dice += dice
        running_iou += iou
        running_hd += h_dist

        # metrics for epoch
        epoch_loss = running_loss / (n + 1)
        epoch_dice = running_dice / (n + 1)
        epoch_iou = running_iou / (n + 1)
        epoch_hd = running_hd / (n + 1)
        
        # add loss and lr to progress bar
        memory = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        val_bar.set_postfix(
            val_loss=f"{epoch_loss:0.4f}",
            current_dice=f"{dice:0.4f}",
            epoch_dice=f"{epoch_dice:0.4f}",
            current_iou=f"{iou:0.4f}",
            epoch_iou=f"{epoch_iou:0.4f}",
            epoch_hd=f"{epoch_hd:0.4f}",
            gpu_memory=f"{memory:0.2f} GB",
        )
    val_metrics["dice"] = epoch_dice
    val_metrics["iou"] = epoch_iou
    val_metrics["loss"] = epoch_loss
    val_metrics['hd'] = epoch_hd
    
    gc.collect()
    torch.cuda.empty_cache()
    return val_metrics

def train(
    model,
    optimizer,
    scheduler,
    n_epochs,
    config,
    train_dataloader,
    val_dataloader,
    loss_fn,
    wandb_log=False,
    early_stopping=5,
):

    if wandb_log:
        wandb.watch(model, log_freq=100)
    device = config["DEVICE"]
    folder = os.path.join(config["MODEL_PATH"], "weights")
    os.makedirs(folder, exist_ok=True)

    best_dice = 0  # min dise_score = 0
    best_loss = 1e10  # max loss
    best_dice_epoch = None
    best_loss_epoch = None
    early_stopping_counter = 0
    best_weights = {"dice": None, "loss": None}
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
        "val_hd": [],
        "best_dice": None,
        "best_loss": None,
        "best_dice_epoch": None,
        "best_loss_epoch": None,
    }
    for epoch in range(n_epochs):
        # training for one epoch
        train_loss = train_one_epoch(
            epoch=epoch,
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        # validation
        val_metrics = val_one_epoch(
            epoch=epoch,
            model=model,
            dataloader=val_dataloader,
            loss_fn=loss_fn,
        )
        # saving epoch results to history and wandb
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_dice"].append(val_metrics["dice"])
        history["val_iou"].append(val_metrics["iou"])
        history["val_hd"].append(val_metrics["hd"])
        if wandb_log:
            wandb.log(
                {
                    "Train Loss": train_loss,
                    "Valid Loss": val_metrics["loss"],
                    "Valid Dice": val_metrics["dice"],
                    "Valid IOU": val_metrics["iou"],
                    "Valid HD": val_metrics["hd"],
                    "LR": scheduler.get_last_lr()[0],
                }
            )
        # check if there is improvement in val_loss or val_dice
        is_dice = val_metrics["dice"] >= best_dice
        is_loss = val_metrics["loss"] <= best_loss

        if is_dice:
            best_dice = val_metrics["dice"]
            best_dice_epoch = epoch
            history["best_dice"] = best_dice
            history["best_dice_epoch"] = best_dice_epoch
            best_weights["dice"] = copy.deepcopy(model.state_dict())
            PATH = os.path.join(folder, "best_dice_weights.pth")
            print(f"Valid dice_score improved, model saved to {PATH}")
            torch.save(model.state_dict(), PATH)
            early_stopping_counter = 0

        if is_loss:
            best_loss = val_metrics["loss"]
            best_loss_epoch = epoch
            history["best_loss"] = best_loss
            history["best_loss_epoch"] = best_loss_epoch
            best_weights["loss"] = copy.deepcopy(model.state_dict())
            PATH = os.path.join(folder, "best_loss_weights.pth")
            print(f"Valid loss improved, model saved to {PATH}")
            torch.save(model.state_dict(), PATH)
            early_stopping_counter = 0

        if not is_dice and not is_loss:
            early_stopping_counter += 1  # with no improvement
        print("\n")

        # save last checkpoints
        LAST_PATH = os.path.join(folder, "last.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss_fn,
            },
            LAST_PATH,
        )

        if early_stopping_counter == early_stopping:
            print("early_stopping")
            break

    # print result after training
    print(f"Best dice_score: {best_dice:0.4f} at {best_dice_epoch} epoch.")
    print(f"Best loss:       {best_loss:0.4f} at {best_loss_epoch} epoch.")
    with open(os.path.join(config["MODEL_PATH"], "config.json"), "w") as f:
        json.dump(config, f)
    with open(os.path.join(config["MODEL_PATH"], "history.json"), "w") as f:
        json.dump(history, f)
    return history, best_weights


# -