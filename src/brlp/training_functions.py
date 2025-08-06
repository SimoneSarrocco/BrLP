"""
Collection of training functions
"""
from pathlib import Path
import time
from typing import Dict, List
import logging

import numpy as np

from monai import data
from monai import transforms
from monai.data import decollate_batch

import torch


class AverageMeter(object):
    """
    Average Meter to keep track of the metrics
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def save_checkpoint(model, epoch, filename="model.pt", best_acc=0):
    """
    Saves the model checkpoint to a file
    """
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    torch.save(save_dict, filename)
    logging.info(f"Saving checkpoint {filename}")


def datafold_read(data_list: Dict, fold: int = 0, key: str = "training") -> tuple[List, List]:
     """
     Get only the proper fold, and split it into training and validation
     """
     train = []
     val = []
     for data in data_list[key]:
         if "fold" in data and data["fold"] == fold:
             val.append(data)
         else:
             train.append(data)
 
     return train, val


def get_loader(
        batch_size: int, 
        data_list: Dict, 
        fold: int, 
        num_workers: int, 
        train_transform: transforms.Transform, 
        val_transform: transforms.Transform
    ):
    """
    Create the dataloaders.
    """
    # get the fold appropriate data
    train_files, val_files = datafold_read(data_list=data_list, fold=fold)

    train_ds = data.Dataset(data=train_files, transform=train_transform)
    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_ds = data.Dataset(data=val_files, transform=val_transform)
    val_loader = data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, device, config):
    """
    Train an epoch
    """
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        data, target = batch_data["image"].to(device), batch_data["label"].to(device)
        with torch.autocast(device_type="cuda"):
            logits = model(data)
            loss = loss_func(logits, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        run_loss.update(loss.item(), n=config["batch_size"])
        logging.info(f"Epoch {epoch}/{config['max_epochs']} {idx}/{len(loader)}, \
                     loss: {run_loss.avg:.4f}, \
                     time {time.time() - start_time:.2f}s")
        start_time = time.time()
    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, device, config, model_inferer=None, post_sigmoid=None, post_pred=None):
    """
    Validate an epoch
    """
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"].to(device), batch_data["label"].to(device)
            logits = model_inferer(data)
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            dice_score = run_acc.avg
            logging.info(f"Val {epoch}/{config['max_epochs']} {idx}/{len(loader)}, \
                         dice_score: {dice_score}, \
                         time {time.time() - start_time:.2f}s")
            start_time = time.time()

    return run_acc.avg


def trainer(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_func,
        acc_func,
        scaler,
        device,
        config,
        writer,
        checkpoint_filename,
        model_inferer=None,
        start_epoch=0,
        post_sigmoid=None,
        post_pred=None,
    ):
    """
    Train and validate the model
    """
    val_acc_max = 0.0
    dices_tc = []
    loss_epochs = []
    trains_epoch = []

    for epoch in range(start_epoch, config["max_epochs"]):

        # do the training
        logging.info(f"{time.ctime()}, Epoch: {epoch}")
        epoch_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, epoch=epoch, loss_func=loss_func, device=device, config=config)
        writer.add_scalar(f"loss/Train loss", train_loss, epoch)
        logging.info(
            f"Final training {epoch}/{config['max_epochs'] - 1},\
            loss: {train_loss:.4f},\
            time {time.time() - epoch_time :.2f}s"
        )

        # validate if in the right epoch
        if (epoch + 1) % config["val_every"] == 0 or epoch == 0:
            loss_epochs.append(train_loss)
            trains_epoch.append(int(epoch))
            epoch_time = time.time()
            val_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                device=device,
                config=config,
                model_inferer=model_inferer,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
            )
            dice_score = val_acc
            val_avg_acc = np.mean(val_acc)
            writer.add_scalar(f"accuracy/Dice Score", val_avg_acc, epoch)
            logging.info(f"Final validation stats {epoch}/{config['max_epochs'] - 1}, \
                         dice_score: {dice_score}, \
                         time {time.time() - epoch_time:.2f}s")

            dices_tc.append(dice_score)
            if val_avg_acc > val_acc_max:
                logging.info("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                val_acc_max = val_avg_acc
                save_checkpoint(model, epoch, filename=checkpoint_filename, best_acc=val_acc_max)

        # save checkpoint if in the right epoch
        if (epoch+1) % config["checkpoint_every"] == 0:
            checkpoint_name = Path(checkpoint_filename).parent / f"model_ep{epoch}.pt"
            save_checkpoint(model, epoch, filename=checkpoint_name, best_acc=val_acc_max)
            logging.info(f"Saving a checkpoint at epoch {epoch}")
    
        writer.flush()

    logging.info(f"Training Finished !, Best Accuracy: {val_acc_max}")
    return (val_acc_max, dices_tc, loss_epochs, trains_epoch)