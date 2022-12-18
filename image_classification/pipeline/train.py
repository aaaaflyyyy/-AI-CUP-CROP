import logging
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import timm
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import confusion_matrix
from timm.loss import LabelSmoothingCrossEntropy
from timm.utils import AverageMeter, accuracy
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from datasets.datasets import TrainDataset, ValDataset


def split_val_train(folds, idx):
    train_folds = folds.copy()
    val_fold = [train_folds.pop(idx)]
    return train_folds, val_fold

class Train:

    def __init__(self, config: dict, device: str):
        logging.info(f'Training {config.name}')
        logging.info(f'Epochs: {config.epochs}')
        logging.info(f'Learning rate: {config.lr}')

        self.config = config
        self.device = device
        
        # timm Model
        self.model = timm.create_model(model_name=config.model_name, pretrained=config.pretrained, num_classes = config.num_classes, checkpoint_path=config.checkpoint)
        self.model.to(device)

        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        # Loss function
        if config.smoothing > 0.:
            self.loss = LabelSmoothingCrossEntropy(smoothing=config.smoothing)
            logging.info(f'Loss: Label Smoothing Cross Entropy')
        else:
            self.loss = CrossEntropyLoss()
            logging.info(f'Loss: Cross Entropy')
        self.loss.to(device)

        # Data Augmentation
        if config.finetune:
            logging.info('finetune')
            train_transform = A.Compose([
                A.Resize(config.size, config.size),
                A.Normalize(),
                ToTensorV2()
            ])
        else:
            train_transform = A.Compose([
                A.RandomResizedCrop(config.size, config.size),
                A.ColorJitter(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Blur(blur_limit= 3, p=0.3),
                A.GaussNoise(p=0.3),
                A.ShiftScaleRotate(border_mode=0, p=0.7),
                A.Normalize(),
                ToTensorV2()
            ])
        
        # Datasets
        # --------------------------------------------
        # If use use only train/val datasets.
        # folds = ['val', 'train']
        # --------------------------------------------
        folds = ['fold0', 'fold1', 'fold2', 'fold3', 'fold4']
        train_folds, val_fold = split_val_train(folds, config.val_fold)
        logging.info(f'folds | tarin fold: {str(train_folds)} | val fold: {str(val_fold)}')

        train_dataset = TrainDataset(config.folder, train_folds, train_transform)
        val_dataset = ValDataset(config.folder, val_fold, config.size)
        self.cls = train_dataset.class_to_idx

        # Dataloader
        self.train_dataloader = DataLoader(train_dataset, config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
        self.val_dataloader = DataLoader(val_dataset, config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
        
        logging.info(f'dataset size | train size: {len(self.train_dataloader) * config.batch_size} | val size: {len(self.val_dataloader) * config.batch_size}')

    def train_batch(self, inputs, targets):
        self.model.train()

        outputs = self.model(inputs)
        loss = self.loss(outputs, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    @torch.no_grad()
    def test_batch(self, inputs, targets):
        self.model.eval()

        outputs = self.model(inputs)
        preds = outputs.cpu().softmax(dim=-1).numpy().argmax(1)

        loss = self.loss(outputs, targets)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

        return loss, acc1, acc5, preds
        

    def run(self):
        best = -float('Inf')
        best_epoch = 0

        for epoch in range(1, self.config.epochs + 1):

            train_loss_meter = AverageMeter()
            val_loss_meter = AverageMeter()
            acc1_meter = AverageMeter()
            acc5_meter = AverageMeter()

            train_process = tqdm(self.train_dataloader, total=len(self.train_dataloader))
            for train_sample in train_process:
                train_images, train_labels = train_sample[0], train_sample[1]

                train_images = train_images.to(self.device)
                train_labels = train_labels.to(self.device)

                loss = self.train_batch(train_images, train_labels)

                train_loss_meter.update(loss.item())

            y_true = np.array([])
            y_pred = np.array([])

            val_process = tqdm(self.val_dataloader, total=len(self.val_dataloader))
            for val_sample in val_process:
                val_images, val_labels = val_sample[0], val_sample[1]

                val_labels_np = val_labels.numpy()

                val_images = val_images.to(self.device)
                val_labels = val_labels.to(self.device)
                loss, acc1, acc5, preds = self.test_batch(val_images, val_labels)

                y_true = np.concatenate((y_true, val_labels_np))
                y_pred = np.concatenate((y_pred, preds))
                
                val_loss_meter.update(loss.item())
                acc1_meter.update(acc1.item())
                acc5_meter.update(acc5.item())

            cm = confusion_matrix(y_true, y_pred, labels=range(self.config.num_classes))

            fp = cm.sum(axis=0) - np.diag(cm)  
            fn = cm.sum(axis=1) - np.diag(cm)
            tp = np.diag(cm)
            tn = cm.sum() - (fp + fn + tp)

            # recall = tp / (tp+fn)
            precision = tp / (tp+fp)
            # f1 = 2 * precision * recall / (precision + recall)
            wp = np.sum(precision * (tp + fn)) / len(y_true)

            logging.info(
                f'[{epoch}]  '
                f'Train Loss: {train_loss_meter.avg:.4f} | '
                f'Val Loss: {val_loss_meter.avg:.4f} | '
                f'Acc@1: {acc1_meter.avg:.2f} | '
                f'Acc@5: {acc5_meter.avg:.2f} | '
                f'Weighted Precision: {wp:.7f}')

            wandb.log({
                'Loss/Train': train_loss_meter.avg,
                'Loss/Val': val_loss_meter.avg,
                'metrics/Acc@1': acc1_meter.avg,
                'metrics/Acc@5': acc5_meter.avg,
                'metrics/Weighted Precision': wp,
            })

            df_cm = pd.DataFrame(cm, index = [c for c in self.cls.keys()], columns = [c for c in self.cls.keys()])
            
            if wp >= best or epoch==1:
                best = wp
                best_epoch = epoch

                self.save(epoch, df_cm, is_best=True)

            # self.save(epoch)
            logging.info(f'best epoch is {best_epoch}, Weighted Precision is {best:.07f}')
            


    def save(self, epoch: int, df_cm = None, is_best = False):
        path = Path(self.config.cache) / self.config.name
        path.mkdir(parents=True, exist_ok=True)

        if is_best:
            cache = path / f'best.pth'

            plt.figure(figsize = (20,14))
            sn.heatmap(df_cm, annot=True)
            plt.savefig(path / 'cm.jpg')
        else:
            cache = path / f'{epoch:03d}.pth'

        torch.save(self.model.state_dict(), cache)
        logging.info(f'save checkpoint to {str(cache)}')
