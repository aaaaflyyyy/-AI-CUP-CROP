import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import timm
import torch
from sklearn.metrics import confusion_matrix
from timm.loss import LabelSmoothingCrossEntropy
from timm.utils import AverageMeter, accuracy
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.datasets import ValDataset


class Test:

    def __init__(self, config: dict, device: str):
        logging.info(f'Testing {config.name}')

        self.config = config
        self.device = device
        
        # modules
        model = timm.create_model(model_name=config.model_name, pretrained=config.pretrained, num_classes = config.num_classes, checkpoint_path=config.checkpoint)
        model.to(device)
        self.model = model

        # loss
        if config.smoothing > 0.:
            self.loss = LabelSmoothingCrossEntropy(smoothing=config.smoothing)
        else:
            self.loss = CrossEntropyLoss()
        self.loss.to(device)
       
        folds = ['fold0', 'fold1_400', 'fold2', 'fold3', 'fold4']
        val_dataset = ValDataset(config.folder, [folds[config.val_fold]], config.size)
        self.val_dataloader = DataLoader(val_dataset, config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
        self.cls = val_dataset.class_to_idx

        logging.info(f'val fold: {folds[config.val_fold]}')
        logging.info(f'dataset | folder: {str(config.folder)} | val size: {len(self.val_dataloader) * config.batch_size}')

    @torch.no_grad()
    def test(self, inputs, targets):
        self.model.eval()

        outputs = self.model(inputs)
        preds = outputs.cpu().softmax(dim=-1).numpy().argmax(1)

        loss = self.loss(outputs, targets)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

        return loss, acc1, acc5, preds
        
    def run(self):
        loss_meter = AverageMeter()
        acc1_meter = AverageMeter()
        acc5_meter = AverageMeter()

        y_true = np.array([])
        y_pred = np.array([])

        val_process = tqdm(self.val_dataloader, total=len(self.val_dataloader))
        for val_sample in val_process:
            val_images, val_labels = val_sample[0], val_sample[1]
            val_labels_np = val_labels.numpy()

            val_images = val_images.to(self.device)
            val_labels = val_labels.to(self.device)

            loss, acc1, acc5, preds = self.test(val_images, val_labels)

            loss_meter.update(loss.item())
            acc1_meter.update(acc1.item())
            acc5_meter.update(acc5.item())

            y_true = np.concatenate((y_true, val_labels_np))
            y_pred = np.concatenate((y_pred, preds))
                
        cm = confusion_matrix(y_true, y_pred, labels=range(33))

        fp = cm.sum(axis=0) - np.diag(cm)  
        fn = cm.sum(axis=1) - np.diag(cm)
        tp = np.diag(cm)
        tn = cm.sum() - (fp + fn + tp)

        # recall = tp / (tp+fn)
        precision = tp / (tp+fp)
        # f1 = 2 * precision * recall / (precision + recall)
        wp = np.sum(precision * (tp + fn)) / len(y_true)

        logging.info(
            f'Loss: {loss_meter.avg:.4f} | '
            f'Acc@1: {acc1_meter.avg:.2f} | '
            f'Acc@5: {acc5_meter.avg:.2f} | '
            f'Weighted Precision: {wp:.7f}')

        df_cm = pd.DataFrame(cm, index = [c for c in self.cls.keys()], columns = [c for c in self.cls.keys()])
        self.save(df_cm)

    def save(self, df_cm):
        path = Path(self.config.cache) / self.config.name
        path.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize = (20,14))
        sn.heatmap(df_cm, annot=True)
        plt.savefig(f'{self.config.cache}/{self.config.name}/confusion_matrix.jpg')

        