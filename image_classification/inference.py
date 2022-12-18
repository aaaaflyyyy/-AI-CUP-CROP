import os
import warnings
from datetime import datetime

import torch
from pytorch_toolbelt.inference import tta
from torch.utils.data import DataLoader
from tqdm import tqdm

import timm

from datasets.datasets import TestDataset

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

class Pred:
    def __init__(self, model_list: list, image_path: str, image_list: str, size:int):
        # device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # models
        self.models = []
        for model_name, checkpoint_path, weight in model_list:
            model = timm.create_model(model_name, pretrained=False, num_classes = 33, checkpoint_path=checkpoint_path)
            model.to(device)
            model.eval()
            self.models.append((model, weight))
        # datasets
        test_dataset = TestDataset(image_path, image_list, size)
        self.test_dataloader = DataLoader(test_dataset, 32, shuffle=False, num_workers=4, pin_memory=True)
        self.class_to_idx = test_dataset.class_to_idx

    def __call__(self, output_filename, location_rate=0, output_mode = 'default'):
        # open output file.
        if output_mode == 'default':
            output_file = open(f'results/{output_filename}.csv','w')
            output_file.write(f'filename,label\n')

        elif output_mode == 'tta':
            output_file = open(f'results/{output_filename}_tta.csv','w')
            output_file.write(f'filename,label\n')

        elif output_mode == 'both':
            output_file = open(f'results/{output_filename}.csv','w')
            output_tta_file = open(f'results/{output_filename}_tta.csv','w')
            output_file.write(f'filename,label\n')
            output_tta_file.write(f'filename,label\n')

        else:
            assert False
        
        # load test data.
        process = tqdm(self.test_dataloader, total=len(self.test_dataloader))
        for sample in process:
            
            images, locs, filenames = sample[0], sample[1], sample[2]

            images = images.to(self.device)
            locs = locs.to(self.device)
            
            if output_mode == 'both':
                ensemble_img_p = torch.zeros((2,len(images),33), dtype=torch.float32).to(self.device)
            else:
                ensemble_img_p = torch.zeros((1,len(images),33), dtype=torch.float32).to(self.device)

            # model forwardimg.
            with torch.no_grad():
                for model, weight in self.models:
                    if output_mode == 'default':
                        img_p = model(images)
                        img_p = img_p.softmax(dim=-1)
                        ensemble_img_p[0] += (img_p * weight)

                    elif output_mode == 'tta':
                        img_p = tta.fliplr_image2label(model, images)
                        img_p = img_p.softmax(dim=-1)
                        ensemble_img_p[0] += (img_p * weight)

                    elif output_mode == 'both':
                        
                        img_p = model(images)
                        img_p = img_p.softmax(dim=-1)
                        ensemble_img_p[0] += (img_p * weight)

                        img_p_tta = tta.fliplr_image2label(model, images)
                        img_p_tta = img_p_tta.softmax(dim=-1)
                        ensemble_img_p[1] += (img_p * weight)
            
            # using location imformation.
            if location_rate > 0:
                locs = torch.nn.functional.normalize(locs)
                outputs = ensemble_img_p[0] * (1-location_rate) + locs * location_rate
            else:
                outputs = ensemble_img_p[0]
            
            # max probability to label
            preds = outputs.cpu().numpy().argmax(1)

            # wirte results.
            for filename, pred in zip(filenames, preds):
                output_file.write(f'{filename},{list(self.class_to_idx.keys())[int(pred)]}\n')

            if output_mode == 'both':
                if location_rate > 0:
                    locs = torch.nn.functional.normalize(locs)
                    output_tta = ensemble_img_p[1] * (1-location_rate) + locs * location_rate
                else:
                    output_tta = ensemble_img_p[1]

                preds_tta = output_tta.cpu().numpy().argmax(1)
                
                for filename, pred_tta in zip(filenames, preds_tta):
                    output_tta_file.write(f'{filename},{list(self.class_to_idx.keys())[int(pred_tta)]}\n')
            

if __name__ == '__main__':

    w = [45, 45, 10]
    weights = [w[0] / 100 / 5, w[1] / 100 / 5, w[2] / 100 / 1]
    assert sum(w) == 100

    model_list = [
        # ==================
        #  transformer base
        # ==================

        # swin
        ('swin_base_patch4_window12_384_in22k', './cache/swin/f0_swin_aug_lr1e-5_ft/best.pth', weights[0]),
        ('swin_base_patch4_window12_384_in22k', './cache/swin/f1_swin_aug_lr1e-5_ft/best.pth', weights[0]),
        ('swin_base_patch4_window12_384_in22k', './cache/swin/f2_swin_aug_lr1e-5_ft/best.pth', weights[0]),
        ('swin_base_patch4_window12_384_in22k', './cache/swin/f3_swin_aug_lr1e-5_ft/best.pth', weights[0]),
        ('swin_base_patch4_window12_384_in22k', './cache/swin/f4_swin_aug_lr1e-5_ft/best.pth', weights[0]),

        # swinv2
        ('swinv2_base_window12to24_192to384_22kft1k', './cache/swinv2_aug_lr1e-5_ft/best.pth', weights[2]),

        # ==================
        #     CNN base
        # ==================

        # efficientnet
        # ('tf_efficientnet_b4_ns', './cache/efficientnet/f0_efficientnet_aug_lr1e-5_ft/best.pth', weights[1]),
        # ('tf_efficientnet_b4_ns', './cache/efficientnet/f1_efficientnet_aug_lr1e-5_ft/best.pth', weights[1]),
        # ('tf_efficientnet_b4_ns', './cache/efficientnet/f2_efficientnet_aug_lr1e-5_ft/best.pth', weights[1]),
        # ('tf_efficientnet_b4_ns', './cache/efficientnet/f3_efficientnet_aug_lr1e-5_ft/best.pth', weights[1]),
        # ('tf_efficientnet_b4_ns', './cache/efficientnet/f4_efficientnet_aug_lr1e-5_ft/best.pth', weights[1]),

        # convnext
        ('convnext_base_384_in22ft1k', './cache/convnext/f0_convnext_aug_lr1e-5_ft/best.pth', weights[1]),
        ('convnext_base_384_in22ft1k', './cache/convnext/f1_convnext_aug_lr1e-5_ft/best.pth', weights[1]),
        ('convnext_base_384_in22ft1k', './cache/convnext/f2_convnext_aug_lr1e-5_ft/best.pth', weights[1]),
        ('convnext_base_384_in22ft1k', './cache/convnext/f3_convnext_aug_lr1e-5_ft/best.pth', weights[1]),
        ('convnext_base_384_in22ft1k', './cache/convnext/f4_convnext_aug_lr1e-5_ft/best.pth', weights[1]),
    ]

    # test dir.
    image_path = '../datasets/CROPS/test/'
    image_list = '../datasets/CROPS/tag_locCoor_test.csv'

    output_filename = f'ans_{datetime.now().month:02d}{datetime.now().day:02d}_swin_convnext_swinv2_all_{w[0]}{w[1]}{w[2]}'
    location_rate = 0.2

    if location_rate > 0.:
        output_filename += '_loc2'

    # default / tta / both (2 files)
    output_mode = 'tta' 

    print(f'[INFO]: location_rate: {location_rate}')
    print(f'[INFO]: output_mode: {output_mode}')
    print(f'[INFO]: output_filename: {output_filename}')

    size = 384

    assert not os.path.exists(f'results/{output_filename}.csv')
    # predict
    predictor = Pred(model_list, image_path, image_list, size)
    predictor(output_filename, location_rate=location_rate, output_mode=output_mode)
