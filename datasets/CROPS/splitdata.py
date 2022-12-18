from glob import glob
# import cv2
import shutil
import os
import random
import time

from tqdm import tqdm

cls = [
    'asparagus',
    'bambooshoots',
    'betel',
    'broccoli',
    'cauliflower',
    'chinesecabbage',
    'chinesechives',
    'custardapple',
    'grape',
    'greenhouse',
    'greenonion',
    'kale',
    'lemon',
    'lettuce',
    'litchi',
    'longan',
    'loofah',
    'mango',
    'onion',
    'others',
    'papaya',
    'passionfruit',
    'pear'
    'pennisetum',
    'redbeans',
    'roseapple',
    'sesbania',
    'soybeans',
    'sunhemp',
    'sweetpotato',
    'taro',
    'tea',
    'waterbamboo'
]

for c in cls:

    if not os.path.exists(f'./fold0'):
        os.mkdir(f'./fold0')
    if not os.path.exists(f'./fold0/{c}'):
        os.mkdir(f'./fold0/{c}')

    if not os.path.exists(f'./fold1'):
        os.mkdir(f'./fold1')
    if not os.path.exists(f'./fold1/{c}'):
        os.mkdir(f'./fold1/{c}')

    if not os.path.exists(f'./fold2'):
        os.mkdir(f'./fold2')
    if not os.path.exists(f'./fold2/{c}'):
        os.mkdir(f'./fold2/{c}')

    if not os.path.exists(f'./fold3'):
        os.mkdir(f'./fold3')
    if not os.path.exists(f'./fold3/{c}'):
        os.mkdir(f'./fold3/{c}')

    if not os.path.exists(f'./fold4'):
        os.mkdir(f'./fold4')
    if not os.path.exists(f'./fold4/{c}'):
        os.mkdir(f'./fold4/{c}')

    samples = glob(f'./train/{c}/*.jpg')
    n_1fold_sample = len(samples) // 5

    # print(f'{len(samples):5d} | {n_1fold_sample:5d}')

    random.shuffle(samples)


    for image_path in samples[:n_1fold_sample]:
        src = image_path.replace('\\','/')
        dst = src.replace('train', 'fold0')
        shutil.copy(src, dst)

    for image_path in samples[n_1fold_sample:n_1fold_sample*2]:
        src = image_path.replace('\\','/')
        dst = src.replace('train', 'fold1')
        shutil.copy(src, dst)

    for image_path in samples[n_1fold_sample*2:n_1fold_sample*3]:
        src = image_path.replace('\\','/')
        dst = src.replace('train', 'fold2')
        shutil.copy(src, dst)

    for image_path in samples[n_1fold_sample*3:n_1fold_sample*4]:
        src = image_path.replace('\\','/')
        dst = src.replace('train', 'fold3')
        shutil.copy(src, dst)

    for image_path in samples[n_1fold_sample*4:]:
        src = image_path.replace('\\','/')
        dst = src.replace('train', 'fold4')
        shutil.copy(src, dst)
