# 農地作物現況調查影像辨識競賽–秋季賽：AI作物影像判釋

## 1. 安裝環境

```
git clone https://github.com/aaaaflyyyy/AI-CUP-CROP.git

cd AI-CUP-CROP.git
pip install -r requirements.txt
```

```
AI-CUP-CROP
├── datasets/CROP
│   ├── loc_cnt.txt
│   ├── splitdata.py
│   |   ...
├── image_classification
|   ...
```

## 3. 準備訓練資料

下載資料集放在datasets/CROP/train下
```
datasets/CROP/train
├── asparagus
│   ├── xxxxxxx.jpg
│   ├── xxxxxxx.jpg
│   |   ...
├── bambooshoots
│   ├── xxxxxxx.jpg
│   ├── xxxxxxx.jpg
│   |   ...
|   ...
```

```
python splitdata.py
```

## 4. Train
```
cd ../../image_classification
python train.py --name "$name" \
                --folder ../datasets/CROPS \
                --val_fold 0 \
                --cache cache/swin \
                --model_name swin_base_patch4_window12_384_in22k \
                --pretrained \
                --num_classes 33 \
                --epochs 10 \
                --lr 5e-5
```

## 5. inference
```
python inference.py 
```
