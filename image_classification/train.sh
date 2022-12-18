name="f0_swin" 

python train.py --name "$name" \
                --folder ../datasets/CROPS \
                --val_fold 0 \
                --cache cache/swin \
                --model_name swin_base_patch4_window12_384_in22k \
                --pretrained \
                --num_classes 33 \
                --epochs 10 \
                --lr 5e-5