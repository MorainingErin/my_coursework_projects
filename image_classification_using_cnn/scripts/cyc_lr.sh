#!/bin/bash 

python main.py \
    --mode train \
    --dataset split \
    --epoch_num 150 \
    --lr 1e-10 \
    --lr_step -10 \
    --network lenet \
    --dropout 0.0 \
    --data_aug none

python main.py \
    --mode train \
    --dataset split \
    --epoch_num 150 \
    --lr 1e-10 \
    --lr_step -10 \
    --network vgg \
    --dropout 0.0 \
    --data_aug none

python main.py \
    --mode train \
    --dataset split \
    --epoch_num 150 \
    --lr 1e-10 \
    --lr_step -10 \
    --network resnet \
    --dropout 0.0 \
    --data_aug none
