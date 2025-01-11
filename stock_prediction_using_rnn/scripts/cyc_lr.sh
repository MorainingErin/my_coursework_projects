#!/bin/bash 

python main.py \
    --mode train \
    --epoch_num 150 \
    --lr 1e-10 \
    --lr_step -10 \
    --network rnn \
    --dataset fix \
    --normalize returns \
    --seqlen 8

python main.py \
    --mode train \
    --epoch_num 150 \
    --lr 1e-10 \
    --lr_step -10 \
    --network rnn \
    --dataset fix \
    --normalize minmax \
    --seqlen 8

python main.py \
    --mode train \
    --epoch_num 150 \
    --lr 1e-10 \
    --lr_step -10 \
    --network lstm \
    --dataset fix \
    --normalize returns \
    --seqlen 8

python main.py \
    --mode train \
    --epoch_num 150 \
    --lr 1e-10 \
    --lr_step -10 \
    --network lstm \
    --dataset fix \
    --normalize minmax \
    --seqlen 8
