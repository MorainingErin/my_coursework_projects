#!/bin/bash 

NETWORKS=( "lenet" "vgg" "resnet" )
LRS=( 1e-2 1e-2 1e-2 )

for (( i=0; i<${#NETWORKS[*]}; i++ ))
do
    NETWORK=${NETWORKS[i]}
    LR=${LRS[i]}

    python main.py \
        --mode train \
        --dataset split \
        --epoch_num 300 \
        --lr ${LR} \
        --lr_step 0 \
        --network ${NETWORK} \
        --dropout 0.3 \
        --data_aug all

done
