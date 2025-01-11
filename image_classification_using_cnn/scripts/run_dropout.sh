#!/bin/bash 

NETWORKS=( "lenet" "vgg" "resnet" )
LRS=( 1e-2 1e-2 1e-2 )
DROPOUTS=( 0.1 0.3 0.5 )

for (( i=0; i<${#NETWORKS[*]}; i++ ))
do
    NETWORK=${NETWORKS[i]}
    LR=${LRS[i]}

    for DROPOUT in ${DROPOUTS[*]}
    do
        python main.py \
            --mode train \
            --dataset split \
            --epoch_num 300 \
            --lr ${LR} \
            --lr_step 0 \
            --network ${NETWORK} \
            --dropout ${DROPOUT} \
            --data_aug none
    done
done
