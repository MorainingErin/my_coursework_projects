#!/bin/bash 

NETWORKS=( "lenet" "vgg" "resnet" )
LRS=( 1e-2 1e-2 1e-2 )

for (( i=0; i<${#NETWORKS[*]}; i++ ))
do
    NETWORK=${NETWORKS[i]}
    LR=${LRS[i]}

    python main.py \
        --mode test \
        --dataset full \
        --epoch_num 300 \
        --lr ${LR} \
        --lr_step 0 \
        --network ${NETWORK} \
        --dropout 0.0 \
        --data_aug none
    
    python main.py \
        --mode test \
        --dataset full \
        --epoch_num 300 \
        --lr ${LR} \
        --lr_step 0 \
        --network ${NETWORK} \
        --dropout 0.3 \
        --data_aug all

    for DATAAUG in "all" "color" "geo"
    do
        python main.py \
            --mode test \
            --dataset full \
            --epoch_num 300 \
            --lr ${LR} \
            --lr_step 0 \
            --network ${NETWORK} \
            --dropout 0.0 \
            --data_aug ${DATAAUG}
        
        python main.py \
            --mode test \
            --dataset split \
            --epoch_num 300 \
            --lr ${LR} \
            --lr_step 0 \
            --network ${NETWORK} \
            --dropout 0.0 \
            --data_aug ${DATAAUG}
    done

    for DROPOUT in 0.1 0.3 0.5
    do
        python main.py \
            --mode test \
            --dataset full \
            --epoch_num 300 \
            --lr ${LR} \
            --lr_step 0 \
            --network ${NETWORK} \
            --dropout ${DROPOUT} \
            --data_aug none
        
        python main.py \
            --mode test \
            --dataset split \
            --epoch_num 300 \
            --lr ${LR} \
            --lr_step 0 \
            --network ${NETWORK} \
            --dropout ${DROPOUT} \
            --data_aug none
    done
done
