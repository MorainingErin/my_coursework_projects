#!/bin/bash 

NETWORKS=( "rnn" "lstm" )
LRS=( 1e-2 1e-2 )
DATASETS=( "raw" "fix" "part" )
NORMALIZES=( "log" "winminmax" "minmax" "returns" )

for (( i=0; i<${#NETWORKS[*]}; i++ ))
do
    NETWORK=${NETWORKS[i]}
    LR=${LRS[i]}

    for DATASET in ${DATASETS[*]}
    do

        for NORMALIZE in ${NORMALIZES[*]}
        do

            python main.py \
                --mode train \
                --epoch_num 500 \
                --lr ${LR} \
                --network ${NETWORK} \
                --dataset ${DATASET} \
                --normalize ${NORMALIZE} \
                --seqlen 8

        done

    done

    for (( l=4; l<21; l=l+2 ))
    do
        python main.py \
            --mode train \
            --epoch_num 500 \
            --lr ${LR} \
            --network ${NETWORK} \
            --dataset fix \
            --normalize minmax \
            --seqlen ${l}
    done

done
