#!/bin/bash
CUDA_VISIBLE_DEVICES=0

data_set=cifar100
seed=1027
output_dir=exp/


python main.py \
    --data_set=$data_set \
    --output_dir=$output_dir/$data_set/"cifar100_seed($seed)" \
    --seed=$seed \
    --topK=100 \