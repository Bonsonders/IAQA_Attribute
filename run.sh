#!/bin/bash

AADB="../../DataBase/AADB/images/"
AVA=" ../../DataBase/AVA_dataset/images"
LABEL_AADB="./utils/AADB_label.txt"
LABEL_AVA="./utils/AVA_label.txt"
LR=1e-3
date=$(date '+%Y-%m-%d-%H-%M-%S')

python3 train.py --data_dir $AADB --label_dir $LABEL_AADB --lr $LR --name $date"AADB_Reisze" --batch_size 32
