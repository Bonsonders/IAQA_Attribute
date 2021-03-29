#!/bin/bash

AADB="../../DataBase/AADB/AADB_newtest/"
AVA=" ../../DataBase/AVA_dataset/images"
LABEL_AADB="./utils/AADB_score.txt"
LABEL_AVA="./utils/AVA_label.txt"
LR=1e-4
date=$(date '+%Y-%m-%d-%H-%M-%S')

python3 train.py --data_dir $AVA --label_dir $LABEL_AVA --lr $LR --name $date"AVA"
