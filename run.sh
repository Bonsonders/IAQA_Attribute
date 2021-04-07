#!/bin/bash

AADB="../../DataBase/AADB/datasetImages_originalSize"
AVA=" ../../DataBase/AVA_dataset/images"
LABEL_AADB="./utils/AADB_label.txt"
LABEL_AVA="./utils/AVA_label.txt"
LR=1e-4
date=$(date '+%Y-%m-%d-%H-%M-%S')
AADB_test="../../DataBase/AADB/AADB_newtest"
AADB_test_label="../../DataBase/AADB/AADB_newtest/labels.txt"

python3 train.py --data_dir $AADB --label_dir $LABEL_AADB --lr $LR --name $date"AADB" --batch_size 32 --testdata_dir $AADB_test --testlabel_dir $AADB_test_label
