#!/bin/bash
#######################################
#Path for dataset
#&& Labels
AADB="../../DataBase/AADB/datasetImages_originalSize"
AVA=" ../../DataBase/AVA_dataset/images"

LABEL_AADB="./utils/AADB_label.txt"
LABEL_AVA="./utils/AVA_label.txt"
date=$(date '+%Y-%m-%d-%H-%M-%S')
AADB_test_label="./utils/AADB_Test.txt"
#######################################
#Super Parameters
LR=1e-4

######################################
python3 train.py --data_dir $AADB --label_dir $LABEL_AADB --lr $LR --name $date"AADB_MobileNet" --batch_size 32 --testdata_dir $AADB --testlabel_dir $AADB_test_label
