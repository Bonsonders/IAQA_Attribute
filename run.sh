#!/bin/bash
#######################################
#Path for dataset
#&& Labels
AADB="../../DataBase/AADB/datasetImages_originalSize"
AVA=" ../../DataBase/AVA_dataset/images"
SPAQ="../..//DataBase/SPAQ/Dataset/TestImage"

LABEL_AADB="./utils/AADB_label.txt"
LABEL_AVA="./utils/AVA_label.txt"
LABEL_SPAQ="./utils/SPAQ_label.txt"
date=$(date '+%Y-%m-%d-%H-%M-%S')
AADB_test_label="./utils/AADB_Test.txt"
SPAQ_test_label="./utils/SPAQ_Test.txt"
#######################################
#Super Parameters
LR=1e-4

######################################
python3 train.py --data_dir $AADB --label_dir $LABEL_AADB --lr $LR --name $date"AADB_MobileNet_2_" --batch_size 32 --testdata_dir $AADB --testlabel_dir $AADB_test_label --crop_num 2
