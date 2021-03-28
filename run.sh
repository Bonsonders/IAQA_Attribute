#!/bin/bash

DATA="../../DataBase/AADB/AADB_newtest/"
LABEL="./utils/AADB_score.txt"
LR=1e-3

python3 train.py --data_dir $DATA --label_dir $LABEL --lr $LR
