#!/bin/bash

DATA="../../DataBase/AADB/AADB_newtest/"
LABEL="./utils/AADB_score.txt"
python3 train.py --data_dir $DATA --label_dir $LABEL
