#!/bin/bash

DATA="../../DataBase/AADB/AADB_newtest/"
LABEL="../../DataBase/AADB/AADB_newtest/labels.txt"
python3 train.py --data_dir $DATA --label_dir $LABEL
