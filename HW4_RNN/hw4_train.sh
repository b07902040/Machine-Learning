#!/bin/bash
echo $1 #:training label data
echo $2 #:training unlabel data 
python3 hw4_train.py $1 $2