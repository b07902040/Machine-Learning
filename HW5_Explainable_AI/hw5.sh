#!/bin/bash
echo $1 #:Food dataset directory
echo $2 #:Output images directory
python3 hw5.py $1 $2
python3 hw5_lime.py $1 $2
python3 deep_dream.py $1 $2