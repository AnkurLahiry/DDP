#! /usr/bin/env bash

epochs=$1
gpus_per_node=$2
nproc=$3
WORK=$4
HOST=$5
rank=$PMI_RANK
nodes=22470
edges=171002


echo "Printing rank from the bash script $rank"

python3 facebook_train.py -r $rank -g $gpus_per_node -n $nproc -w $WORK -a $HOST --nodes $nodes --edges $edges

