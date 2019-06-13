#!/bin/bash

#$ -l h_rt=24:00:00
#$ -l h_vmem=8G
#$ -cwd
#$ -N CNN_Train_DellyDels
#$ -o logs
#$ -e logs


conda activate breakpoint-pairs
conda activate mcfly2

./OC_cross_validation_dropout_zeroed.py -cal $1 -input $2 -erase $3 -outpre $4 - ch $5


