#!/bin/bash

#$ -l h_rt=00:05:00
#$ -l h_vmem=5G
#$ -cwd
#$ -N CNN_Train_DellyDels
#$ -o logs
#$ -e logs


conda activate breakpoint-pairs
conda activate mcfly2

./ChannelImportance_Testing.py -inmod $1 -testd $2 -calmod $3 -outpre $4 -neg $5 -iter $6 


