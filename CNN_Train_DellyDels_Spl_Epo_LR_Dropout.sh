#!/bin/bash
#$ -l h_rt=24:00:00
#$ -l h_vmem=8G
#$ -cwd
#$ -N CNN_Train_DellyDels
#$ -o logs
#$ -e logs


conda activate breakpoint-pairs
conda activate mcfly2

spl=$1
epo=$2
lra=$3
cal=$4
drp1=$5
drp2=$6

./OC_cross_validation.py -spl $spl -epo $epo -lr $lra -cal $cal -drp1 $drp1 -drp2 $drp2

