#!/bin/bash
#$ -l h_rt=24:00:00
#$ -l h_vmem=10G
#$ -cwd
#$ -N CNN_Train_DellyDels
#$ -o logs
#$ -e logs


conda activate mcfly

spl=$1
epo=$2
lra=$3
cal=$4

./OC_cross_validation.py -spl $spl -epo $epo -lr $lra -cal $cal

