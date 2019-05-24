#!/bin/bash

#$ -l h_rt=00:05:00
#$ -l h_vmem=1G
#$ -cwd
#$ -N CNN_Train_DellyDels
#$ -o logs_qsub
#$ -e logs_qsub



while read -a paras; do

qsub ChannelImportance_Run.sh ${paras[0]}  ${paras[1]} ${paras[2]} ${paras[3]} ${paras[4]} ${paras[5]}

done < ChannelImportance_Paras.txt  

