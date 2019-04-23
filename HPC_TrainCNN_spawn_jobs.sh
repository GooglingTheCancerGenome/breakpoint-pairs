#!/bin/bash

while read -a paras; do
echo "Processing parameters Split:"${paras[0]}", Epochs:"${paras[1]}", Learning rate:"${paras[2]}" Caller:"${paras[3]}"."
   qsub CNN_Train_DellyDels_Spl_Epo_LR.sh ${paras[0]} ${paras[1]} ${paras[2]} ${paras[3]}
done <  Parameters_HPC_LR5.txt
 

