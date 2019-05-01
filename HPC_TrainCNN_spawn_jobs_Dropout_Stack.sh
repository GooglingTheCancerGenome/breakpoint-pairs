#!/bin/bash

while read -a paras; do

echo "Processing parameters Split:"${paras[1]}", Epochs:"${paras[2]}", Learning rate:"${paras[3]}" Caller:"${paras[0]}", Dropout1: "${paras[4]}", Dropout2: "${paras[5]}", Stack:"${paras[6]}
   qsub CNN_Train_DellyDels_Spl_Epo_LR_Dropout_Stack.sh ${paras[1]} ${paras[2]} ${paras[3]} ${paras[0]} ${paras[4]} ${paras[5]} ${paras[6]}
done < Parameters_StackedWindows.txt 

