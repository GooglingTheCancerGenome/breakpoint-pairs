#!/bin/bash

while read -a paras; do

qsub ChannelImportance_Run.sh ${paras[0]}  ${paras[1]} ${paras[2]} ${paras[3]} ${paras[4]} ${paras[5]}

done < ChannelImportance_Paras.txt 

