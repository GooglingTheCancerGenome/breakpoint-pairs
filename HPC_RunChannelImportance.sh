#!/bin/bash

while read -a paras; do

qsub ChannelImportance_Run.sh ${paras[0]}  ${paras[1]} ${paras[2]} ${paras[3]}

done < ChannelImportance_Paras.txt 

