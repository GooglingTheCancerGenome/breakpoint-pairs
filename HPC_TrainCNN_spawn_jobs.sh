#!/bin/bash

while read -a paras; do
<<<<<<< HEAD
echo "Processing parameters Split:"${paras[0]}", Epochs:"${paras[1]}", Learning rate:"${paras[2]}" Caller:"${paras[3]}"."
   qsub CNN_Train_DellyDels_Spl_Epo_LR.sh ${paras[0]} ${paras[1]} ${paras[2]} ${paras[3]}
done <  Parameters_HPC_LR5.txt
 
=======
    ./CNN_Train_DellyDels_Spl_Epo_LR.sh ${paras[0]} ${paras[1]} ${paras[2]} ${paras[3]}
    echo ${paras[0]} ${paras[1]} ${paras[2]} ${paras[3]} >> log
    echo "Processed parameters Split:"${paras[0]}", Epochs:"${paras[1]}", Learning rate:"${paras[2]}" Caller:"${paras[3]}"."
done < Parameters_HPC.txt
>>>>>>> 9e128b3ca1f54c5476d100b879c63d0adb87e704

