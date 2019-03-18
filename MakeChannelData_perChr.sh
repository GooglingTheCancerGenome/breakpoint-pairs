#!/bin/bash

inbam=$1
chr=$2
#while read chr; do
    echo 'Processing '$chr'.'
    for prg in coverage split_read_distance clipped_reads clipped_read_distance; do
        ./$prg.py -b $inbam  -c $chr -o $chr'_'$prg -l $chr'_'$prg'.log'
        echo $prg" done!"
    done
    echo 'Finished with '$chr'.'
#done < ../MinorResearchInternship/BAM/BAM_chr_list 
