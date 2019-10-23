#!/bin/bash
#$ -l h_rt=02:00:00
#$ -l h_vmem=50G
#$ -cwd
#$ -N genomewide_windowpairs_negative
#$ -o logs_neg
#$ -e logs_neg
#$ -m beas
#$ -M S.L.Mehrem-2@umcutrecht.nl


conda activate breakpoint-pairs

while read chr; do
    echo 'Processing chromosome '$chr
    ./channel_maker.py -b ../NA12878/BAM/NA12878.bam  -c $chr -o NEWgenomewide_windowpairs_CR_TrueSV/$chr'_negative_windowpairs_DEL' -l NEWgenomewide_windowpairs_CR_TrueSV/$chr'_negative_channelmaker_DEL.log' -vcf ../NA12878/VCF/manta'.new.vcf' -svt DEL -neg True -negf Negative_CR_TrueSV_Balanced.txt
    echo $chr' processed.'
done < BAM_chr_list.txt
