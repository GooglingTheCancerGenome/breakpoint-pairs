#!/bin/bash
#$ -l h_rt=01:00:00
#$ -l h_vmem=20G
#$ -cwd
#$ -N genomewide_windowpairs
#$ -o logs
#$ -e logs
#$ -m beas
#$ -M S.L.Mehrem-2@umcutrecht.nl


conda activate breakpoint-pairs

while read chr; do
    echo 'Processing chromosome '$chr
    for caller in gridss; do
        echo 'Processing caller '$caller
	./channel_maker.py -b ../NA12878/BAM/NA12878.bam  -c $chr -o genomewide_windowpairs/$caller/$chr'_windowpairs_DEL' -l genomewide_windowpairs/$caller/$chr'_channelmaker_DEL.log' -vcf ../NA12878/VCF/$caller'.sym.vcf' -svt DEL
	echo $caller' processed.'
    done
    echo $chr' processed.'
done < BAM_chr_list.txt

       
