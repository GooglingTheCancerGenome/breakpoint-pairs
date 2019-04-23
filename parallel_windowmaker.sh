#! /bin/bash
#$ -l h_rt=08:00:00
#$ -l h_vmem=50G
#$ -pe threaded 6
#$ -N genomewide_windowpairs
#$ -o logs
#$ -e logs
#$ -m beas
#$ -M S.L.Mehrem-2@umcutrecht.nl

#Creating window pairs per chromosome and caller (only deletions for now) using gnu parallel
#Loading conda environment for channelmaker etc

conda activate breakpoint-pairs

parallel -j 5 './MakeWindowPairs_genomewide.sh {}' :::: BAM_chr_list
