#!/bin/bash
fi=$1
#$ -l h_rt=01:30:00
#$ -l h_vmem=18G
#$ -cwd
#$ -N genomewide_windowpairs
#$ -o logs_$fi
#$ -e logs_$fi


conda activate breakpoint-pairs

while read chr; do
    echo 'Processing chromosome '$chr
    for caller in gridss delly lumpy manta; do
        echo 'Processing caller '$caller
	./channel_maker.py -b ../NA12878/BAM/NA12878.bam  -c $chr -o genomewide_windowpairs/$caller/$chr'_windowpairs_DEL' -l genomewide_windowpairs/$caller/$chr'_channelmaker_DEL.log' -vcf ../NA12878/VCF/$caller'.sym.vcf' -svt DEL
	echo $caller' processed.'
    done
    echo $chr' processed.'
done < $fi

       
