#!/bin/bash

$chr=$1

echo 'Processing chromosome '$chr
for caller in gridss delly lumpy manta; do
	echo 'Processing caller '$caller
	./channel_maker.py -b ../MinorResearchInternship/BAM/NA12878.bam -c $chr -o genomewide_windowpairs/$caller/$chr'_windowpairs_DEL' -l $chr'_channelmaker_DEL.log' -vcf ../MinorResearchInternship/VCF/$caller'.sym.vcf' -svt DEL
	echo $caller' processed.'
done
echo $chr' processed.'
#done < ../MinorResearchInternship/BAM/BAM_chr_list

       
