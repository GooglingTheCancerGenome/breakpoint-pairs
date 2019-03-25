import numpy as np
import gzip
from labels import SVRecord_generic
from pysam import VariantFile
import re
import os

chr_list = []
with open("../MinorResearchInternship/BAM/BAM_chr_list","r") as f:
    for line in f:
        line = line.strip()
        chr_list += [line]

for chrom in chr_list:
    #filename = "genomewide_windowpairs/delly/"+chrom+"_windowpairs_DEL.npy.gz"
    #with gzip.GzipFile(filename, "rb") as f:
        ##shape = X.shape
    counter = 0
    for vcf_file in ["../MinorResearchInternship/VCF/delly.sym.vcf"]:
        assert os.path.isfile(vcf_file)
        vcf_in = VariantFile(vcf_file, 'r')
        caller = re.findall(r'^\w*', vcf_file)
        for rec in vcf_in.fetch():
            svrec = SVRecord_generic(rec, "delly")
            startCI = abs(svrec.cipos[0]) + svrec.cipos[1]
            endCI = abs(svrec.ciend[0]) + svrec.ciend[1]
            if startCI <= 200 and endCI <= 200 and svrec.chrom == "1" and svrec.svtype == "DEL":
                counter += 1

    #if counter == shape[0]:
        #print("OK")

print(counter)