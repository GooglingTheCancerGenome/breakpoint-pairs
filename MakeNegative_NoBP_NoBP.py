from intervaltree import Interval, IntervalTree
from labels import SVRecord_generic
from pysam import VariantFile
import re
import os
import random

callers=["delly", "manta", "lumpy", "gridss"]
chr_list = []
#with open("/home/cog/smehrem/MinorResearchInternship/BAM/BAM_chr_list", "r") as f:
with open("/home/cog/smehrem/MinorResearchInternship/BAM/BAM_chr_list", "r") as f:
    for line in f:
        line = line.strip()
        chr_list += [line]

#out=open("Gridss_WeirdDels.vcf", "w")
counter=0
interval_dict={}
sv_counter_dict={}
for caller in callers:
    vcf_file="/home/cog/smehrem/MinorResearchInternship/VCF/"+caller+".sym.vcf"
    assert os.path.isfile(vcf_file)
    vcf_in = VariantFile(vcf_file, 'r')
    #if caller == "gridss":
        #out.write(str(vcf_in.header) + "\n")
    for chrom in chr_list:
        interval_dict[chrom]=IntervalTree()
        sv_counter_dict[chrom]=0
        for rec in vcf_in.fetch():
            svrec = SVRecord_generic(rec, caller)
            startCI = abs(svrec.cipos[0]) + svrec.cipos[1]
            endCI = abs(svrec.ciend[0]) + svrec.ciend[1]
            if startCI <= 200 and endCI <= 200 and svrec.chrom == chrom and svrec.svtype == "DEL" and svrec.start != svrec.end:
                try:
                    interval_dict[chrom][svrec.start+svrec.cipos[0]:svrec.end+svrec.ciend[1]]=(svrec.start+svrec.cipos[0],svrec.end+svrec.ciend[1])
                    sv_counter_dict[chrom]+=1
                except ValueError:
                        print(rec)

#out.close()

genomepos_dict={}

with open("/home/cog/smehrem/MinorResearchInternship/VCF/gridss.sym.vcf") as ingen:
    for line in ingen:
        line=line.strip()
        if "contig" in line:
            gen_pos=re.findall(r'ID=(\S*),|length=(\d*)',line)
            genomepos_dict[gen_pos[0][0]]=int(gen_pos[1][1])
startpos=1
genome_intervals={}

for i in range(0, len(chr_list)):
        genome_intervals[chr_list[i]] = IntervalTree()
        genome_intervals[chr_list[i]][1:genomepos_dict[chr_list[i]]] = (startpos, genomepos_dict[chr_list[i]])



rand_windowpairs={}

for key in sv_counter_dict:
    rand_windowpairs[key]=set()
    inv=genome_intervals[key]
    strt=inv.begin()
    end=inv.end()
    og_tree=interval_dict[key]
    while len(rand_windowpairs[key]) != sv_counter_dict[key]:
        a=random.randrange(strt,end)
        b=random.randrange(strt,end)
        rand_strt=IntervalTree()
        rand_end=IntervalTree()
        if a > b:
            rand_strt=Interval(b-100,b+100)
            rand_end=Interval(a-100,a+100)
        else:
            rand_strt=Interval(a-100,a+100)
            rand_end=Interval(b-100,b+100)
        if not og_tree.overlaps(rand_strt.begin, rand_end.end):
            rand_windowpairs[key].add((a, b))
    print(key+" processed.")

with open("Negative_NoBP_NoBP.txt",'w') as out:
    for key in rand_windowpairs:
        for element in rand_windowpairs[key]:
            out.write("\t".join([key,str(element[0]),str(element[1])])+"\n")
