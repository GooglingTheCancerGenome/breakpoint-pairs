import os
import re
from pysam import
from matplotlib import pyplot as plt

from labels import SVRecord_generic

windowsizes=[]
lost_SVs=0
total_SVs=0
for vcf_file in os.listdir('../../../VCF'):
    vcf_in = VariantFile('../../../VCF/'+vcf_file, 'r')
    caller=re.findall(r'^\w*',vcf_file)

    for rec in vcf_in.fetch():
        total_SVs+=1
        svrec = SVRecord_generic(rec, caller)
        startCI=abs(svrec.cipos[0]) + svrec.cipos[1]
        endCI = abs(svrec.ciend[0]) + svrec.ciend[1]
        windowsizes+=[startCI,endCI]
        if startCI > 200 or endCI > 200:
            lost_SVs+=1
    vcf_in.close()

print ('\n'+str(round(lost_SVs/total_SVs*100,2))+'%.')



n, bins, patches = plt.hist(windowsizes, bins=int(len(set(windowsizes))*0.07),log=True)
plt.xlabel('Breakpoint window sizes')
plt.ylabel('Counts')
plt.title('Windowsizes of called SVs from Delly, Lumpy, Gridss and Manta\n')
plt.show()
plt.close()
