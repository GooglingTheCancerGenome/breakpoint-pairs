import os
import re
from pysam import VariantFile
from matplotlib import pyplot as plt
from labels import SVRecord_generic
from collections import defaultdict


windowsizes = []
windowsizes_by_caller = {}
windowsizes_by_SVType = defaultdict(list)
SVCount_bytype = defaultdict(int)

callers = []
lost_SVs = 0
total_SVs = 0

for vcf_file in os.listdir('../MinorResearchInternship/VCF'):
    vcf_in = VariantFile('../MinorResearchInternship/VCF/'+vcf_file, 'r')
    caller = re.findall(r'^\w*', vcf_file)
    callers += [caller[0]]
    windowsizes_by_caller[caller[0]] = {"CI_sizes": [], "Lost_SVs": 0, "Total_SVs": 0}
    for rec in vcf_in.fetch():
        total_SVs += 1
        windowsizes_by_caller[caller[0]]["Total_SVs"] += 1
        svrec = SVRecord_generic(rec, caller[0])
        SVCount_bytype[svrec.svtype] += 1
        startCI = abs(svrec.cipos[0]) + svrec.cipos[1]
        endCI = abs(svrec.ciend[0]) + svrec.ciend[1]
        windowsizes += [startCI, endCI]
        windowsizes_by_caller[caller[0]]["CI_sizes"] += [startCI, endCI]
        windowsizes_by_SVType[svrec.svtype] += [startCI, endCI]
        if startCI > 200 or endCI > 200:
            lost_SVs += 1
            windowsizes_by_caller[caller[0]]["Lost_SVs"] += 1
            SVCount_bytype[svrec.svtype+"_lost"] += 1

    vcf_in.close()

fraction_lostSVs_allcallers = round(lost_SVs/total_SVs, 4)

bins=list(range(0, max(windowsizes),50))

plt.hist(windowsizes, bins=bins,log=True, edgecolor='black', linewidth=0.5, zorder=3, color="seagreen")
plt.xlabel('Breakpoint interval sizes [bp]')
plt.ylabel('Counts')
plt.title('Confidence interval distribution of called SVs\nfrom Delly, Lumpy, Gridss and Manta\n')
plt.text(250,14000,str(fraction_lostSVs_allcallers*100)+"% with CI >200bp.", fontsize= 7)
plt.axvline(x=200,color='r',zorder=3)
plt.savefig('ConfidenceIntervalDistr_allCallers.pdf', format="pdf", dpi=300, bbox_inches="tight")
plt.grid(True, which="both",axis="y", alpha=0.5, zorder=0)
plt.show()
plt.tight_layout()
plt.close()

fig, axs = plt.subplots(2, 2, figsize=(6, 6), sharex="all", sharey="all")
axs = axs.ravel()
fig.suptitle("Confidence interval distribution by caller")
fig.tight_layout()

fig.subplots_adjust(top=0.88)
fig.subplots_adjust(hspace=0.25)
fig.subplots_adjust(wspace=0.25)


for i in range(len(callers)):
    axs[i].hist(windowsizes_by_caller[callers[i]]["CI_sizes"], bins=bins, log=True, edgecolor='black', linewidth=0.5, zorder=3, color="seagreen")
    axs[i].xaxis.set_tick_params(labelbottom=True)
    axs[i].set_title(callers[i]+" ("+str(round(windowsizes_by_caller[callers[i]]["Total_SVs"]/total_SVs*100,2))+"%)")
    axs[i].text(600, 14000, str(round(windowsizes_by_caller[callers[i]]["Lost_SVs"]/lost_SVs*100, 2))+"% of lost SVs\n(CI >200bp)", fontsize=7)
    axs[i].axvline(x=200, color='r', zorder=3)
    axs[i].yaxis.set_tick_params(labelleft=True)
    axs[i].xaxis.set_tick_params(labelbottom=True)
    axs[i].grid(True, which="major", axis="y", alpha=0.5, zorder=0)

axs[2].set_xlabel("Interval size [bp]")
axs[3].set_xlabel("Interval size [bp]")
axs[0].set_ylabel("Counts")
axs[2].set_ylabel("Counts")
plt.savefig("Confidence_interval_distribution_by_caller.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


svtypes = list(windowsizes_by_SVType.keys())

fig, axs = plt.subplots(2, 3, figsize=(15, 6))
axs = axs.ravel()
fig.delaxes(ax=axs[5])
fig.tight_layout()
fig.subplots_adjust(top=0.88)
fig.subplots_adjust(hspace=0.33)
fig.subplots_adjust(wspace=0.33)
fig.suptitle("Confidence interval distribution by SV type")


for i in range(len(svtypes)):
    axs[i].hist(windowsizes_by_SVType[svtypes[i]], log=True, bins=bins, edgecolor='black', linewidth=0.5, zorder=3, color="seagreen")
    axs[i].set_ylim([1, 70000])
    axs[i].set_xlim([1, 1450])
    axs[i].set_title(svtypes[i]+" ("+str(round(SVCount_bytype[svtypes[i]]/total_SVs*100, 2))+"%)")
    axs[i].axvline(x=200, color='r', zorder=3)
    axs[i].grid(True, which="major", axis="y", alpha=0.5, zorder=0)
    axs[i].text(600, 14000, str(round(SVCount_bytype[svtypes[i]+"_lost"] / lost_SVs * 100, 2)) + "% of lost SVs\n(CI >200bp)", fontsize=7)

axs[2].set_xlabel("Interval size [bp]")
axs[3].set_xlabel("Interval size [bp]")
axs[4].set_xlabel("Interval size [bp]")
axs[0].set_ylabel("Counts")
axs[3].set_ylabel("Counts")

plt.savefig("Confidence_interval_distribution_by_Type.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
