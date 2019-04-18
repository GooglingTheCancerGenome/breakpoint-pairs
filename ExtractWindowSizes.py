import os
import re
from pysam import VariantFile
from matplotlib import pyplot as plt
from labels import SVRecord_generic
from collections import defaultdict
import brewer2mpl
import matplotlib.patches as patches


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
    windowsizes_by_caller[caller[0]] = {"CI_sizes": {"Start": {"DEL": [],
                                                               "INS": [],
                                                               "BND": [],
                                                               "INV": [],
                                                               "DUP": []},
                                                     "End": {"DEL": [],
                                                             "INS": [],
                                                             "BND": [],
                                                             "INV": [],
                                                             "DUP": []},
                                                     "All": []},
                                        "Lost_SVs": 0,
                                        "Total_SVs": 0}
    for rec in vcf_in.fetch():
        total_SVs += 1
        windowsizes_by_caller[caller[0]]["Total_SVs"] += 1
        svrec = SVRecord_generic(rec, caller[0])
        SVCount_bytype[svrec.svtype] += 1
        startCI = abs(svrec.cipos[0]) + svrec.cipos[1]
        endCI = abs(svrec.ciend[0]) + svrec.ciend[1]
        windowsizes += [startCI, endCI]
        windowsizes_by_caller[caller[0]]["CI_sizes"]["All"] += [startCI, endCI]
        windowsizes_by_caller[caller[0]]["CI_sizes"]["Start"][svrec.svtype] += [startCI]
        windowsizes_by_caller[caller[0]]["CI_sizes"]["End"][svrec.svtype] += [endCI]
        windowsizes_by_SVType[svrec.svtype] += [startCI, endCI]
        if startCI > 200 or endCI > 200:
            lost_SVs += 1
            windowsizes_by_caller[caller[0]]["Lost_SVs"] += 1
            SVCount_bytype[svrec.svtype+"_lost"] += 1
        elif svrec.svtype == "DEL":
            print(svrec.chrom+":"+str(svrec.start)+"-"+str(svrec.end))

    vcf_in.close()

fraction_lostSVs_allcallers = round(lost_SVs/total_SVs, 4)

bins=list(range(0, max(windowsizes),50))

plt.hist(windowsizes, bins=bins,log=True, edgecolor='black', linewidth=0.5, zorder=3, color="seagreen")
plt.xlabel('Breakpoint interval sizes [bp]')
plt.ylabel('Counts [log]')
plt.title('Confidence interval distribution of called SVs\nfrom Delly, Lumpy, Gridss and Manta\n')
plt.text(250,14000,str(fraction_lostSVs_allcallers*100)+"% with CI >200bp.", fontsize= 7)
plt.axvline(x=200,color='r',zorder=3)
plt.savefig('ConfidenceIntervalDistr_allCallers.pdf', format="pdf", dpi=300, bbox_inches="tight")
plt.grid(True, which="both",axis="y", alpha=0.5, zorder=0)
#plt.show()
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
    axs[i].hist(windowsizes_by_caller[callers[i]]["CI_sizes"]["All"], bins=bins, log=True, edgecolor='black', linewidth=0.5, zorder=3, color="seagreen")
    axs[i].xaxis.set_tick_params(labelbottom=True)
    axs[i].set_title(callers[i]+" ("+str(round(windowsizes_by_caller[callers[i]]["Total_SVs"]/total_SVs*100,2))+"%)")
    axs[i].text(600, 14000, str(round(windowsizes_by_caller[callers[i]]["Lost_SVs"]/lost_SVs*100, 2))+"% of lost SVs\n(CI >200bp)", fontsize=7)
    axs[i].axvline(x=200, color='r', zorder=3)
    axs[i].yaxis.set_tick_params(labelleft=True)
    axs[i].xaxis.set_tick_params(labelbottom=True)
    axs[i].grid(True, which="major", axis="y", alpha=0.5, zorder=0)

axs[2].set_xlabel("Interval size [bp]")
axs[3].set_xlabel("Interval size [bp]")
axs[0].set_ylabel("Counts [log]")
axs[2].set_ylabel("Counts [log]")
plt.savefig("Confidence_interval_distribution_by_caller.png", format="png", dpi=300, bbox_inches="tight")
#plt.show()
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
axs[0].set_ylabel("Counts [log]")
axs[3].set_ylabel("Counts [log]")

plt.savefig("Confidence_interval_distribution_by_Type.png", format="png", dpi=300, bbox_inches="tight")
#plt.show()
plt.close()


fig, axs = plt.subplots(4, 2, figsize=(8, 8))
axs = axs.ravel()
fig.suptitle("Breakpoint confidence interval distribution")
fig.tight_layout()

fig.subplots_adjust(top=0.88)
fig.subplots_adjust(hspace=0.60)
fig.subplots_adjust(wspace=0.0001)

colors = brewer2mpl.get_map('Set2', 'Qualitative', 5).mpl_colors

counter = 0
for i in range(0, len(callers)*2, 2):
    axs[i].hist([windowsizes_by_caller[callers[counter]]["CI_sizes"]["Start"]["INV"],
                 windowsizes_by_caller[callers[counter]]["CI_sizes"]["Start"]["DEL"],
                 windowsizes_by_caller[callers[counter]]["CI_sizes"]["Start"]["DUP"],
                 windowsizes_by_caller[callers[counter]]["CI_sizes"]["Start"]["BND"],
                 windowsizes_by_caller[callers[counter]]["CI_sizes"]["Start"]["INS"]],
                bins=bins, log=True, edgecolor='black', linewidth=0.5,  zorder=3, stacked=True, histtype="bar", label=["INV", "DEL", "DUP", "BND", "INS"], color=colors)
    axs[i].xaxis.set_tick_params(labelbottom=True)
    #axs[i].set_title(callers[counter]+" ("+str(round(windowsizes_by_caller[callers[counter]]["Total_SVs"]/total_SVs*100,2))+"%)")
    axs[i].axvline(x=200, color='r', zorder=3)
    axs[i].yaxis.set_tick_params(labelleft=True)
    axs[i].xaxis.set_tick_params(labelbottom=True)
    axs[i].grid(True, which="major", axis="y", alpha=0.5, zorder=0)
    axs[i].set_ylim([1, 80000])
    axs[i].set_xlim([1, 1450])
    axs[i].set_ylabel("Counts [log]")
    axs[i].text(1200, 60000, callers[counter], style='oblique',
                    bbox={'facecolor': 'lightgrey', 'alpha': 0.99, 'pad': 10})

    axs[i+1].hist([windowsizes_by_caller[callers[counter]]["CI_sizes"]["End"]["INV"],
                 windowsizes_by_caller[callers[counter]]["CI_sizes"]["End"]["DEL"],
                 windowsizes_by_caller[callers[counter]]["CI_sizes"]["End"]["DUP"],
                 windowsizes_by_caller[callers[counter]]["CI_sizes"]["End"]["BND"],
                 windowsizes_by_caller[callers[counter]]["CI_sizes"]["End"]["INS"]],
                bins=bins, log=True, edgecolor='black', linewidth=0.5, zorder=3, stacked=True, histtype="bar",
                label=["INV", "DEL", "DUP", "BND", "INS"], color=colors)
    axs[i+1].xaxis.set_tick_params(labelbottom=True)
    axs[i+1].set_ylim([1, 80000])
    axs[i+1].set_xlim([1, 1450])
    axs[i+1].axvline(x=200, color='r', zorder=3)
    axs[i+1].yaxis.set_ticklabels([])
    axs[i+1].yaxis.set_ticks_position('none')
    axs[i+1].xaxis.set_tick_params(labelbottom=True)
    axs[i+1].grid(True, which="major", axis="y", alpha=0.5, zorder=0)

    counter += 1

axs[6].set_xlabel("Interval size [bp]")
axs[7].set_xlabel("Interval size [bp]")
axs[1].legend(edgecolor="black")
axs[0].set_title("Start")
axs[1].set_title("End")

plt.savefig("Confidence_interval_distribution_by_caller_type_start_end.png", format="png", dpi=300, bbox_inches="tight")
#plt.show()
plt.close()

