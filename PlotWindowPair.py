import numpy as np
from collections import Counter


data = []
label=[]

svtype = 'DEL'
data_file = "gridss_DEL_NA12878_1_10482300_10483950"
with gzip.GzipFile(data_file, "rb") as f:
        data_mat = np.load(f)
        data.extend(data_mat)

data = np.array(data)

# Fill labels for legend

label[0] = "coverage"
label[1] = "#left clipped reads"
label[2] = "#right clipped reads"
label[3] = "INV_before"
label[4] = "INV_after"
label[5] = "DUP_before"
label[6] = "DUP_after"
label[7] = "TRA_opposite"
label[8] = "TRA_same"

i = 9
for direction in ['F', 'R']:
    for clipped in ['L', 'R', 'U']:
        for value in ['sum', 'num', 'median']:
            label[i] = direction + '_' + clipped + '_CR_' + value
            i = i + 1

label[i] = "#left split reads"
i = i + 1
label[i] = "#right split reads"
i = i + 1

for clipped in ['L', 'R']:
    for value in ['sum', 'num', 'median']:
        label[i] = clipped + '_SR_' + value
        i = i + 1

label[i] = "GC"
i = i + 1
label[i] = "Mappability"
i = i + 1
label[i] = "One_hot_Ncoding"



