import numpy as np
import os
import gzip
import matplotlib.pyplot as plt

data = []
data_file = os.path.join("/home/cog/smehrem/breakpoint-pairs/manta_DEL_NA12878_1_10482300_10483950.npy.gz")
with gzip.GzipFile(data_file, "rb") as f:
    data_mat = np.load(f)
    data.extend(data_mat)

data = np.array(data)
data = np.swapaxes(data, 1, 2)
number_channels = data.shape[1]
label = list(range(1, number_channels+1))

for j in range(number_channels-1, -1, -1):
    shift = 0
    start = 0
    if sum(data[0][j]) != 0:
        X_win = (data[0][j]-min(data[0][j]))/max(data[0][j])

    else:
        X_win = data[0][j]

    Z = [start + shift + x + j for x in X_win]
    plt.ylim([0, number_channels+5])
    plt.plot(Z, label=label[j], linewidth=2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop={'size': 10})
plt.show()
plt.close()
