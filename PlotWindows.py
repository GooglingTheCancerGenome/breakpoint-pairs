import numpy as np
import os
import gzip
import matplotlib.pyplot as plt

data = []
data_file = os.path.join("/home/cog/smehrem/breakpoint-pairs/bla.npy.gz")
with gzip.GzipFile(data_file, "rb") as f:
    data_mat = np.load(f)
    data.extend(data_mat)

data = np.array(data)
data = np.swapaxes(data, 1, 2)
print(data.shape)
number_channels = data.shape[1]

with open("Channel_Labels.txt", "r") as inlab:
    label = [line.strip() for line in inlab]
print(label)

fig = plt.figure(figsize=(6, 4))
for i in range(7, 8):
    for j in range(number_channels-1, -1, -1):
        shift = 0
        start = 0
        if sum(data[i][j]) != 0:
            X_win = (data[i][j]-min(data[i][j]))/max(data[i][j])

        else:
            X_win = data[i][j]

        Z = [x + j+1 for x in X_win]
        plt.plot(Z, label=label[j], linewidth=0.9)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop={'size': 5})
        plt.yticks(range(0, len(label)+1, 1))
        plt.tick_params(axis='both', which='major', labelsize=5)
        plt.axvline(x=200, color='r', linewidth=0.05, alpha=0.5)
        plt.axvline(x=209, color='r', linewidth=0.05, alpha=0.5)

        #if j == 1 or j == 2:
            #print(j)
            #print(X_win)
            #print(Z)
        #plt.grid(which='major', axis='y')
    plt.savefig("DEL_Gridss_ChannelPlot.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
