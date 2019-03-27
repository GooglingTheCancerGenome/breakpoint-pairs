import numpy as np
import gzip

with gzip.GzipFile("../MinorResearchInternship/NA12878/TrainingData/NA12878_Mills2011_nanosv_labels_binary.npy.gz", "rb") as f:
    windowpairs = np.load(f)

print(windowpairs)