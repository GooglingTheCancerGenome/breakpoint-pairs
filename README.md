# Breakpoint-pairs
Minor research internship of Sarah Mehrem.

The task of this project was to optimise DeepSV in regards to hyperparameters to increase its performance, and afterwards to explore its theoretical capabilities as an SV caller.
For that I used existing calls of GRIDSS, Manta, Lumpy and Delly. I trained and tested the DeepSV CNN on these variants (per caller) and investigated aspects such as perfromance with different
negative sets, channel importance and also compared it to the perfromance of the traditional SV callers using the gold-standard callset:

ftp://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/technical/svclassify_Manuscript/Supplementary_Information/Personalis_1000_Genomes_deduplicated_deletions.bed


The internship report can be found here:

https://docs.google.com/document/d/10gqmzAgrfbinHVsMw2v9aNGe5i7_1rCQ1tvS-QR3MYA/edit?usp=sharing


# Gerenal workflow

I started off by creating the test and training data per caller. For that I generated genomewide channel data of the NA12878 once.


The negative set genomic coordinates were generated using:

MakeNegative_NoBP_NoBP.py
MakeNegative_CR_CR.py
MakeNegative_CR_TrueSV.py
MakeNegative_TrueSV_TrueSV.py 

We used window pairs (breakpoint-breakpoint) of channel windows as input. These were created using the scripts:

MakeWindowPairs_genomewide.sh
MakeWindowPairs_genomewide_negative.sh	

and the VCF files of the respective callers and the genomic coordinates of the negative sets.



Final training and test sets were created using:

MakeTrainingData.py



Next, to train the DeepSV CNN this script is used:

OC_cross_validation_dropout.py

In case of calculating channel importance, the above mentioned script was modified into:

OC_cross_validation_dropout_zeroed.py



