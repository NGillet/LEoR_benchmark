# LEoR_benchmark

Convolutionnal Neural Network benchmark code on GPU and low memory machine.
Used to time the Epoch to compare machine.
The Network is the same as https://arxiv.org/pdf/1805.02699.pdf.
It should be able to reproduce those results.

It need the database : LC_SLICE10_px100_2200_N10000_randICs_train.hdf5

The memory management is set in lightcone_functions.py -> DataGenerator -> self.n_pre_load. 80000 is the whole database load in memory. Fiducial low memory node set it to 8000. 
