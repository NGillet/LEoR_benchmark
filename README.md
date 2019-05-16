# LEoR_benchmark

Convolutionnal Neural Network benchmark code on GPU and low memory machine.
Used to time the Epoch to compare machine.
The Network is the same as https://arxiv.org/pdf/1805.02699.pdf.
It should be able to reproduce those results.

It need the database : LC_SLICE10_px100_2200_N10000_randICs_train.hdf5 (~70GB)

The memory management is set in lightcone_functions.py -> DataGenerator -> self.n_pre_load. 80000 is the whole database load in memory. Fiducial low memory node set it to 8000. 

The code as been runing on :

- Nefertem : CPU 
    - load data timing : 
        - data loaded from Amphora (through network?)
        - low memory mode : ...
        - large memory mode : 18min / 80000LC
    - epoch timing : 
        - low memory mode : ...
        - large memory mode : ~90min / epoch

- Thanatos : GPU - Titan V
    - load data timing : 
        - data loaded from the computer hardrive
        - low memory mode : ~42s / 8000LC 
    - epoch timing : 
        - low memory mode : 645s / epoch
    
- PizDaint : GPU - Tesla P100
    - load data timing : 
        - data loaded from the SCRATCH repo
        - low memory mode : ... (I do not remember...)
    - epoch timing : 
        - low memory mode : 720s / epoch

- DELL test machine : GPU - Tesla P100
    - load data timing : 
        - data loaded from their own mounted storage (no info)
        - low memory mode : 6.2s / 8000LC 
        - large memory mode : 75s / 80000LC
    - epoch timing : 
        - low memory mode : 140s / epoch
        - large memory mode : 170s / epoch
        
        
