# LEoR_benchmark

Convolutionnal Neural Network benchmark code on GPU and low memory machine.
Used to time the Epoch to compare machine.
The Network is the same as https://arxiv.org/pdf/1805.02699.pdf.
It should be able to reproduce those results.

It need the database : LC_SLICE10_px100_2200_N10000_randICs_train.hdf5 (~70GB). Ask me for the data.

The memory management is set in lightcone_functions.py -> DataGenerator -> self.n_pre_load. 80000 is the whole database load in memory. Fiducial low memory node set it to 8000. 

For each execution, the loading time is estimated (from a print inside the loading function), the total time of the execution of the epoch (from the print of keras execution bar). Therefore the computation time can be estimated as the difference of the two previous. 
In the case of low memory machine/node, the database is chunk and read peace by peace during the learning of one epoch. The loading time is therefore the time to load a chunck of data. To extract the computation timing, this loading time has to be multiply by the number of chunck. 

The code as been runing on :

- **Nefertem** : CPU 
    - load data timing : 
        - data loaded from Amphora (through network? why is it so long)
        - low memory mode : ...
        - large memory mode : 18min / 80000LC
    - Total epoch timing : 
        - low memory mode : ...
        - large memory mode : 90min / epoch
    - **Computation timing : 70min / epoch**

- **Thanatos** : GPU - Titan V
    - load data timing : 
        - data loaded from the computer hardrive
        - low memory mode : 42s / 8000LC (1/10)
    - Total epoch timing : 
        - low memory mode : 645s / epoch
    - **Computation timing : 225s / epoch**
    
- **PizDaint** : GPU - Tesla P100
    - load data timing : 
        - data loaded from the SCRATCH repo
        - low memory mode : ~50s (1/10) (THIS IS A CRUDE REMENBER...)
    - epoch timing : 
        - low memory mode : 720s / epoch
    - **Computation timing : ~220 / epoch**

- **DELL test machine** : GPU - Tesla P100
    - load data timing : 
        - data loaded from their own mounted storage (no info)
        - low memory mode : 6.2s / 8000LC (1/10)
        - large memory mode : 75s / 80000LC (1)
    - epoch timing : 
        - low memory mode : 140s / epoch
        - large memory mode : 170s / epoch
    - **Computation timing** : 
        - **low memory mode : 80s / epoch**
        - **large memory mode : 95s / epoch**
        
All the timing are highly approximate (i) in most of the machine I do not crontrol that other job might have step in during the execution, especially for Nefertem and Thanatos. (ii) I just read the time, I stored some of the benchmark, but I did not extract and average the timing.
