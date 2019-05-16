cmdTrace.c(730):ERROR:104: 'module' is an unrecognized subcommand
cmdModule.c(423):ERROR:104: 'module' is an unrecognized subcommand
Currently Loaded Modulefiles:
  1) modules/3.2.10.6
  2) cray-mpich/7.7.2
  3) slurm/17.11.12.cscs-1
  4) xalt/daint-2016.11
  5) daint-gpu
  6) cray-hdf5-parallel/1.10.2.0
  7) cudatoolkit/9.1.85_3.18-6.0.7.0_5.1__g2eb7c52
  8) cray-python/2.7.15.1
  9) cray-python/3.6.5.1
 10) gcc/6.2.0
 11) craype-haswell
 12) craype-network-aries
 13) craype/2.5.15
 14) cray-libsci/18.07.1
 15) udreg/2.3.2-6.0.7.0_33.18__g5196236.ari
 16) ugni/6.0.14.0-6.0.7.0_23.1__gea11d3d.ari
 17) pmi/5.0.14
 18) dmapp/7.1.1-6.0.7.0_34.3__g5a674e0.ari
 19) gni-headers/5.0.12.0-6.0.7.0_24.1__g3b1768f.ari
 20) xpmem/2.2.15-6.0.7.1_5.10__g7549d06.ari
 21) job/2.2.3-6.0.7.0_44.1__g6c4e934.ari
 22) dvs/2.7_2.2.113-6.0.7.1_7.6__g1bbc03e
 23) alps/6.6.43-6.0.7.0_26.4__ga796da3.ari
 24) rca/2.2.18-6.0.7.0_33.3__g2aa4f39.ari
 25) atp/2.1.2
 26) perftools-base/7.0.2
 27) PrgEnv-gnu/6.0.4
 28) CrayGNU/.18.08
 29) cuDNN/.7.1.4-cuda-9.1
 30) PCRE/.8.42-CrayGNU-18.08
 31) SWIG/.3.0.12-CrayGNU-18.08-python3
 32) PyExtensions/3.6.5.1-CrayGNU-18.08
 33) protobuf-core/.3.6.0-CrayGNU-18.08
 34) protobuf/.3.6.0-CrayGNU-18.08-python3
 35) backports.weakref/.1.0.post1-CrayGNU-18.08-python3
 36) Werkzeug/.0.14.1-CrayGNU-18.08-python3
 37) NCCL/2.2.13-cuda-9.1
 38) absl/.0.2.2-CrayGNU-18.08-python3
 39) dask/.0.18.1-CrayGNU-18.08-python3
 40) Keras_Applications/1.0.6-CrayGNU-18.08-python3
 41) Keras_Preprocessing/1.0.5-CrayGNU-18.08-python3
 42) TensorFlow/1.11.0-CrayGNU-18.08-cuda-9.1-python3
 43) IPython/.5.7.0-CrayGNU-18.08-python3
 44) jupyter/1.0.0-CrayGNU-18.08
 45) Base-opts/2.4.135-6.0.7.0_38.1__g718f891.ari
2019-01-21 16:48:35.157677: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1411] Found device 0 with properties: 
name: Tesla P100-PCIE-16GB major: 6 minor: 0 memoryClockRate(GHz): 1.3285
pciBusID: 0000:02:00.0
totalMemory: 15.90GiB freeMemory: 15.61GiB
2019-01-21 16:48:35.157705: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1490] Adding visible gpu devices: 0
2019-01-21 16:48:35.854341: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-01-21 16:48:35.854365: I tensorflow/core/common_runtime/gpu/gpu_device.cc:977]      0 
2019-01-21 16:48:35.854371: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990] 0:   N 
2019-01-21 16:48:35.855672: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1103] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 15127 MB memory) -> physical GPU (device: 0, name: Tesla P100-PCIE-16GB, pci bus id: 0000:02:00.0, compute capability: 6.0)
srun: Job step aborted: Waiting up to 32 seconds for job step to finish.
slurmstepd: error: *** STEP 11478979.0 ON nid04886 CANCELLED AT 2019-01-21T16:49:49 DUE TO TIME LIMIT ***
slurmstepd: error: *** JOB 11478979 ON nid04886 CANCELLED AT 2019-01-21T16:49:49 DUE TO TIME LIMIT ***
srun: got SIGCONT
srun: forcing job termination
