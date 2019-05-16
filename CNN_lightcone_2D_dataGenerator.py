import numpy as np
from time import time

import sys, argparse, textwrap

#############
### INPUT ###
#############

#######################################################
### FIDUCIAL :                                      ###
### param_all4_2D_smallFilter_1batchNorm_multiSlice ###
### 10 slices per training                          ###
### dataGenerator : for low memory GPU              ###
#######################################################

from param_all4_2D_smallFilter_1batchNorm_multiSlice_dataGenerator import *

####################
### FOR CPU ONLY ###
####################

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

#############
### KERAS ###
#############

from keras import backend as K
print( K.tensorflow_backend._get_available_gpus() )
from tensorflow.python.client import device_lib
print( device_lib.list_local_devices() )
    
######################
### CODE PARAMETER ###
######################

### INITIATE RANDOM STATE
np.random.RandomState( np.random.seed(RandomSeed) )

print( 'Param          :', paramName[paramNum] )
print( 'file           :', model_file )
print( 'RandomSeed     :', RandomSeed )
print( 'trainSize      :', trainSize )
print( 'LHS            :', LHS )
if( LHS ):
    print( 'Nbins_LHS      :', Nbins_LHS )
print( 'epochs         :', epochs )
print( 'batch_size     :', batch_size )
print( 'DATABASE       :', DATABASE )
print( 'validation     :', validation )
print( 'all4           :', all4           )
print( 'reduce_LC      :', reduce_LC      )
print( 'substract_mean :', substract_mean )
print( 'apply_gauss    :', apply_gauss    )
print( 'reduce_CNN     :', reduce_CNN    )
print( 'use_dropout    :', use_dropout    )
print( 'CNN loss       :', loss )
print( 'CNN optimizer  :', optimizer )
print( 'LR factor      :', factor)
print( 'LR patience    :', patience)

### Variables not define in all parameter file!!
try:
    print( 'LeackyRelu     :',LeackyRelu_alpha )
except:
    LeackyRelu_alpha = 0
    print( 'LeackyRelu     :',LeackyRelu_alpha )
    
try:
    print( 'batchNorm      :',batchNorm )
except:
    batchNorm = False
    print( 'batchNorm      :',batchNorm )
    
try:
    print( 'FirstbatchNorm :',FirstbatchNorm )
except:
    FirstbatchNorm = False
    print( 'FirstbatchNorm :',FirstbatchNorm )
     
try:
    print( 'Nfilter1       :',Nfilter1 )
    print( 'Nfilter2       :',Nfilter2 )
    print( 'Nfilter3       :',Nfilter3 )
except:
    Nfilter1 = 16 
    Nfilter2 = 32 
    Nfilter3 = 64 
    print( 'Nfilter1       :',Nfilter1 )
    print( 'Nfilter2       :',Nfilter2 )
    print( 'Nfilter3       :',Nfilter3 )
    


epochs = 10

HDF5 = True
with open( '/etc/hostname' ) as f:
    hostname = f.read()[:-1]
print( "HOSTNAME : ", hostname )
if( hostname=='thanatos' ):
    DATA_DIR = '/media/yqin/81614149-2ed8-4c76-82e1-c46763d086fa/ngillet/LC_SLICE10_px100_2200_N10000_randICs_train.hdf5'

elif( hostname=='c4140-6.hpc.local' ):
    DATA_DIR = '/u1/nicolas.gillet/LC_SLICE10_px100_2200_N10000_randICs_train.hdf5'
else: ### daint
    DATA_DIR = '/scratch/snx3000/ngillet/LC_SLICE10_px100_2200_N10000_randICs_train.hdf5'

    
from lightcone_functions import DataGenerator 
# Data Generator Parameters
if not(HDF5):
    DATA_DIR='/media/yqin/81614149-2ed8-4c76-82e1-c46763d086fa/ngillet/LC_SLICE10_px100_2200_N10000_randICs/train/'
params_train = {'Ndata'      : 80000,
                'batch_size' : batch_size, 
                'dim'        : (100,2200), 
                'n_channels' : 1,
                'n_params'   : 4, 
                'shuffle'    : True, 
                'DATA_DIR'   : DATA_DIR, 
                'HDF5'       : HDF5, 
                'which_set'  : 'train',
                'pre_loaded_data' : True, 
               }
if not(HDF5):
    DATA_DIR='/media/yqin/81614149-2ed8-4c76-82e1-c46763d086fa/ngillet/LC_SLICE10_px100_2200_N10000_randICs/validation/'
params_valid = {'Ndata'      : 1000,
                'batch_size' : batch_size, 
                'dim'        : (100,2200), 
                'n_channels' : 1,
                'n_params'   : 4, 
                'shuffle'    : False, 
                'DATA_DIR'   : DATA_DIR, 
                #'pre_loaded_data' : True, 
                'HDF5'       : HDF5, 
                'which_set'  : 'validation',
               }
if not(HDF5):
    DATA_DIR='/media/yqin/81614149-2ed8-4c76-82e1-c46763d086fa/ngillet/LC_SLICE10_px100_2200_N10000_randICs/test/'
params_test  = {'Ndata'      : 1000,
                'batch_size' : batch_size, 
                'dim'        : (100,2200), 
                'n_channels' : 1,
                'n_params'   : 4, 
                'shuffle'    : False, 
                'DATA_DIR'   : DATA_DIR, 
                #'pre_loaded_data' : True, 
                'HDF5'       : HDF5, 
                'which_set'  : 'test',
               }

# Generators
training_generator   = DataGenerator( **params_train)
validation_generator = DataGenerator( **params_valid)
# test_generator       = DataGenerator( **params_test)

### DATA SHAPE
### ADAPTE IT BY HAND! BE CAREFULL 
K.set_image_data_format('channels_last')
input_shape = (100,2200,1)
print('input shape : ',input_shape)

padding = 'valid' ### 'same' or 'valid
filter_size = (10,10)
activation = 'relu' ### 'linear' 'relu'
use_bias=True

from keras.layers import Activation
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.models import Sequential, Model
model = Sequential()

### CONV 1
model.add( Convolution2D( Nfilter1, filter_size, activation=activation, 
                          input_shape=input_shape, name='Conv-1', padding=padding, use_bias=use_bias ) )

### MAXPOOL 1
model.add( MaxPooling2D( pool_size=(2,2), name='Pool-1' ) )

### CONV 2
model.add( Convolution2D( Nfilter2, filter_size, activation=activation, 
                          name='Conv-2', padding=padding, use_bias=use_bias ) )

### MAXPOOL 2
model.add( MaxPooling2D( pool_size=(2,2), name='Pool-2' ) )

### FLATTEN
model.add( Flatten( name='Flat' ) )
if use_dropout: ### AFTER TEST THIS ONE AT 0.2 WORK WELL
    model.add( Dropout(use_dropout) )

### DENSE 1
model.add( Dense( Nfilter3, activation=activation, name='Dense-1', use_bias=use_bias ) )

### BATCHNORM
if( batchNorm or FirstbatchNorm ):
    model.add( BatchNormalization() )
if( ( batchNorm or FirstbatchNorm ) and not(LeackyRelu_alpha) ):
    model.add( Activation('relu') )

### DENSE 2
model.add( Dense( Nfilter2, activation=activation, name='Dense-2', use_bias=use_bias ) )

### DENSE 3
model.add( Dense( Nfilter1, activation=activation, name='Dense-3', use_bias=use_bias ) )

### DENSE OUT
if all4:
    model.add( Dense( 4, activation='linear', name='Out' ) )
else:
    model.add( Dense( 1, activation='linear', name='Out' ) )
   
model.summary(line_length=120) 

from keras.layers import Input
input_layer1 = Input( (100,2200,1) )
output_layer1 = model( input_layer1 )
input_layer2 = Input( (100,2200,1) )
output_layer2 = model( input_layer2 )
multi_model = Model( [input_layer1,input_layer2], [output_layer1,output_layer2] )

######################
### LEARNING PHASE ###
######################

### callback list: list on functions call at the end of epochs
callbacks_list=[]

### Learning Rate on the fly
if( 1 ):
    from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
    #lrate = LearningRateScheduler( step_decay )
    lrate = ReduceLROnPlateau( monitor='loss', factor=factor, patience=patience )
    callbacks_list.append( lrate )

### to print the Learning Rate
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
class LR_tracer(Callback):
    def on_epoch_begin(self, epoch, logs={}):
        lr = K.eval( self.model.optimizer.lr )
        print( ' LR: %.10f '%(lr) )
callbacks_list.append( LR_tracer() )

### R2 coefficient
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

### STOP when it stop to learn
early_stopping = EarlyStopping(monitor='loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')
callbacks_list.append( early_stopping )

### SAVE when it stop to learn
save_checkpoint = ModelCheckpoint( CNN_folder + model_file + 'weight_checkpoint.h5', monitor='loss', save_weights_only=True )
callbacks_list.append( save_checkpoint )

### model compilations
model.compile( loss=loss,
               optimizer=optimizer,
               metrics=[coeff_determination] )

### model compilations
#multi_model.compile( loss=loss,
#               optimizer=optimizer,
#               metrics=[coeff_determination] )

########################
### SAVING THE MODEL ###
########################
def save_model( model, fileName ):
    """
    save a model
    """
    ### save the model
    model_json = model.to_json(  )
    with open( fileName+'.json', 'w' ) as json_file:
        json_file.write( model_json )
    ### save the weights
    model.save_weights( fileName+'_weights.h5' )
### ONE SAVED BEFORE THE TRAINING    
save_model( model, CNN_folder + model_file )

###########################################
### THE LEARNING FUNCTION: THE TRAINING ###
###########################################
history = model.fit_generator(generator=training_generator,
                              validation_data=validation_generator,
                              epochs=epochs,
                              callbacks=callbacks_list,
                              verbose=True,
                              use_multiprocessing=False,
                              #workers=12,
                              max_queue_size=1, 
                              shuffle=False,
                             )

#history = multi_model.fit_generator(
#                                    #generator=[training_generator,training_generator],
#                                    #validation_data=[validation_generator,validation_generator],
#                                    generator=training_generator,
#                                    validation_data=validation_generator,
#                                    epochs=epochs,
#                                    callbacks=callbacks_list,
#                                    verbose=True,
#                                    use_multiprocessing=False,
#                                    #workers=12,
#                                    max_queue_size=1, 
#                                    shuffle=False,
#                                    steps_per_epoch=400,
#                                   )

### ONE SAVE THE RESULT OF THE TRAINING
np.save( CNN_folder + history_file, history.history )

### ONE SAVE AFTER THE TRAINING, ERASING THE ONE BEFORE
save_model( model, CNN_folder + model_file )

##################
### PREDICTION ###
##################

#predictions = model.predict( LC_test, verbose=True )
test_generator       = DataGenerator( **params_test)
predictions = model.predict_generator( test_generator, verbose=True )
np.save( CNN_folder + prediction_file, predictions )

### Predict the validation, to be use only at the end end end ....
predictions_val = model.predict_generator( validation_generator, verbose=True )
np.save( CNN_folder + prediction_file_val, predictions_val )



