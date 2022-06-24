#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 15:49:08 2022

@author: hao
"""
#%% Train Example

hdf5_file_path = './ModelsAndSampleData/100samples.hdf5'
csv_file_path = './ModelsAndSampleData/100samples.csv'
from EQTransformer.core.trainer import trainer
trainer(input_hdf5=hdf5_file_path,
        input_csv=csv_file_path,
        output_name='ps_samples_trainer',                
        cnn_blocks=2,
        lstm_blocks=1,
        padding='same',
        activation='relu',
        drop_rate=0.2,
        label_type='gaussian',
        add_event_r=0.6,
        add_gap_r=0.2,
        shift_event_r=0.9,
        add_noise_r=0.5, 
        mode='generator',
        train_valid_test_split=[0.80, 0.10, 0.10],
        loss_types=['binary_crossentropy', 'binary_crossentropy'], # if only 2 output
        batch_size=20,
        epochs=10, 
        patience=20,
        gpuid=None,
        gpu_limit=None,
        phase_types=['P','S'])
#%% Train to Test
from EQTransformer.core.tester import tester
tester(input_hdf5='./ModelsAndSampleData/100samples.hdf5',
       input_testset='./ps_samples_trainer_outputs/test.npy',
       input_model='./ps_samples_trainer_outputs/final_model.h5',
       output_name='tl_noaug_samples_tester',
       detection_threshold=0.20,                
       P_threshold=0.1,
       S_threshold=0.1, 
       number_of_plots=3,
       estimate_uncertainty=True, 
       number_of_sampling=2,
       input_dimention=(6000, 3),
       normalization_mode='std',
       mode='generator',
       loss_weights=[ 0.40, 0.55], #0.05,
       loss_types=['binary_crossentropy', 'binary_crossentropy'],# if only 2 output
       batch_size=50,
       gpuid=None,
       gpu_limit=None)
#%% test
from EQTransformer.core.tester import tester
tester(input_hdf5='./ModelsAndSampleData/100samples.hdf5',
       input_testset='./samples_trainer_outputs/test.npy',
       input_model='./samples_trainer_outputs/final_model.h5',
       output_name='samples100_tester',
       detection_threshold=0.20,                
       P_threshold=0.1,
       S_threshold=0.1, 
       number_of_plots=3,
       estimate_uncertainty=True, 
       number_of_sampling=2,
       input_dimention=(6000, 3),
       normalization_mode='std',
       mode='generator',
       batch_size=10,
       gpuid=None,
       gpu_limit=None)


#%% Detection
from EQTransformer.core.predictor import predictor
predictor(input_dir='./ModelsAndSampleData',   
         input_model='./samples_trainer_outputs/final_model.h5',
         output_dir='tl_detections_p',
         estimate_uncertainty=False, 
         output_probabilities=False,
         number_of_sampling=5,
         loss_weights=[0.02, 0.40, 0.58],          
         detection_threshold=0.3,                
         P_threshold=0.1,
         S_threshold=0.1, 
         number_of_plots=10,
         plot_mode='time',
         batch_size=500,
         number_of_cpus=4,
         keepPS=False,
         spLimit=60) 


