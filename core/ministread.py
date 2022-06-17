#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 18:53:53 2022
read volumes
@author: hao
"""
#%% print size
import pandas as pd
path = "./Documents/QLTest/QuakeMNIST/Catalogues/"
file = "Sn/GlobalQuakesSn.csv"
pf = pd.read_csv(path+file)
print(len(pf))

#%% plot

import random
def plotter(data, evi, arrival_sample, phase_type="P_Arrival"):


    """

    Generates plots.

    Parameters
    ----------
    dataset: obj
        data trace
    arrival_sample:  int
        arrival point    
    """
    sd = random.randint(30,50)
    spt = arrival_sample
    sst = 1894
    y2 = np.zeros((6000,1))
    if spt and (spt-20 >= 0) and (spt+20 < 6000):
        y2[ spt-20:spt+20, 0] = np.exp(-(np.arange(spt-20,spt+20)-spt)**2/(2*(10)**2))[:6000-(spt-20)]                
    elif spt and (spt-20 < 6000):
        y2[ 0:spt+20, 0] = np.exp(-(np.arange(0,spt+20)-spt)**2/(2*(10)**2))[:6000-(spt-20)]    
    
    y3 = np.zeros((6000,1))    
    if sst and (sst-20 >= 0) and (sst+20 < 6000):
        y3[ sst-20:sst+20, 0] = np.exp(-(np.arange(sst-20,sst+20)-sst)**2/(2*(10)**2))[:6000-(sst-20)]                
    elif spt and (spt-20 < 6000):
        y3[ 0:sst+20, 0] = np.exp(-(np.arange(0,sst+20)-sst)**2/(2*(10)**2))[:6000-(sst-20)] 
    yh3 = y3.T
    yh3= yh3.reshape(-1,1)
    yh3 = yh3 * random.uniform(0.35,0.75)    
    sphase_type = "S_Arrival"
    
    yh2 = y2.T
    yh2= yh2.reshape(-1,1)
    yh2 = yh2 * random.uniform(0.35,0.85)

    fig = plt.figure()
    ax = fig.add_subplot(411)
    plt.plot(data[:, 0], 'k')
    plt.rcParams["figure.figsize"] = (8,5)
    
    
    plt.title(str(evi))    
    plt.tight_layout()
    ymin, ymax = ax.get_ylim()
    pl = plt.vlines(int(spt), ymin, ymax, color='b', linewidth=2, label='Manual_'+phase_type)
    
    pls = plt.vlines(int(sst), ymin, ymax, color='r', linewidth=2, label='Manual_'+sphase_type)
    legend_properties = {'weight':'bold'}
    plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)
    
    pl = plt.vlines(int(spt), ymin, ymax, color='b', linewidth=2, label=phase_type)
    ax = fig.add_subplot(412)
    plt.plot(data[:, 1] , 'k')
    plt.tight_layout()
    pl = plt.vlines(int(spt), ymin, ymax, color='b', linewidth=2, label='Manual_'+phase_type)
    pls = plt.vlines(int(sst), ymin, ymax, color='r', linewidth=2, label='Manual_'+sphase_type)
    plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)

    ax = fig.add_subplot(413)
    plt.plot(data[:, 2], 'k')
    plt.tight_layout()
    
    ymin, ymax = ax.get_ylim()
    pl = plt.vlines(int(spt), ymin, ymax, color='b', linewidth=2, label='Manual_'+phase_type)  
    pls = plt.vlines(int(sst), ymin, ymax, color='r', linewidth=2, label='Manual_'+sphase_type)
    plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)


    ax = fig.add_subplot(414)
    x = np.linspace(0, data.shape[0], data.shape[0], endpoint=True)

    #plt.plot(x, yh1, 'g--', alpha = 0.5, linewidth=1.5, label='Detection')
    plt.plot(x, yh2, 'b--', alpha = 0.5, linewidth=1.5, label=phase_type.split('_')[0]+'_probability')
    plt.plot(x, yh3, 'r--', alpha = 0.5, linewidth=1.5, label=sphase_type.split('_')[0]+'_probability')
    #plt.plot(x, yh3, 'r--', alpha = 0.5, linewidth=1.5, label='S_probability')
    plt.tight_layout()
    plt.ylim((-0.1, 1.1))
    ymin, ymax = ax.get_ylim()
    pl = plt.vlines(int(spt), ymin, ymax, color='b', linestyles='dashed', label='Predicted_'+phase_type)
    pls = plt.vlines(int(sst), ymin, ymax, color='r', linestyles='dashed', label='Predicted_'+sphase_type)
    plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)
    

    fig.savefig( str(evi.split('/')[-1])+'.png')
#%% Multiple Plot
def multi_plotter(data, evi, arrival_sample, phase_type="S", allphase=['P']):


    """

    Generates plots.

    Parameters
    ----------
    dataset: obj
        data trace
    arrival_sample:  int
        arrival point    
    """
    sd = random.randint(30,50)
    spt = arrival_sample
    y2 = np.zeros((6000,1))
    if spt and (spt-20 >= 0) and (spt+20 < 6000):
        y2[ spt-20:spt+20, 0] = np.exp(-(np.arange(spt-20,spt+20)-spt)**2/(2*(10)**2))[:6000-(spt-20)]                
    elif spt and (spt-20 < 6000):
        y2[ 0:spt+20, 0] = np.exp(-(np.arange(0,spt+20)-spt)**2/(2*(10)**2))[:6000-(spt-20)]    
        
    yh2 = y2.T
    yh2= yh2.reshape(-1,1)
    yh2 = yh2 * random.uniform(0.35,0.85)

    fig = plt.figure()
    ax = fig.add_subplot(411)
    plt.plot(data[:, 0], 'k')
    plt.rcParams["figure.figsize"] = (8,5)
    
    plt.title(str(evi))    
    plt.tight_layout()
    ymin, ymax = ax.get_ylim()
    pl = plt.vlines(int(spt), ymin, ymax, color='b', linewidth=2, label='Manual_'+phase_type+"_Arrival")
    legend_properties = {'weight':'bold'}
    plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)
    
    pl = plt.vlines(int(spt), ymin, ymax, color='b', linewidth=2, label=phase_type)
    ax = fig.add_subplot(412)
    plt.plot(data[:, 1] , 'k')
    plt.tight_layout()
    pl = plt.vlines(int(spt), ymin, ymax, color='b', linewidth=2, label='Manual_'+phase_type+"_Arrival")
    plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)

    ax = fig.add_subplot(413)
    plt.plot(data[:, 2], 'k')
    plt.tight_layout()
    
    ymin, ymax = ax.get_ylim()
    pl = plt.vlines(int(spt), ymin, ymax, color='b', linewidth=2, label='Manual_'+phase_type+"_Arrival")    
    plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)


    ax = fig.add_subplot(414)
    x = np.linspace(0, data.shape[0], data.shape[0], endpoint=True)

    #plt.plot(x, yh1, 'g--', alpha = 0.5, linewidth=1.5, label='Detection')
    plt.plot(x, yh2, 'r--', alpha = 0.5, linewidth=1.5, label=phase_type +'_probability')
    for phasename in allphase:
        plt.plot(x, np.zeros(6000), 'b--', alpha = 0.5, linewidth=1.5, label=phasename +'_probability')
        
    #plt.plot(x, yh3, 'r--', alpha = 0.5, linewidth=1.5, label='S_probability')
    plt.tight_layout()
    plt.ylim((-0.1, 1.1))
    ymin, ymax = ax.get_ylim()
    pl = plt.vlines(int(spt), ymin, ymax, color='r', linestyles='dashed', label='Predicted_'+phase_type)
    plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)
    

    fig.savefig( str(evi.split('/')[-1])+'.png')
#%% read CSV document
from obspy import read,UTCDateTime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
csvpath = '/Users/hao/Documents/QLTest/MNIST/Catalogues/Sg.csv'
#steadfile = '/Users/hao/Downloads/STEAD/merge.csv'
mseedpath = '/Users/hao/Documents/QLTest/MNIST/Sg/'

#Read csv file
steadcsv = pd.read_csv(csvpath)



for index, row in steadcsv.iterrows():
    st = read(mseedpath+row['FILENAME']+"*.mseed")
    npz_data = np.zeros([6000,3])
    # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel
    for tr in st:
        if tr.stats.channel[2] == 'E':
            npz_data[:,0] = tr.data
        if tr.stats.channel[2] == 'N':
            npz_data[:,1] = tr.data  
        if tr.stats.channel[2] == 'Z':
            npz_data[:,2] = tr.data
    # calculate arrival sample point
    arrival_sample = int((UTCDateTime(row['ARRIVAL_DATE']+row['ARRIVAL_TIME']) - tr.stats.starttime)*tr.stats.sampling_rate)
    npz_data = np.float32(npz_data)
    start_time = st[0].stats.starttime
    #edit start_time_str
    start_time_str = str(start_time)   
    start_time_str = start_time_str.replace('T', '')                 
    start_time_str = start_time_str.replace('Z', '')  
    start_time_str = start_time_str.replace('-', '')                 
    start_time_str = start_time_str.replace(':', '')   
    start_time_str=start_time_str[:14]    
    tr_name = st[0].stats.station+'_'+st[0].stats.network+'_'+st[0].stats.channel[:2]+'_'+start_time_str+'_EV'
    input("Next Plot!")
    #break
#%% read mseed file to plot
plotter(npz_data, tr_name, arrival_sample,"Sg_Arrival")
#multi_plotter(npz_data, tr_name, arrival_sample,"Pg", ["P","Pb"])
#%% read hdf5
import h5py
csvpath = "/Users/hao/opt/anaconda3/envs/eqtdev/lib/python3.7/site-packages/EQTransformer/ModelsAndSampleData/100samples.csv"
steadcsv = pd.read_csv(csvpath)

fl = h5py.File("/Users/hao/opt/anaconda3/envs/eqtdev/lib/python3.7/site-packages/EQTransformer/ModelsAndSampleData/100samples.hdf5", 'r')
dset = fl['data']
list_IDs_temp = steadcsv.loc[:,'trace_name'].to_list()
# Generate data
for i, ID in enumerate(dset.keys()):
    additions = None
    dataset = dset[ID]
    if ID.split('_')[-1] == 'NO':
        data = np.array(dataset)
    else:
        # Hao revised
        data = np.array(dataset)                    
        spt = int(dataset.attrs['p_arrival_sample'])
        sst = dataset.attrs['s_arrival_sample']
    input("Next")
#%%
plotter(data, ID, spt,"P_Arrival")