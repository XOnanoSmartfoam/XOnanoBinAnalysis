# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 12:40:52 2016

@author: Jake

This script is used to loop through a folder of bin files created with the XOnano board.
The files are read, filtered, plotted and analyzed. I have attempted to keep a well commented
script. Any questions email me @ jake.merrell@gmail.com or text/call at 801-369-2026.

About my system
Python Version: 3.5
IDE: Spyder 3.0.2
OS: Windows 10
"""

# Initialization
import struct
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.fftpack import fft
import pandas as pd
import peakutils as pks
from peakutils.plot import plot as pplot

##########  Functions  ##########

# Create bandpass filter function
def butter_bandpass(lowcut, highcut, fs, order=5):
  nyq = 0.5 * fs
  low = lowcut / nyq
  high = highcut / nyq
  b, a = butter(order, [low, high], btype='band')
  return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
  b, a = butter_bandpass(lowcut, highcut, fs, order=order)
  y = lfilter(b, a, data)
  return y
  
# Function to create variables dependent upon Firmware version
def firmware_version(version, numChannels):
  numChan = numChannels
  if (version >= 21):
      byteLine = 2 + accelerometerBytes + numChan*2
      data_fmt = '<h'+str(accelerometerBytes)+'b'+str(numChan)+'h'
      columns = 1 + accelerometerBytes + numChan
  else:
      byteLine = 2 + numChan*2
      data_fmt = '<h'+str(numChan)+'h'
      columns = 1 + numChan
  return byteLine, data_fmt, columns
  
# Create structures to pull data from bin file
def get_structure(dataformat):
  data_fmt = dataformat
  struct_len = struct.calcsize(data_fmt)
  struct_unpack = struct.Struct(data_fmt).unpack_from
  return struct_len,struct_unpack
  

#%% Build structure to read the header of the files
version_fmt = '<3b' # int[5], float, byte[255]
struct_len = struct.calcsize(version_fmt)
struct_unpack = struct.Struct(version_fmt).unpack_from

# Create global Variables
plotOn = 1  # Change to something other than 1 if you don't want to plot
cwd=os.getcwd()
headerSize = 3
accelerometerBytes = 3
DCoffset = .8266

# Create list of files in the directory that will be evaluated
binlist = [f for f in os.listdir(cwd) if f.endswith(".bin")]

# Create more length dependent variables
columns = np.zeros((len(binlist)))
samples = np.zeros((len(binlist)))
data_fmt = [None] * len(binlist)
version = np.zeros((len(binlist)))

# Determine the size of each file in the loop           
for hh in range(len(binlist)):
   
    with open(binlist[hh], "rb") as f:
        statinfo = os.stat(binlist[hh])
        numBytes = statinfo[6]
        
        # Find information about datafile
        ctdata = f.read(struct_len)
        header = struct_unpack(ctdata)
        version[hh] = header[0]
        numChan = header[1]
        byteChan = header[2]
        
        # Determine length of each line and create structure
        byteLine, data_fmt[hh], columns[hh] = firmware_version(version[hh], numChan)            
        samples[hh] = (numBytes - headerSize)/byteLine
            
# Create the 3D array that holds all of the data and populate with the time vector
rData = np.zeros((len(binlist),int(max(samples)),int(max(columns))))
fData = np.zeros((len(binlist),int(max(samples)),int(max(columns))))
time = np.arange(0, max(samples)/1000, 1/1000)
rData[:,:,0] = time
fData[:,:,0] = time

#%% Loop through the bin files to pull out the data and convert to voltage           
for ii in range(len(binlist)):
    print(binlist[ii])
    with open(binlist[ii], "rb") as f:
                  
      # Find information about datafile
      ctdata = f.read(struct_len)
      VersionChannel = struct_unpack(ctdata)
      
      # Create structures to unpack the bin files
      struct_len2, struct_unpack2 = get_structure(data_fmt[ii])
      
      jj = 0
      while True:
          ctdata = f.read(struct_len2)
          if not ctdata: 
              break
          raw = np.array(struct_unpack2(ctdata))
          jj += 1
          
          rData[ii,jj-1,1:int(columns[ii])] = raw[1:int(columns[ii])]
          
      # Convert bit data to Voltage on all ADC Channels
      if (version[ii] >= 21):
        for kk in range(0,numChan):
          rData[ii,0:int(samples[ii]),kk+4] = ((rData[ii,0:int(samples[ii]),kk+4])*3.3/np.power(2,12))-DCoffset
      else:
        for kk in range(0,numChan):
          rData[ii,0:int(samples[ii]),kk+1] = ((rData[ii,0:int(samples[ii]),kk+1])*3.3/np.power(2,12))-DCoffset

#%% Create filter and filter data
  
# Loop through and filter all data into fData array
for ll in range(len(fData)):
  
  # Filter parameters
  fs = 1000     # Hz
  lowcut = 10   # Hz
  highcut = 100 # Hz
  order = 5
  
  for mm in range(int(numChan)):
    # Filter each channel's signal
    if (version[mm] >= 21):
      fData[ll,:,mm+4] = butter_bandpass_filter(rData[ll,:,mm+4],lowcut,highcut,fs,order)
    else:
      fData[ll,:,mm+1] = butter_bandpass_filter(rData[ll,:,mm+1],lowcut,highcut,fs,order)
   
  
#%% Plot data
# Plot the data if plotOn == 1
for nn in range(len(fData)):
  if plotOn == 1:
    fileName = binlist[nn]
    fileName = fileName[:-4]
    plt.figure(fileName)
    
    if (version[nn] >= 21):
      firstChanIndex = 4
    else:
      firstChanIndex = 1
    
    # Plot the raw signal  
    plt.plot(rData[nn,:,0],rData[nn,:,firstChanIndex:firstChanIndex+numChan], label='Raw Signal')
    plt.xlabel('Time (sec)')
    plt.ylabel('Response (V)')
    
    # Plot the filtered voltage signal
    plt.plot(fData[nn,:,0],fData[nn,:,firstChanIndex:firstChanIndex+numChan], label='Filtered Signal')
    plt.legend()
    plt.show()
    
#%% Find Peaks and Integral

# Find peaks
for ii in range(len(fData)):
    if (version[ii] >= 21):
        for kk in range(0,numChan):
            maxV = max(fData[ii,:,kk+4])
            indexes = pks.indexes(fData[ii,:,kk+4], thres=maxV*.2, min_dist=1500)
            New_indexes = indexes[fData[ii,indexes,kk+4]>.04]
            peaks = fData[ii,New_indexes,kk+4]
            Stats = pd.DataFrame()
#            pData[ii,(kk-1)*3:]
    else:
        for kk in range(0,numChan):
            maxV = max(fData[ii,0:int(samples[ii]),kk+1])
            indexes[kk] = pks.indexes(fData[ii,0:int(samples[ii]),kk+1], thres=maxV*.15, min_dist=2000)
            
            
#    maxV = max(V_data_filt)
##        print(maxV)
#    indexes = pks.indexes(V_data_filt, thres=maxV*.15, min_dist=2000)
#    #  create the x values that correspond to voltage peaks greater than a certain threshold voltage  
#    New_indexes= indexes[V_data_filt[indexes]>.04]
##    print(indexes)
##    print(time[indexes], V_data_filt[indexes])
#    peaks = V_data_filt[New_indexes]        
#    Peak_Data.append(peaks)
#    plt.figure('Peak Voltages')
#    pplot(testdata[:,0], V_data_filt, New_indexes)
#        

    
    
    
#%% FFT (Sam Wilding)
for nn in range(len(fData)):
    fileName = binlist[nn]
    fileName = fileName[:-4]
    plt.figure(fileName + "FFT")  
    
    if (version[nn] >= 21):
      firstChanIndex = 4
    else:
      firstChanIndex = 1
    
    #Take the fft of a single channel  
    fft_fData=fft(fData[nn,:,firstChanIndex])
    N=len(fft_fData)
    
    #Sample spacing
    T=1.0/fs
    
    #Set the length of the X vector to half of the original time vector
#    fft_x = fData[nn,:,0][0:int(N/2)]
    xf = np.linspace(0.0, 500, N/2)
#    plt.plot(xf, 2.0/N * np.abs(fft_fData[0:int(N/2)]))
    plt.plot(xf, np.abs(fft_fData[0:int(N/2)]))
    plt.grid()
    plt.show()
    
