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
# from __future__ import division, print_function
import struct
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
# from pypeaks import Data, Intervals
from scipy.signal import butter, lfilter, filtfilt, welch
from scipy.integrate import simps
from DetectPeaks import detect_peaks
import os
import pandas as pd
import sys
from scipy.fftpack import fft

plt.close("all")
##########  Functions  ##########

# Create bandpass filter function
def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    #  y = lfilter(b, a, data)
    y = filtfilt(b, a, data)
    return y


# Function to create variables dependent upon Firmware version
def firmware_version(version, numChannels):
    numChan = numChannels
    if (version >= 21):
        byteLine = 2 + accelerometerBytes + numChan * 2
        data_fmt = '<h' + str(accelerometerBytes) + 'b' + str(numChan) + 'h'
        columns = 1 + accelerometerBytes + numChan
    else:
        byteLine = 2 + numChan * 2
        data_fmt = '<h' + str(numChan) + 'h'
        columns = 1 + numChan
    return byteLine, data_fmt, columns


# Create structures to pull data from bin file
def get_structure(dataformat):
    data_fmt = dataformat
    struct_len = struct.calcsize(data_fmt)
    struct_unpack = struct.Struct(data_fmt).unpack_from
    return struct_len, struct_unpack


# Find the integral of a peak given the peak data and a peak index
def peak_integral(peakdata, peakindexes):
    abspeak_ints = []
    peak_ints = []
    to_peak_ints = []
    for ii in range(len(peakindexes)):
        intpeak = peakdata[peakindexes[ii] - 150: peakindexes[ii] + 200]
        inttopeak = peakdata[peakindexes[ii]-150]
        absintpeak = abs(intpeak)
        absVintegral = np.trapz(absintpeak, x=None, dx=1 / 1000)
        ToPeakInt = simps(intpeak, dx=1/1000)
        Vintegral = simps(intpeak, dx=1 / 1000)
        abspeak_ints.insert(ii, absVintegral)
        peak_ints.insert(ii, Vintegral)
        to_peak_ints.insert(ii,ToPeakInt)
    return abspeak_ints, peak_ints, to_peak_ints


# Determine the FFT of each peak on each channel
def find_fft(rawdata, peakindexes):
    global FFT, FFT_PkI, FFT_Pfreq, xf, N, f1, Pwelch
    FFT = []
    Pwelch = []
    FFT_PkI = []
    FFT_Pfreq = []
    preImpact = 150
    postImpact = 200
    sampfeq = 500
    # Take the fft of each impact on a single channel
    for ff in range(len(peakindexes)):
        fftdata = rawdata[peakindexes[ff] - preImpact: peakindexes[ff] + postImpact]
        fft_rData = fft(fftdata)
        N = len(fft_rData)
        #Power Spectral Analysis using Welch method
        f1, Pwelch_spec = welch(fftdata, sampfeq, nperseg=200, scaling='spectrum')

        # Set the length of the X vector to half of the original time vector
        xf = np.linspace(0.0, sampfeq / 2, N / 2)

        # Find peaks in FFT data
        fftpkI = detect_peaks(abs(fft_rData[0:int(N / 2)]), mpd=3, edge='rising')
        fftpkI = list(fftpkI)
        Pwelch.insert(ii,Pwelch_spec)
        FFT.insert(ii, fft_rData)
        FFT_PkI.append(fftpkI[0:numFreq])
        FFT_Pfreq.extend(xf[fftpkI[0:1]])


# return FFT, FFT_PkI, primary_freq, xf, N

# If two peaks are close together (impact peak and pick up peak),
# removes the second peak from the array
def remove_too_close(peakindexes):
    temp_indexes = peakindexes.tolist()
    too_close_samples = 1500
    previous_index = -1 * too_close_samples - 1
    for x in peakindexes:
        if ((x - previous_index) <= too_close_samples):
            temp_indexes.remove(x)
        else:
            previous_index = x
    return temp_indexes


# %% Build structure to read the header of the files
version_fmt = '<3b'  # int[5], float, byte[255]
struct_len = struct.calcsize(version_fmt)
struct_unpack = struct.Struct(version_fmt).unpack_from

# Create global Variables
cwd = os.getcwd()
headerSize = 3
accelerometerBytes = 3
DCoffset = .8266
peakData = []
numPeaks = 3

# Create list of files in the directory that will be evaluated
binlist = [f for f in os.listdir(cwd) if f.endswith(".bin")]

# Look for a pickle of the imported data
picklelist = [f for f in os.listdir(cwd) if f.endswith(".p")]

# Only start this portion if the pickle wasn't created
if len(picklelist) == 0:
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
            samples[hh] = (numBytes - headerSize) / byteLine

    # Create the 3D array that holds all of the data and populate with the time vector
    rData = np.zeros((len(binlist), int(max(samples)), int(max(columns))))
    fData = np.zeros((len(binlist), int(max(samples)), int(max(columns))))
    time = np.arange(0, max(samples) / 1000, 1 / 1000)
    pData = np.zeros((len(binlist), numChan * numPeaks))
    rData[:, :, 0] = time
    fData[:, :, 0] = time

    # %% Loop through the bin files to pull out the data and convert to voltage
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

                rData[ii, jj - 1, 1:int(columns[ii])] = raw[1:int(columns[ii])]

            # Convert bit data to Voltage on all ADC Channels
            if (version[ii] >= 21):
                for kk in range(0, numChan):
                    rData[ii, 0:int(samples[ii]), kk + 4] = ((rData[ii, 0:int(samples[ii]), kk + 4]) * 3.3 / np.power(2,
                                                                                                                      12)) - DCoffset
            else:
                for kk in range(0, numChan):
                    rData[ii, 0:int(samples[ii]), kk + 1] = ((rData[ii, 0:int(samples[ii]), kk + 1]) * 3.3 / np.power(2,
                                                                                                                      12)) - DCoffset
    ExcelFile = pd.ExcelFile('Input.xlsx')
    inputDF = ExcelFile.parse('Sheet1')
    inputDF.set_index('Bin File Names')
    # Save a pickle of all of the imported data
    with open("DataStash.p", "wb") as f:
        pickle.dump([rData, fData, numChan, version, samples, time], f)

else:
    # Load the pickled data
    ExcelFile = pd.ExcelFile('Input.xlsx')
    inputDF = ExcelFile.parse('Sheet1')
    inputDF = inputDF.set_index('Bin File Names')
    with open("DataStash.p", "rb") as f:
        rData, fData, numChan, version, samples, time = pickle.load(f)

# %% Filter data

# print('There are ', numChan, ' channels...')
# ExptdChan = input('Would you like to process all of them? (Y/N): ')

# if ExptdChan == "y" or ExptdChan == "Y":
processChan = 5
# else:
#  processChan = input('How many channels would you like to process?: ')


# Loop through and filter all data into fData array

# Filter parameters
fs = 1000  # Hz
lowcut = 10  # Hz
highcut = 100  # Hz
order = 5

for ll in range(len(fData)):


    for mm in range(int(processChan)):
        # Filter each channel's signal
        if (version[mm] >= 21):
            fData[ll, :, mm + 4] = butter_bandpass_filter(rData[ll, :, mm + 4], lowcut, highcut, fs, order)
        else:
            fData[ll, :, mm + 1] = butter_bandpass_filter(rData[ll, :, mm + 1], lowcut, highcut, fs, order)

# %% Plot data
# Determine if we will plot data and what to plot
plt.rcParams.update({'figure.max_open_warning': 0})
plotOn = input('Would you like to plot the data? (Y/N): ')
if plotOn == "Y" or plotOn == "y":
    plotFiltered = input('Options: Raw and Filetered signal (both), Filtered only (filtered), Raw only (raw):')

    for nn in range(len(fData)):
        fileName = binlist[nn]
        fileName = fileName[:-4]
        plt.figure(fileName)

        # Verify firmware version
        if (version[nn] >= 21):
            firstChanIndex = 4
        else:
            firstChanIndex = 1

            # Plot signal if decided above
        for aa in range(int(processChan)):
            if plotFiltered == "both" or plotFiltered == "raw":
                plt.plot(rData[nn, :, 0], rData[nn, :, firstChanIndex + aa], label=str(aa+1) + ' Raw Signal')

            # Plot the filtered voltage signal
            if plotFiltered == "both" or plotFiltered == "filtered":
                if plotFiltered == "filtered":
                    plt.plot(fData[nn, :, 0], fData[nn, :, firstChanIndex + aa], label=str(aa+1) + ' Filtered Signal')
                else:
                    plt.plot(fData[nn, :, 0], fData[nn, :, firstChanIndex + aa], 'r--',
                             label=str(aa+1) + ' Filtered Signal')

        plt.title('NCF Multiple Electrode Voltage Response')
        plt.xlabel('Time (sec)')
        plt.ylabel('Response (V)')
#        plt.legend()

        # This makes it plot full screen
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()

# %% Find Peaks and Integral

# Create DataFrame to hold all measures of voltage response (ie. Peak, Integral, Frequency)
expectedPeaks = 3
pltpks = input('Would you like to plot the peaks and fft? (Y/N): ')

numFreq = 5
impactData = []
testData = []
#allData = pd.DataFrame
FFT_Primary_frq = []
FFT_IArray = np.zeros((len(binlist), int(expectedPeaks), numFreq))
# Row_fft = np.zeros((len(binlist),(int(expectedPeaks*numFreq))))
for ii in range(len(fData)):
#    #New try
#    testDataPeak1 = [binlist[ii]]
#    testDataPeak1 = np.array(testDataPeak1)
#    testDataPeak2 = np.array(testDataPeak1)
#    testDataPeak3 = np.array(testDataPeak1)

    #end of new format try
    if pltpks == "Y" or pltpks == "y":
      plt.figure('Test ' + str(ii))
#for ii in range(12, 25):
    testPeaks = []
    if (version[ii] >= 21):
        for kk in range(0, int(processChan)):
            maxV = max(fData[ii, 0:int(samples[ii]), kk + 4])
            thresh = .075 * maxV

            # A bunch of if statements to remove unwanted peaks
            if ii == 3 and kk == 3:
              thresh = .1
            if ii == 0 or ii == 1 or ii == 7 or ii == 12 or ii == 13 or ii == 16:
              thresh = .35*maxV
              if ii == 16 and kk == 1:
                thresh = .002
            if ii == 20:
                Mthresh = .002
                if kk == 1:
                    Mthresh = .001
                    minpkdis = 100
            if ii == 5: Mthresh = .05
            if ii == 8 and kk == 2: Mthresh = .22
            if ii == 10 and kk == 1: Mthresh = .25
            if ii == 16 and kk == 2: Mthresh = .05
            if ii == 12 and kk == 4: Mthresh = .2
            if ii == 11 and kk == 2: Mthresh = .15


            # Find Peaks in signal
            pkI = detect_peaks(fData[ii, 0:int(samples[ii]), kk + 4], mpd=300, mph=thresh, edge='rising')
            ind = remove_too_close(pkI)

            # Once again used to remove unwanted peaks
            if len(ind) > int(expectedPeaks):
                ind = ind[0:int(expectedPeaks)]
            reading = list(fData[ii, ind, kk + 4])
            absInteg, Integ, ToPeakInt = peak_integral(fData[ii, 0:int(samples[ii]), kk + 4], ind)

            # Find the FFT and Power Spectral Analysis of the data
            find_fft(rData[ii, 0:int(samples[ii]), kk + 4], ind)

#            print('Test ' + str(ii) + ' Channel ' + str(kk))
            FFT = np.array(FFT)
            FFT_PkI = np.array(FFT_PkI)
            FFT_Primary_frq.append(FFT_Pfreq)

            chan = 'Chan ' + str(kk+1)
            pName = 'Peak ' + chan
            iName = 'Int ' + chan
            aiName = 'Abs Int ' + chan
            tpiName = 'To Peak Int ' + chan
            fftName = 'FFT  ' + chan

#            dataChan = pd.DataFrame({pName:reading,
#                                 aiName:absInteg,
#                                 iName:Integ,
#                                 tpiName:ToPeakInt,
#                                 fftName:FFT_Pfreq,
#                                 'Test Name':binlist[ii],
#                                 'Impact Location': inputDF['Hit Location'][binlist[ii]],
#                                 'Film 1 (mm^2)': inputDF['Film 1 (mm^2)'][binlist[ii]],
#                                 'Film 2 (mm^2)': inputDF['Film 2 (mm^2)'][binlist[ii]],
#                                 'Film 3 (mm^2)': inputDF['Film 3 (mm^2)'][binlist[ii]],
#                                 'Film 4 (mm^2)': inputDF['Film 4 (mm^2)'][binlist[ii]],
#                                 'Film 5 (mm^2)': inputDF['Film 5 (mm^2)'][binlist[ii]]})
            if kk == 0:
                datatest = pd.DataFrame({pName:reading,
                                 aiName:absInteg,
                                 iName:Integ,
                                 tpiName:ToPeakInt,
                                 fftName:FFT_Pfreq,
                                 'Test Name':binlist[ii],
                                 'Impact Location': inputDF['Hit Location'][binlist[ii]],
                                 'Film 1 (mm^2)': inputDF['Film 1 (mm^2)'][binlist[ii]],
                                 'Film 2 (mm^2)': inputDF['Film 2 (mm^2)'][binlist[ii]],
                                 'Film 3 (mm^2)': inputDF['Film 3 (mm^2)'][binlist[ii]],
                                 'Film 4 (mm^2)': inputDF['Film 4 (mm^2)'][binlist[ii]],
                                 'Film 5 (mm^2)': inputDF['Film 5 (mm^2)'][binlist[ii]],
                                 'Film Type': inputDF['Film Type'][binlist[ii]]})
            else:
                dataChan = pd.DataFrame({pName:reading,
                                 aiName:absInteg,
                                 iName:Integ,
                                 tpiName:ToPeakInt,
                                 fftName:FFT_Pfreq})
                datatest = pd.concat([datatest,dataChan],axis=1)

            # Plot the FFT of the raw signal
            if pltpks == "Y" or pltpks == "y":
#                plt.figure('Test ' + str(ii) + ' Channel ' + str(kk))
                for ff in range(len(FFT)):
                    plt.subplot(3, 1, 1)
                    plt.plot(xf, abs(FFT[ff, 0:int(N / 2)]))
                    plt.plot(xf[FFT_PkI[ff]], abs(FFT[ff, FFT_PkI[ff]]), 'k+')
                    plt.subplot(3, 1, 2)
                    plt.semilogy(f1, Pwelch[ff])
                    plt.title('Power Spectral Density')
                    plt.xlabel('Frequency (Hz)')

                #            plt.legend(['1st Impact','2nd Impact','3rd Impact','4th Impact','5th Impact'])
#                plt.xlabel('Frquency (Hz)')
#                plt.ylabel('FFT Response')

                # Plot the peaks
                plt.subplot(3, 1, 3)
                plt.plot(fData[ii, ind, 0], fData[ii, ind, kk + 4], 'r+', fData[ii, :, 0], fData[ii, :, kk + 4])
#                plt.xlabel('Time (sec)')
#                plt.ylabel('Foam Response (V)')
#                plt.suptitle('Signal Analysis - FFT and Peak Detection', size=16)
        if ii == 0:
            allData = datatest
        else:
            allData = allData.append(datatest, ignore_index=True)
allData = allData.set_index('Test Name')
allData = allData.reindex_axis(sorted(allData.columns), axis=1)
allData.to_excel('Final Output.xlsx')
