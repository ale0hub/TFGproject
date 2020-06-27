# -*- coding: utf-8 -*-
"""
Created on Mon May  4 21:20:09 2020

@author: okeyr
"""

import matplotlib.pyplot as plt
from math import pi,floor
import numpy as np
import wave
import sys
from scipy.signal import hilbert,chirp,butter,filtfilt,find_peaks,find_peaks_cwt
from midiutil import MIDIFile


spf = wave.open("output.wav", "r")
signal = spf.readframes(-1)
AAsignal = np.fromstring(signal, "Int16")

fs = 44100
AAabssignal = np.abs(AAsignal)

threshold=2000
AAdenoisedsignal = [0 if sample < threshold else sample for sample in AAabssignal]

cutoff = 5
b, a = butter(2, 2*pi*cutoff/fs)
AAenvelope = filtfilt(b, a, AAdenoisedsignal)

thrs = 100
peaks, _ = find_peaks(AAenvelope, distance=150)
realPeaks = [peak for peak in peaks if AAenvelope[peak] > thrs]

assuranceThrs = 100
realLowestValues = list()
for peak in realPeaks:
    assuranceCount = 0
    lowestValue = peak
    previousValue = peak - 1
    while assuranceCount < assuranceThrs:
        if AAenvelope[lowestValue] > AAenvelope[previousValue]:
            lowestValue = previousValue
            assuranceCount = 0
        else:
            assuranceCount = assuranceCount + 1
        previousValue = previousValue - 1
    realLowestValues.append(lowestValue)

bpm = 100
delayPython = 0.33
rhythmNumBeats = 8
startSample = int(round(delayPython * fs))
noteNumSamples = int(round(fs * 60 / (bpm*4)))
rhythmNumSamples = noteNumSamples * rhythmNumBeats * 4
endSample = rhythmNumSamples + startSample
offsetMaxSamples = int(floor(noteNumSamples/2))

rhythm = list()
offsetsMap = {}
for i in range(rhythmNumBeats*4):
    currentGridSample = startSample + i*noteNumSamples
    stepValuesArray = np.zeros(27)
    hitsInStep = list()
    peaksInStep = list()
    for low,peak in zip(realLowestValues,realPeaks):
        if currentGridSample-offsetMaxSamples < low < currentGridSample+offsetMaxSamples:
            hitsInStep.append(low)
            peaksInStep.append(peak)
    if(len(hitsInStep)==1):
        stepValuesArray[3]=1
        stepValuesArray[3+9]=1
        stepValuesArray[3+18]=(hitsInStep[0]-currentGridSample)/(2*offsetMaxSamples)
        offsetsMap[i] = stepValuesArray[3+18]
    elif(len(hitsInStep)>1):
        index=np.argmax(AAenvelope[peaksInStep])
        stepValuesArray[3]=1
        stepValuesArray[3+9]=1
        stepValuesArray[3+18]=(hitsInStep[index]-currentGridSample)/(2*offsetMaxSamples)
        offsetsMap[i] = stepValuesArray[3+18]
    rhythm.append(stepValuesArray)

if rhythm[0][21] < 0:
    rhythm[0][21]=0
    offsetsMap[0]=0
            
track    = 0
channel  = 0
duration = 1/4    # In beats
volume   = 127  # 0-127, as per the MIDI standard
pitch = 46

MyMIDI = MIDIFile(1)  # One track, defaults to format 1 (tempo track is created
                      # automatically)
MyMIDI.addTempo(track, 0, bpm)

for i, offset in offsetsMap.items():
    MyMIDI.addNote(track, channel, pitch, (i + offset)/4, duration, volume)

with open("output.mid", "wb") as output_file:
    MyMIDI.writeFile(output_file)        

fig = plt.figure()
ax0 = fig.add_subplot(211)
ax0.plot(AAsignal, label='signal')
ax0.set_xlabel("Original signal")

fig = plt.figure()
ax0 = fig.add_subplot(211)
ax0.plot(AAabssignal, label='envelope')
ax0.set_xlabel("Absolut signal")

fig = plt.figure()
ax0 = fig.add_subplot(211)
ax0.plot(AAdenoisedsignal, label='envelope')
ax0.set_xlabel("Denoised signal")

fig = plt.figure()
ax0 = fig.add_subplot(211)
ax0.plot(AAenvelope, label='envelope')
ax0.set_xlabel("Envelope")

plt.plot(AAenvelope)
plt.plot(realPeaks, AAenvelope[realPeaks], "x")
plt.plot(realLowestValues, AAenvelope[realLowestValues], "o")
plt.show()
