# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 23:34:37 2020

@author: okeyr
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from magenta.music import midi_io as mmm

import trained_model, configs

import os

import tensorflow as tf

import matplotlib.pyplot as plt
from math import pi,floor
import numpy as np
import wave
from scipy.signal import butter,filtfilt,find_peaks
from midiutil import MIDIFile

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_dir', os.getcwd(),
    'The directory where wav files are located.')
flags.DEFINE_string(
    'input_file', 'output',
    'The name of the wav input file to convert')
flags.DEFINE_string(
    'output_dir', '',
    'The output directory where MIDI output files are being placed.')
flags.DEFINE_string(
    'checkpoints_dir', None,
    'The directory where checkpoints for the trained model are located')
flags.DEFINE_string(
    'config', 'groovae_2bar_tap_fixed_velocity',
    'The name of the config to use.')
flags.DEFINE_integer(
    'bpm', 100,
    'The tempo in BPM')
flags.DEFINE_string(
    'hparams', '',
    'A comma-separated list of `name=value` hyperparameter values to merge '
    'with those in the config.')
flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')

def write_midi_file(midiBytes,path=os.getcwd(),filename='output'):

    with open(os.path.join(path,filename+".mid"), "wb") as output_file:
        midiBytes.writeFile(output_file)

def get_midi_bytes(offsetsMap,bpm,track=0,channel=0,duration=1/4,volume=127,
                     pitch=46):

    MyMIDI = MIDIFile(1)  # One track, defaults to format 1 (tempo track is created
                      # automatically)
    MyMIDI.addTempo(track, 0, bpm)

    for i, offset in offsetsMap.items():
        MyMIDI.addNote(track, channel, pitch, (i + offset)/4, duration, volume)
    
    return MyMIDI

# Params: bpm (Int16)
#         delayPython (float16)
#         rhythmNumBeats (Int16)
#         fs (Int16)
# Return: constants
def get_sampling_constants(bpm,delayPython=0.33,rhythmNumBeats=8,fs=44100):

    startSample = int(round(delayPython * fs))
    noteNumSamples = int(round(fs * 60 / (bpm*4)))
    rhythmNumSamples = noteNumSamples * rhythmNumBeats * 4
    endSample = rhythmNumSamples + startSample
    offsetMaxSamples = int(floor(noteNumSamples/2))
    
    return startSample,endSample,noteNumSamples,rhythmNumSamples,offsetMaxSamples

# Params: signal (array Int16): original signal
#         peakValues (list Int16): peak positions on the original signal
#         lowValues (list Int16): peak start positions on the original signal
#         startSample (Int16): num of samples where to start the midi file
#         noteNumSamples (Int16): num of samples that last a note (1/4 beat)
#         offsetMaxSamples (Int16): max num of samples of the grid offset deviation
#         rhythmNumBeats (Int16): num of beats per rhythm
# Return: rhythm matrix (steps(32) x input depth (27))
def get_rhythm_from_peaks_and_lowest(signal,peakValues,lowValues,
                                     startSample,noteNumSamples,
                                     offsetMaxSamples,rhythmNumBeats=8):

    rhythm = list()
    for i in range(rhythmNumBeats*4):
        currentGridSample = startSample + i*noteNumSamples
        stepValuesArray = np.zeros(27)
        hitsInStep = list()
        peaksInStep = list()
        for low,peak in zip(peakValues,lowValues):
            if currentGridSample-offsetMaxSamples < low < currentGridSample+offsetMaxSamples:
                hitsInStep.append(low)
                peaksInStep.append(peak)
        if(len(hitsInStep)==1):
            stepValuesArray[3]=1
            stepValuesArray[3+9]=1
            stepValuesArray[3+18]=(hitsInStep[0]-currentGridSample)/(2*offsetMaxSamples)
        elif(len(hitsInStep)>1):
            index=np.argmax(signal[peaksInStep])
            stepValuesArray[3]=1
            stepValuesArray[3+9]=1
            stepValuesArray[3+18]=(hitsInStep[index]-currentGridSample)/(2*offsetMaxSamples)
        rhythm.append(stepValuesArray)

    if rhythm[0][21] < 0:
        rhythm[0][21]=0

    return rhythm

# Params: signal (array Int16): original signal
#         peakValues (list Int16): peak positions on the original signal
#         lowValues (list Int16): peak start positions on the original signal
#         startSample (Int16): num of samples where to start the midi file
#         noteNumSamples (Int16): num of samples that last a note (1/4 beat)
#         offsetMaxSamples (Int16): max num of samples of the grid offset deviation
#         rhythmNumBeats (Int16): num of beats per rhythm
# Return: offsetsMap (map(note position (Int16), offset value (float16))) to construct the midi file
def get_offsets_from_peaks_and_lowest(signal,peakValues,lowValues,
                                     startSample,noteNumSamples,
                                     offsetMaxSamples,rhythmNumBeats=8):

    offsetsMap = {}
    for i in range(rhythmNumBeats*4):
        currentGridSample = startSample + i*noteNumSamples
        hitsInStep = list()
        peaksInStep = list()
        for low,peak in zip(peakValues,lowValues):
            if currentGridSample-offsetMaxSamples < low < currentGridSample+offsetMaxSamples:
                hitsInStep.append(low)
                peaksInStep.append(peak)
        if(len(hitsInStep)==1):
            offsetsMap[i]=(hitsInStep[0]-currentGridSample)/(2*offsetMaxSamples)
            if i==0:
                offsetsMap[i] = np.clip(offsetsMap[i],0,0.5)
        elif(len(hitsInStep)>1):
            index=np.argmax(signal[peaksInStep])
            offsetsMap[i]=(hitsInStep[index]-currentGridSample)/(2*offsetMaxSamples)
            if i==0:
                offsetsMap[i] = np.clip(offsetsMap[i],0,0.5)
    
    return offsetsMap

# Params: signal (array Int16): original signal
#         thrs (int): threshold to discard too low peaks
#         peaksMinDistance (int): distance to discard peaks too close each other
#         assuranceThrs (int): assurance to locate correct lowest for every peak
# Return: lowest, peak positions (list Int16) 
def get_signal_peaks_position(signal,thrs=100,peaksMinDistance=150,
                              assuranceThrs=100):

    peaks, _ = find_peaks(signal, distance=peaksMinDistance)
    realPeaks = [peak for peak in peaks if signal[peak] > thrs]
    realLowestValues = list()
    for peak in realPeaks:
        assuranceCount = 0
        lowestValue = peak
        previousValue = peak - 1
        while assuranceCount < assuranceThrs:
            if signal[lowestValue] > signal[previousValue]:
                lowestValue = previousValue
                assuranceCount = 0
            else:
                assuranceCount = assuranceCount + 1
            previousValue = previousValue - 1
        realLowestValues.append(lowestValue)
    return realLowestValues,realPeaks
    
# Params: signal (array Int16): original signal
#         fs (int): sampling frequency
#         noiseThrs (float): noise threshold for denoising
#         cutoff (float): cutoff to calculate envelope
# Return: envelope of the original signal (array Float16)
def get_envelope_from_signal(signal,fs=44100,cutoff=5,noiseThrs=2000):
    absSignal = np.abs(signal)
    denoisedSignal = [0 if sample < noiseThrs else sample for sample in absSignal]
    b, a = butter(2, 2*pi*cutoff/fs)
    return filtfilt(b, a, denoisedSignal)
    
# Params: filename (string): wav file name without extension .wav
# Return: signal values on the file (array of Int16)
def get_signal_from_wav_file(filename,path):
    spf = wave.open(os.path.join(path,filename+".wav"), "r")
    signal = spf.readframes(-1)
    return np.fromstring(signal, "Int16")

# Params: directory (string): parent directory where midi files are located
# Return: midi files path (list string)
def get_midi_files_from_directory(directory):
    allMidiFiles = []
    for obj in os.listdir(directory):
        fullPath = os.path.join(directory,obj)
        if os.path.isdir(fullPath):
            allMidiFiles = allMidiFiles + get_midi_files_from_directory(fullPath)
        else:
            if obj.endswith(".mid"):
                allMidiFiles.append(fullPath)
    return allMidiFiles

# Params: midiFiles (list string): midi files path
# Return: converted midi files to note sequences (list NoteSequence)
def get_note_sequences_from_midi_files(midiFiles):
    allNoteSequences = []
    for file in midiFiles:
       noteSeq = mmm.midi_file_to_note_sequence(file)
       allNoteSequences.append(noteSeq)
    return allNoteSequences

def run(config_map):    
    if FLAGS.config not in config_map:
        raise ValueError('Invalid config: %s' % FLAGS.config)
    config = config_map[FLAGS.config]
    if not FLAGS.checkpoints_dir:
        raise ValueError('You need to specify a value to checkpoints_dir')
    checkpointsDir = FLAGS.checkpoints_dir
    if FLAGS.hparams:
        config.hparams.parse(FLAGS.hparams)   
    if FLAGS.output_dir:
        outputDir = FLAGS.output_dir
    else:
        outputDir = os.path.join(os.getcwd(),"outputs")
    
    wavSignal = get_signal_from_wav_file(FLAGS.input_file,FLAGS.input_dir)
    wavEnvelope = get_envelope_from_signal(wavSignal)
    peaks,lows = get_signal_peaks_position(wavEnvelope)
    start,end,noteNum,rhythmNum,offsetMax = get_sampling_constants(FLAGS.bpm)
    offsets = get_offsets_from_peaks_and_lowest(wavSignal,peaks,lows,start,
                                                noteNum,offsetMax)
    
    midiBytes = get_midi_bytes(offsets,FLAGS.bpm)
    write_midi_file(midiBytes,filename="tap_input")
    noteSequences = get_note_sequences_from_midi_files(
        get_midi_files_from_directory(os.getcwd()))
    
    #noteSequences = mmm.midi_to_note_sequence(midiBytes.bytes)
    
    trainedModel = trained_model.TrainedModel(config,
                                              config.hparams.batch_size,
                                              checkpointsDir)
    latentSpace = trainedModel.encode(noteSequences)
    outputSequences = trainedModel.decode(latentSpace[0],32)
    for seq,i in zip(outputSequences, range(len(outputSequences))):
        mmm.note_sequence_to_midi_file(sequence=seq, 
        output_file=outputDir+'\example_'+str(i)+'.mid')

def main(unused_argv):
    tf.logging.set_verbosity(FLAGS.log)
    run(configs.CONFIG_MAP)


def console_entry_point():
    tf.app.run(main)


if __name__ == '__main__':
    console_entry_point()
