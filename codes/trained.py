# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 23:34:37 2020

@author: okeyr
"""
from magenta.music import midi_io as mmm

from magenta.models.music_vae import trained_model as tm

from magenta.models.music_vae import TrainedModel, configs

import numpy

import os

import tensorflow as tf
from six.moves import urllib

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

# Params: grooveConv (models.music_vae.data.GrooveConverter)
#         noteSeqs (list NoteSequence) 
# Return: data prepared to feed model (list models.music_vae.data.ConverterTensors)
def get_converter_tensors_from_note_sequences(grooveConv,noteSeqs):
    allConverterTensors = [];
    for seq in noteSeqs:
        allConverterTensors.append(grooveConv._to_tensors(seq))
    return allConverterTensors

# Params: index (int). what to return { 0:inputs | 1:outputs | 2: controls | 3:lengths } 
#         tensors (list models.music_vae.data.ConverterTensors) 
# Return: 3-D array ready to feed model [batch_size, seq_len, seq_depth] (numpy.array)
def get_numpy_array_from_converter_tensors(tensors,index=0):
    values = []
    for tsr,i in zip(tensors,range(len(tensors))):
        dictio =  {
            0: tsr.inputs, 
            1: tsr.outputs,
            2: tsr.controls,
            3: tsr.lengths
        }
        value = dictio.get(index)
        if (len(value)>0):
           for val in value:
               values.append(val)
    
    returnValues = numpy.zeros((len(values),len(values[0]),len(values[0][0])))          
    for value,i in zip(values,range(len(values))):
        returnValues[i] = value
    
    return returnValues

# Params: index (int). what to return { 0:inputs | 1:outputs | 2: controls | 3:lengths } 
#         tensors (list models.music_vae.data.ConverterTensors) 
# Return: 3-D array ready to feed model [batch_size, seq_len, seq_depth] (numpy.array)
def get_collection_from_converter_tensors(tensors,index=0):
    values = []
    for tsr,i in zip(tensors,range(len(tensors))):
        dictio =  {
            0: tsr.inputs, 
            1: tsr.outputs,
            2: tsr.controls,
            3: tsr.lengths
        }
        value = dictio.get(index)
        if (len(value)>0):
           for val in value:
               values.append(val)
    
    return values


def download_checkpoint(model_name: str,
                        checkpoint_name: str,
                        target_dir: str):
  """
  Downloads a Magenta checkpoint to target directory.

  Target directory target_dir will be created if it does not already exist.

      :param model_name: magenta model name to download
      :param checkpoint_name: magenta checkpoint name to download
      :param target_dir: local directory in which to write the checkpoint
  """
  tf.gfile.MakeDirs(target_dir)
  checkpoint_target = os.path.join(target_dir, checkpoint_name)
  if not os.path.exists(checkpoint_target):
    response = urllib.request.urlopen(
      f"https://storage.googleapis.com/magentadata/models/"
      f"{model_name}/checkpoints/{checkpoint_name}")
    data = response.read()
    local_file = open(checkpoint_target, 'wb')
    local_file.write(data)
    local_file.close()


def get_model(name: str):
  """
  Returns the model instance from its name.

      :param name: the model name
  """
  checkpoint = name + ".tar"
  download_checkpoint("music_vae", checkpoint, "checkpoints")
  return TrainedModel(
    # Removes the .lohl in some training checkpoint which shares the same config
    configs.CONFIG_MAP[name.split(".")[0] if "." in name else name],
    # The batch size changes the number of sequences to be processed together,
    # we'll be working with maximum 6 sequences (during groove)
    batch_size=1020,
    checkpoint_dir_or_path=os.path.join("checkpoints", checkpoint))



sequences = get_note_sequences_from_midi_files(get_midi_files_from_directory(os.path.abspath(os.getcwd())+'/groove/drummer1/session3'))

trainedModel = TrainedModel(configs.CONFIG_MAP['groovae_2bar_tap_fixed_velocity'],1,os.path.join("notebooks", "train", "taptest8"))

tensors = get_converter_tensors_from_note_sequences(trainedModel._config.data_converter,sequences)

inputs = get_collection_from_converter_tensors(tensors)

seq_len = [len(inputs[0])]*len(inputs)

latentSpace = trainedModel.encode_tensors(inputs,seq_len)

outputSequences = trainedModel.decode(latentSpace[0],32)

for seq,i in zip(outputSequences, range(len(outputSequences))):
    mmm.note_sequence_to_midi_file(sequence=seq, output_file=os.path.abspath(os.getcwd())+'/exportedMidi/example_'+str(i)+'.mid')

