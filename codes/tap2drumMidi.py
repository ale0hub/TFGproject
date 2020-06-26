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

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_dir', None,
    'The directory where MIDI input files are located.')
flags.DEFINE_string(
    'output_dir', '',
    'The output directory where MIDI output files are being placed.')
flags.DEFINE_string(
    'checkpoints_dir', None,
    'The directory where checkpoints for the trained model are located')
flags.DEFINE_string(
    'config', 'groovae_2bar_tap_fixed_velocity',
    'The name of the config to use.')
flags.DEFINE_string(
    'hparams', '',
    'A comma-separated list of `name=value` hyperparameter values to merge '
    'with those in the config.')
flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')

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
    if not FLAGS.input_dir:
        raise ValueError('You need to pick a value to input_dir')
    noteSequences = get_note_sequences_from_midi_files(
        get_midi_files_from_directory(FLAGS.input_dir))
    
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
