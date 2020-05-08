# Copyright 2019 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configurations for MusicVAE models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from magenta.common import merge_hparams
from magenta.models.music_vae.myfiles import data
from magenta.models.music_vae.myfiles import data_hierarchical
from magenta.models.music_vae.myfiles import lstm_models
from magenta.models.music_vae.myfiles.base_model import MusicVAE
import magenta.music as mm
from tensorflow.contrib.training import HParams


class Config(collections.namedtuple(
    'Config',
    ['model', 'hparams', 'note_sequence_augmenter', 'data_converter',
     'train_examples_path', 'eval_examples_path', 'tfds_name'])):

  def values(self):
    return self._asdict()

Config.__new__.__defaults__ = (None,) * len(Config._fields)


def update_config(config, update_dict):
  config_dict = config.values()
  config_dict.update(update_dict)
  return Config(**config_dict)


CONFIG_MAP = {}

#
# Tensorboard on taptest11
CONFIG_MAP['groovae_2bar_tap_fixed_velocity'] = Config(
    model=MusicVAE(lstm_models.BidirectionalLstmEncoder(),
                   lstm_models.GrooveLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=16 * 2,  # 2 bars w/ 16 steps per bar
            z_size=256,
            enc_rnn_size=[512],
            dec_rnn_size=[256, 256],
            max_beta=0.2,
            free_bits=48,
            dropout_keep_prob=0.3,
        )),
    note_sequence_augmenter=None,
    data_converter=data.GrooveConverter(
        split_bars=2, steps_per_quarter=4, quarters_per_bar=4,
        max_tensors_per_notesequence=20, tapify=True, fixed_velocities=True,
        pitch_classes=data.ROLAND_DRUM_PITCH_CLASSES,
        inference_pitch_classes=data.REDUCED_DRUM_PITCH_CLASSES),
    tfds_name='groove/2bar-midionly'
)

CONFIG_MAP['groovae_2bar_tap_fixed_velocity_note_dropout'] = Config(
    model=MusicVAE(lstm_models.BidirectionalLstmEncoder(),
                   lstm_models.GrooveLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=16 * 2,  # 2 bars w/ 16 steps per bar
            z_size=256,
            enc_rnn_size=[512],
            dec_rnn_size=[256, 256],
            max_beta=0.2,
            free_bits=48,
            dropout_keep_prob=0.3,
        )),
    note_sequence_augmenter=None,
    data_converter=data.GrooveConverter(
        split_bars=2, steps_per_quarter=4, quarters_per_bar=4,
        max_tensors_per_notesequence=20, tapify=True, fixed_velocities=True,
        pitch_classes=data.ROLAND_DRUM_PITCH_CLASSES,
        inference_pitch_classes=data.REDUCED_DRUM_PITCH_CLASSES,
        max_note_dropout_probability=0.8),
    tfds_name='groove/2bar-midionly'
)

# groovae_2bar_tap_fixed_velocity with
# z_size=128 (half)
#
# Tensorboard on taptest12
CONFIG_MAP['groovae_2bar_tap_fixed_velocity_z_128'] = Config(
    model=MusicVAE(lstm_models.BidirectionalLstmEncoder(),
                   lstm_models.GrooveLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=16 * 2,  # 2 bars w/ 16 steps per bar
            z_size=128,
            enc_rnn_size=[512],
            dec_rnn_size=[256, 256],
            max_beta=0.2,
            free_bits=48,
            dropout_keep_prob=0.3,
        )),
    note_sequence_augmenter=None,
    data_converter=data.GrooveConverter(
        split_bars=2, steps_per_quarter=4, quarters_per_bar=4,
        max_tensors_per_notesequence=20, tapify=True, fixed_velocities=True,
        pitch_classes=data.ROLAND_DRUM_PITCH_CLASSES,
        inference_pitch_classes=data.REDUCED_DRUM_PITCH_CLASSES),
    tfds_name='groove/2bar-midionly'
)

# groovae_2bar_tap_fixed_velocity with
# z_size=512 (double)
#
# Tensorboard on taptest13
CONFIG_MAP['groovae_2bar_tap_fixed_velocity_z_512'] = Config(
    model=MusicVAE(lstm_models.BidirectionalLstmEncoder(),
                   lstm_models.GrooveLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=16 * 2,  # 2 bars w/ 16 steps per bar
            z_size=512,
            enc_rnn_size=[512],
            dec_rnn_size=[256, 256],
            max_beta=0.2,
            free_bits=48,
            dropout_keep_prob=0.3,
        )),
    note_sequence_augmenter=None,
    data_converter=data.GrooveConverter(
        split_bars=2, steps_per_quarter=4, quarters_per_bar=4,
        max_tensors_per_notesequence=20, tapify=True, fixed_velocities=True,
        pitch_classes=data.ROLAND_DRUM_PITCH_CLASSES,
        inference_pitch_classes=data.REDUCED_DRUM_PITCH_CLASSES),
    tfds_name='groove/2bar-midionly'
)

# groovae_2bar_tap_fixed_velocity with
# z_size=64 (quarter)
#
# Tensorboard on taptest14
CONFIG_MAP['groovae_2bar_tap_fixed_velocity_z_64'] = Config(
    model=MusicVAE(lstm_models.BidirectionalLstmEncoder(),
                   lstm_models.GrooveLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=16 * 2,  # 2 bars w/ 16 steps per bar
            z_size=64,
            enc_rnn_size=[512],
            dec_rnn_size=[256, 256],
            max_beta=0.2,
            free_bits=48,
            dropout_keep_prob=0.3,
        )),
    note_sequence_augmenter=None,
    data_converter=data.GrooveConverter(
        split_bars=2, steps_per_quarter=4, quarters_per_bar=4,
        max_tensors_per_notesequence=20, tapify=True, fixed_velocities=True,
        pitch_classes=data.ROLAND_DRUM_PITCH_CLASSES,
        inference_pitch_classes=data.REDUCED_DRUM_PITCH_CLASSES),
    tfds_name='groove/2bar-midionly'
)

# 
#
#
# Tensorboard on taptest11
CONFIG_MAP['groovae_2bar_tap_fixed_velocity_z_256_beta_0.4'] = Config(
    model=MusicVAE(lstm_models.BidirectionalLstmEncoder(),
                   lstm_models.GrooveLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=16 * 2,  # 2 bars w/ 16 steps per bar
            z_size=256,
            enc_rnn_size=[512],
            dec_rnn_size=[256, 256],
            max_beta=0.4,
            free_bits=48,
            dropout_keep_prob=0.3,
        )),
    note_sequence_augmenter=None,
    data_converter=data.GrooveConverter(
        split_bars=2, steps_per_quarter=4, quarters_per_bar=4,
        max_tensors_per_notesequence=20, tapify=True, fixed_velocities=True,
        pitch_classes=data.ROLAND_DRUM_PITCH_CLASSES,
        inference_pitch_classes=data.REDUCED_DRUM_PITCH_CLASSES),
    tfds_name='groove/2bar-midionly'
)