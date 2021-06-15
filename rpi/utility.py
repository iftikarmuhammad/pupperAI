# Copyright 2017 The TensorFlow Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for using reinforcement learning algorithms."""
import os
import re

import ruamel.yaml as yaml
import tensorflow as tf


def define_saver(exclude=None):
    """Create a saver for the variables we want to checkpoint.

  Args:
    exclude: List of regexes to match variable names to exclude.

  Returns:
    Saver object.
  """
    variables = []
    exclude = exclude or []
    exclude = [re.compile(regex) for regex in exclude]
    for variable in tf.global_variables():
        if any(regex.match(variable.name) for regex in exclude):
            continue
        variables.append(variable)
    saver = tf.train.Saver(variables, keep_checkpoint_every_n_hours=5)
    return saver


def load_config(logdir):
    """Load a configuration from the log directory.

  Args:
    logdir: The logging directory containing the configuration file.

  Raises:
    IOError: The logging directory does not contain a configuration file.

  Returns:
    Configuration object.
  """
    config_path = logdir and os.path.join(logdir, 'config.yaml')
    if not config_path or not tf.gfile.Exists(config_path):
        message = ('Cannot resume an existing run since the logging directory does not '
                   'contain a configuration file.')
        raise IOError(message)
    with tf.gfile.FastGFile(config_path, 'r') as file_:
        config = yaml.load(file_)
    message = 'Resume run and write summaries and checkpoints to {}.'
    tf.logging.info(message.format(config.logdir))
    return config
