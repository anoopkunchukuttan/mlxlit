# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
import sys

from indicnlp import loader

from lm_eval import reader


flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", None, "data_path")
flags.DEFINE_string("lang", None, "Language")
flags.DEFINE_string("representation", "phonetic", "Representation of characters. Can take one of two values: phonetic or onehot")
flags.DEFINE_integer(
    "train_size", -1,
    "Size of the training data (number of sentences). Take the 'train_size' from the training file for training. If not specified, the entire training file is taken.")
flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")

FLAGS = flags.FLAGS


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size
    lang=config.lang

    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
    if is_training and config.keep_prob < 1:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

    self._initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.device("/cpu:0"):
      #embedding = tf.get_variable("embedding", [vocab_size, size])
      #inputs = tf.nn.embedding_lookup(embedding, self._input_data)

      bit_vector_embeddings=reader.get_bitvector_embeddings(lang,vocab_size,config.representation)
      bv_embedding_size=int(bit_vector_embeddings.get_shape()[1])
      embedding = tf.get_variable("embedding", [bv_embedding_size,size])
      embedding_wrapper  = tf.matmul(bit_vector_embeddings , embedding)
      inputs = tf.nn.embedding_lookup(embedding_wrapper, self._input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    ##The alternative version of the code below is:
    
    inputs = [tf.squeeze(input_, [1])
              for input_ in tf.split(axis=1, num_or_size_splits=num_steps, value=inputs)]
    outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)

    #### Tutorial version 

    #outputs = []
    #state = self._initial_state
    #with tf.variable_scope("RNN"):
    #  for time_step in range(num_steps):
    #    if time_step > 0: tf.get_variable_scope().reuse_variables()
    #    (cell_output, state) = cell(inputs[:, time_step, :], state)
    #    outputs.append(cell_output)

    output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, size])
    softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(self._targets, [-1])],
        [tf.ones([batch_size * num_steps])])
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20

  ### not hyper-parameters
  vocab_size = 125
  lang = 'hi'
  representation= 'phonetic'

class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20

  ### not hyper-parameters
  vocab_size = 10000
  lang = 'hi'
  representation= 'phonetic'

def run_epoch(session, m, data, eval_op, verbose=False):
  """Runs the model on the given data."""
  epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = m.initial_state.eval()
  for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size,
                                                    m.num_steps)):
    cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                 {m.input_data: x,
                                  m.targets: y,
                                  m.initial_state: state})
    costs += cost
    iters += m.num_steps

    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * m.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
  #### Load Indic NLP Library ###
  ## Note: Environment variable: INDIC_RESOURCES_PATH must be set
  loader.load()

  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  print('===========  PARAMETERS  ==============')
  print('Data Path: ' + FLAGS.data_path)
  print('Representation: ' + FLAGS.representation)
  print('Language: ' + str(FLAGS.lang))
  print('Corpus Size: ' + str(FLAGS.train_size))
  print('Config: ' + FLAGS.model)
  print('===========  PARAMETERS  ==============')

  raw_data = reader.ptb_raw_data(FLAGS.data_path,FLAGS.lang,FLAGS.train_size)
  train_data, valid_data, test_data, actual_vocab_size = raw_data
  print('Actual Vocab Size: ' + str(actual_vocab_size))

  ### set parameters   
  config = get_config()
  config.vocab_size=actual_vocab_size
  config.lang=FLAGS.lang
  config.representation=FLAGS.representation 

  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1
  eval_config.vocab_size=actual_vocab_size
  eval_config.lang=FLAGS.lang
  eval_config.representation=FLAGS.representation 

  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      m = PTBModel(is_training=True, config=config)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
      mvalid = PTBModel(is_training=False, config=config)
      mtest = PTBModel(is_training=False, config=eval_config)

    tf.global_variables_initializer().run()

    for i in range(config.max_max_epoch):
      lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
      m.assign_lr(session, config.learning_rate * lr_decay)

      print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
      train_perplexity = run_epoch(session, m, train_data, m.train_op,
                                   verbose=True)
      print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
      valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
      print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

    test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
    print("Test Perplexity: %.3f" % test_perplexity)

if __name__ == "__main__":
  tf.app.run()
