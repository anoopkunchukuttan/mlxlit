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


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import numpy as np
import tensorflow as tf

import codecs 
from indicnlp import langinfo as li
from indicnlp.script import indic_scripts as isc

def _read_words(filename,n_lines=-1):
  with codecs.open(filename, "r",  'utf-8') as f:
    lines = list(iter(f))
    lines = lines[:n_lines] if n_lines>0 and n_lines < len(lines) else lines
    lines = u''.join(lines)
    return lines.replace(u"\n", u" <eos> ").split()

def _build_vocab(filename,lang,n_lines=-1):
  """
  filename parameter as well n_lines parameter are not required
  """
  # add symbols inside co-ordinated range 
  words=[unichr(li.SCRIPT_RANGES[lang][0]+x) \
          for x in range(li.COORDINATED_RANGE_START_INCLUSIVE,li.COORDINATED_RANGE_END_INCLUSIVE+1) ]
  words.extend([u'<eos>',u'<unk>'])

  ## add symbols outside co-ordinated range 
  data = _read_words(filename,n_lines)
  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  all_words, _ = list(zip(*count_pairs))
  words.extend(set(all_words)-set(words))

  word_to_id = dict(zip(words, range(len(words))))
  return word_to_id

def _file_to_word_ids(filename, word_to_id, n_lines=-1):
  data = _read_words(filename, n_lines)
  ## unknown vocab iterm is assigned <unk>
  return [word_to_id.get(word,word_to_id[u'<unk>']) for word in data]


def ptb_raw_data(data_path=None,lang=None,train_size=-1):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id = _build_vocab(train_path,lang, train_size)
  train_data = _file_to_word_ids(train_path, word_to_id, train_size)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary


def ptb_iterator(raw_data, batch_size, num_steps):
  """Iterate on the raw PTB data.

  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.

  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.

  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  raw_data = np.array(raw_data, dtype=np.int32)

  data_len = len(raw_data)
  batch_len = data_len // batch_size
  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

  epoch_size = (batch_len - 1) // num_steps

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = data[:, i*num_steps:(i+1)*num_steps]
    y = data[:, i*num_steps+1:(i+1)*num_steps+1]
    yield (x, y)

def get_phonetic_bitvector_embeddings(lang,vocab_size):
    """
    Create bit-vector embeddings for vocabulary items. For phonetic chars,
    use phonetic embeddings, else use 1-hot embeddings
    """

    ##  phonetic embeddings for basic characters 
    pv=isc.get_phonetic_info(lang)[1]

    ## vocab statistics
    pv=np.copy(pv)
    org_shape=pv.shape
    additional_vocab=vocab_size-pv.shape[0]

    ##  new rows added 
    new_rows=np.zeros([additional_vocab,pv.shape[1]])
    pv=np.concatenate([pv,new_rows])

    ##  new columns added 
    new_cols=np.zeros([pv.shape[0],additional_vocab])
    pv=np.concatenate([pv,new_cols],axis=1)

    assert( (pv.shape[0]-org_shape[0]) == (pv.shape[1]-org_shape[1]) )

    ## 1-hot embeddings for new characters 
    for j,k in zip(range(org_shape[0],pv.shape[0]),range(org_shape[1],pv.shape[1])): 
        pv[j,k]=1

    return tf.constant(pv,dtype = tf.float32)

def get_onehot_bitvector_embeddings(lang,vocab_size): 
    return tf.constant(np.identity(vocab_size),dtype = tf.float32)

def get_bitvector_embeddings(lang,vocab_size,representation): 

    #bitvector_embeddings={
    #        'phonetic': get_phonetic_bitvector_embeddings,
    #        'onehot': get_onehot_bitvector_embeddings,
    #        }
    if representation=='phonetic':
        return get_phonetic_bitvector_embeddings(lang,vocab_size)
    elif representation=='onehot': 
        return get_onehot_bitvector_embeddings(lang,vocab_size)

