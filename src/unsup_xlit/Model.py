import ParallelDataReader
import MonoDataReader
import Mapping

import numpy as np 
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

# Do something with output folder
class Model():
    def __init__(self,mapping,representation,embedding_size,max_sequence_length):
        self.embedding_size = embedding_size
        self.mapping = mapping
        rnn_size = embedding_size
        self.rnn_size = rnn_size
        self.max_sequence_length = max_sequence_length
        self.representation=representation

        self.lang_list = self.mapping.get_langs()
        self.vocab_size = self.mapping.get_vocab_size()

        # Finds bit-vector representation for each character of each language
        self.bitvector_embeddings={} 
        for lang in self.lang_list: 
            self.bitvector_embeddings[lang] = self.mapping.get_bitvector_embeddings(lang,self.representation)
        self.bitvector_embedding_size=self.mapping.get_bitvector_embedding_size(self.representation)    

        # Converting the character representation to embedding_size vector
        max_val = 0.1
        self.embed_W0 = tf.Variable(tf.random_uniform([self.bitvector_embedding_size,self.embedding_size], -1*max_val, max_val), name = 'embed_W0') 
        self.embed_W = dict()
        for lang in self.lang_list:
            self.embed_W[lang] = tf.matmul(self.bitvector_embeddings[lang], self.embed_W0)
        self.embed_b = tf.Variable(tf.constant(0., shape=[embedding_size]), name = 'embed_b')

        # Creating BasicLSTM Cells and initial inputs to decoders
        self.encoder_cell = rnn_cell.BasicLSTMCell(rnn_size)

        self.decoder_input = dict()
        for lang in self.lang_list:
            self.decoder_input[lang] = tf.Variable(tf.random_uniform([1, embedding_size], dtype = tf.float32))
        
        self.decoder_cell = dict()
        for lang in self.lang_list:
            self.decoder_cell[lang] = rnn_cell.BasicLSTMCell(rnn_size)

        # Output decoder to vocab_size vector
        self.out_W = dict()
        self.out_b = dict()
        for lang in self.lang_list:
            self.out_W[lang] = tf.Variable(tf.random_uniform([self.rnn_size,self.vocab_size], -1*max_val, max_val))
            self.out_b[lang] = tf.Variable(tf.constant(0., shape = [self.vocab_size]))

    # Given sequences of char_ids, compute hidden representation of each sequence
    def compute_hidden_representation(self, sequences, sequence_lengths, lang):
        x = tf.transpose(tf.add(tf.nn.embedding_lookup(self.embed_W[lang],sequences),self.embed_b),[1,0,2])
        x = tf.reshape(x,[-1,self.embedding_size])
        x = tf.split(0,self.max_sequence_length,x)
        _, states = rnn.rnn(self.encoder_cell, x, dtype = tf.float32, sequence_length = sequence_lengths)
        
        return states

    # Find cross entropy loss in predicting target_sequences from computed hidden representation (intial state)
    def seq_loss(self, target_sequence, target_masks, lang, initial_state):
        state = initial_state
        loss = 0.0
        cell = self.decoder_cell[lang]
        batch_size = tf.shape(target_sequence)[0]

        # One step generate one character for each sequence
        for i in range(self.max_sequence_length):
            # for first iteration, decoder_input embedding is used, otherwise, output from previous iteration is used
            # embedding lookup replace the character index with its embedding_size vector representation, which is given to the rnn_cell
            if(i==0):
                current_emb = tf.reshape(tf.tile(self.decoder_input[lang],tf.pack([batch_size,1])),[-1,self.embedding_size])    
            else:
                current_emb = tf.nn.embedding_lookup(self.embed_W[lang],target_sequence[:,i-1])+self.embed_b

            if i > 0 : tf.get_variable_scope().reuse_variables()

            # Run one step of the decoder cell. Updates 'state' and store output in 'output'
            output, state = cell(current_emb,state)

            # Generating one-hot labels for target_sequences
            labels = tf.expand_dims(target_sequence[:,i],1)
            indices = tf.expand_dims(tf.range(0,batch_size),1)
            concated = tf.concat(1,[indices,labels])
            onehot_labels = tf.sparse_to_dense(concated,tf.pack([batch_size,self.vocab_size]),1.0,0.0)

            # Find probabilities of character
            logit_words = tf.matmul(output,self.out_W[lang])+self.out_b[lang]

            # Predicted char_ids
            output_id = tf.argmax(logit_words,1)

            # Finding cross entropy
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_words, onehot_labels)
            # Takaing cross entropy for only non-padding characters
            cross_entropy = cross_entropy * target_masks[:,i]

            # Add cross entropy to the loss
            loss = loss + tf.reduce_sum(cross_entropy)

        loss = loss / tf.reduce_sum(target_masks[:,1:])

        return loss

    # Same as seq_loss, except that it takes source sequences and source lang, instead of computed hidden states
    # For each lang:
    #    sequences, sequence masks: tensors of shape: [batch_size, max_sequence lengths]
    #    sequence_lengths: tensor of shape: [batch_sizes]
    def seq_loss_2(self,lang1,sequences,sequence_masks,sequence_lengths,lang2,target_sequences,target_sequence_masks,target_sequence_lengths):
        hidden_representation = self.compute_hidden_representation(sequences,sequence_lengths,lang1)
        loss = self.seq_loss(target_sequences,target_sequence_masks,lang2,hidden_representation)
        return loss

    # Get a monolingual optimizer for 'lang' language
    # sequences, sequence masks: tensors of shape: [batch_size, max_sequence lengths]
    # sequence_lengths: tensor of shape: [batch_sizes]
    def get_mono_optimizer(self,learning_rate,lang,sequences,sequence_masks,sequence_lengths):
        hidden_representation = self.compute_hidden_representation(sequences,sequence_lengths,lang)
        loss = self.seq_loss(sequences,sequence_masks,lang,hidden_representation)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return optimizer

    # Get a monolingual optimizer for 'lang' language
    # sequences, sequence masks: tensors of shape: [batch_size, max_sequence lengths]
    # sequence_lengths: tensor of shape: [batch_sizes]
    def get_parallel_optimizer(self,learning_rate,lang1,sequences,sequence_masks,sequence_lengths,lang2,target_sequences,target_sequence_masks,target_sequence_lengths):
        hidden_representation = self.compute_hidden_representation(sequences,sequence_lengths,lang1)
        loss = self.seq_loss(target_sequences,target_sequence_masks,lang2,hidden_representation)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return optimizer

    # Optimizer to minimize mean squared differences of hidden representation of same word from different languages 
    # (mean taken over all batch_size x embedding_size elements)
    def get_parallel_difference_optimizer(self,learning_rate,lang1,sequences1,sequence_lengths1,lang2,sequences2,sequence_lengths2):
        hidden_representation1 = self.compute_hidden_representation(sequences1,sequence_lengths1,lang1)
        hidden_representation2 = self.compute_hidden_representation(sequences2,sequence_lengths2,lang2)

        squared_difference = tf.squared_difference(hidden_representation1,hidden_representation2)
        mean_squared_difference = tf.reduce_mean(squared_difference)
        
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(mean_squared_difference)
        return optimizer

    # Given source sequences, and target language, predict character ids sequences in target_lang
    # Explanation same as that of seq_loss
    # Output is tensorflow op
    def transliterate(self, source_lang, sequences, sequence_lengths, target_lang):
        initial_state = self.compute_hidden_representation(sequences,sequence_lengths, source_lang)
        state = initial_state
        outputs=[]
        batch_size = tf.shape(sequences)[0]

        for i in range(self.max_sequence_length):
            if(i==0):
                current_emb = tf.reshape(tf.tile(self.decoder_input[target_lang],tf.pack([batch_size,1])),[-1,self.embedding_size])
            else:
                current_emb = tf.nn.embedding_lookup(self.embed_W[target_lang],outputs[-1])+self.embed_b
            if i > 0 : tf.get_variable_scope().reuse_variables()

            output,state = self.decoder_cell[target_lang](current_emb,state)
            logit_words = tf.add(tf.matmul(output,self.out_W[target_lang]),self.out_b[target_lang])

            output_id = tf.argmax(logit_words,1)
            outputs += [output_id]

        return tf.transpose(tf.pack(outputs), perm=[1,0])

    # Arguments: target_sequences, and predicted_sequences. Both should have same size, and must be numpy arrays

    # Given 2 set of sequences, find the character-wise and word accuracy
    # For character accuracy, it skips GO character, and take the characters from next character to EOW (inclusive)
    #   actual EOW is taken from target_sequence. So order must be maintained while calling, otherwise exception may be thrown
    def get_accuracy(self, target_sequences, predicted_sequences):
        # Convert sequences from numpy array to list of lists
        target_sequences = target_sequences.tolist()
        predicted_sequences = predicted_sequences.tolist()

        # Finding actual end of words and trims the actual target sequences
        # starts from character after GO and take till EOW

        target_sequences = [ x[ 1 : x.index(u'EOW') ] for x in target_sequences]
        lengths = map(len,target_sequences)     # Actual sequence length
        num_words = len(target_sequences)
        num_chars = sum(lengths)+0.0        # Total number of character

        # Trim predicted sequences so as to match size with the actual words
        predicted_sequences_trimmed = [predicted_sequences[i][1:lengths[i]+1] for i in range(len(num_words))]

        correct_words = 0.0
        correct_chars = 0.0

        # Iterating and increment correct_word/correct_chars when word/character match resp.
        for j in range(num_words):
            if(target_sequences[j] == predicted_sequences_trimmed[j]):
                correct_words += 1
            for k in range(lengths[j]):
                if(target_sequences[j][k] == predicted_sequences_trimmed[j][k]):
                    correct_chars += 1

        # Returns (word_accuracy,character accuracy tuple)
        return (correct_words/num_words, correct_chars/num_chars)
