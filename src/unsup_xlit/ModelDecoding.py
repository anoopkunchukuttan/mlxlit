import argparse
import os
import sys
import codecs

import AttentionModel
import Mapping
import MonoDataReader
import ParallelDataReader

import tensorflow as tf

from indicnlp import loader

if __name__ == '__main__' :

    #### Load Indic NLP Library ###
    ## Note: Environment variable: INDIC_RESOURCES_PATH must be set
    loader.load()

    #####################################
    #    Command line argument parser   #
    #####################################

    # Creating parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_seq_length', type = int, default = 50, help = 'maximum sequence length')
    parser.add_argument('--embedding_size', type = int, default = 256, help = 'size of character representation and RNN')
    parser.add_argument('--representation', type = str, default = 'phonetic',  help = 'input representation, one of "phonetic", "onehot", "onehot_and_phonetic"')

    parser.add_argument('--lang_pair', type = str, help = 'language pair for decoding: "lang1-lang2"')

    parser.add_argument('--data_dir', type = str, help = 'data directory')
    parser.add_argument('--model_fname', type = str, help = 'model file name')
    parser.add_argument('--in_fname', type = str, help = 'input file')
    parser.add_argument('--out_fname', type = str, help = 'results file')

    args = parser.parse_args()

    print '========== Parameters start ==========='
    for k,v in vars(args).iteritems():
        print '{}: {}'.format(k,v)
    print '========== Parameters end ============='

    #Parsing arguments
    max_sequence_length = args.max_seq_length

    embedding_size = args.embedding_size
    representation = args.representation

    data_dir = args.data_dir
    model_fname=args.model_fname
    in_fname=args.in_fname
    out_fname=args.out_fname

    # Setting the language parameters
    lang_pair=tuple(args.lang_pair.split('-'))

    #######################################
    # Reading data and creating mappings  #
    #######################################

    # Creating mapping object to store char-id mappings
    mapping = Mapping.Mapping()

    # Reading Parallel Training data
    ### TODO this is need only because vocabulary has to be restored. Need to save vocabulary too
    parallel_train_data = dict()
    file_prefix = data_dir+'/parallel_train/'+lang_pair[0]+'-'+lang_pair[1]+'.'
    parallel_train_data[lang_pair] = ParallelDataReader.ParallelDataReader(lang_pair[0],lang_pair[1],
        file_prefix+lang_pair[0],file_prefix+lang_pair[1],mapping,max_sequence_length)

    ## complete vocabulary creation
    mapping.finalize_vocab()

    # Reading Test data
    test_data = MonoDataReader.MonoDataReader(lang_pair[0], in_fname,mapping,max_sequence_length)

    ###################################################################
    #    Interacting with model and creating computation graph        #
    ###################################################################

    # Creating Model object
    model = AttentionModel.AttentionModel(mapping,representation,embedding_size,max_sequence_length) # Pass parameters

    ## Creating placeholder for sequences, masks and lengths and dropout keep probability 
    batch_sequences = tf.placeholder(shape=[None,max_sequence_length],dtype=tf.int32)
    batch_sequence_masks = tf.placeholder(shape=[None,max_sequence_length],dtype=tf.float32)
    batch_sequence_lengths = tf.placeholder(shape=[None],dtype=tf.float32)

    # Predict output for test sequences
    infer_output = model.transliterate(lang_pair[0],batch_sequences,batch_sequence_lengths,lang_pair[1])

    #Saving model
    saver = tf.train.Saver(max_to_keep = 3)
    final_saver = tf.train.Saver()

    print "Done with creating graph. Starting session"

    #Start Session
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver.restore(sess,model_fname)
    
    print "Session started"

    source_lang = lang_pair[0]
    target_lang = lang_pair[1]
    sequences, _, sequence_lengths = test_data.get_data()
    predicted_sequences_ids = sess.run(infer_output, feed_dict={batch_sequences: sequences, batch_sequence_lengths: sequence_lengths})
    predicted_sequences = mapping.get_words_from_id_lists(predicted_sequences_ids,target_lang)
    codecs.open(out_fname,'w','utf-8').write(u'\n'.join(predicted_sequences))

