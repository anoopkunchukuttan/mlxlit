import argparse
import os
import sys
import codecs
import itertools as it
import pickle 
import numpy as np

import AttentionModel
import Mapping
import MonoDataReader
import ParallelDataReader

import tensorflow as tf

from indicnlp import loader

if __name__ == '__main__' :

    print 'Process started at: ' + time.asctime()

    #### Load Indic NLP Library ###
    ## Note: Environment variable: INDIC_RESOURCES_PATH must be set
    loader.load()

    #####################################
    #    Command line argument parser   #
    #####################################

    # Creating parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_seq_length', type = int, default = 30, help = 'maximum sequence length')

    parser.add_argument('--enc_type', type = str, default = 'cnn',  help = 'encoder to use. One of (1)simple_lstm_noattn (2) bilstm (3) cnn')
    parser.add_argument('--separate_output_embedding', action='store_true', default = False,  help = 'Should separate embeddings be used on the input and output side. Generally the same embeddings are to be used. This is used only for Indic-Indic transliteration, when input is phonetic and output is onehot_shared')

    parser.add_argument('--embedding_size', type = int, default = 256, help = 'size of character representation')
    parser.add_argument('--enc_rnn_size', type = int, default = 256, help = 'size of output of encoder RNN')
    parser.add_argument('--dec_rnn_size', type = int, default = 256, help = 'size of output of dec RNN')
    parser.add_argument('--representation', type = str, default = 'onehot',  help = 'input representation, which can be specified in two ways: (i) one of "phonetic", "onehot", "onehot_and_phonetic"')

    parser.add_argument('--topn', type = int, default = 10, help = 'The top-n candidates to report')
    parser.add_argument('--beam_size', type = int, default = 5, help = 'beam size for decoding')

    parser.add_argument('--lang_pair', type = str, help = 'language pair for decoding: "lang1-lang2"')

    parser.add_argument('--model_fname', type = str, help = 'model file name')
    parser.add_argument('--mapping_dir', type = str, help = 'directory containing mapping files')
    parser.add_argument('--data_dir', type = str, help = 'directory containing mapping files')
    parser.add_argument('--in_fname', type = str, help = 'input file')
    parser.add_argument('--out_fname', type = str, help = 'results file')

    args = parser.parse_args()

    print '========== Parameters start ==========='
    for k,v in vars(args).iteritems():
        print '{}: {}'.format(k,v)
    print '========== Parameters end ============='

    #Parsing arguments
    max_sequence_length = args.max_seq_length

    enc_type = args.enc_type
    embedding_size = args.embedding_size
    enc_rnn_size = args.enc_rnn_size
    dec_rnn_size = args.dec_rnn_size
    representation = None

    beam_size_val= args.beam_size
    topn_val = args.topn

    model_fname=args.model_fname
    mapping_dir = args.mapping_dir
    data_dir = args.data_dir
    in_fname=args.in_fname
    out_fname=args.out_fname

    # Setting the language parameters
    lang_pair=tuple(args.lang_pair.split('-'))

    #######################################
    # Reading data and creating mappings  #
    #######################################

    ### parse representation argument 
    if args.representation in ['onehot','onehot_shared','phonetic','onehot_and_phonetic']: 
        representation = {} 
        for lang in lang_pair: 
            representation[lang]=args.representation 
    else: 
        representation = dict([ x.split(':') for x in args.representation.split(',') ])

    ## Print Representation and Mappings 
    print 'Represenation'
    print representation 

    ## Creating mapping object to store char-id mappings
    ##lang_pairs=[lang_pair]
    ##all_langs=lang_pair

    #for lang in all_langs: 
    #    representation[lang]='phonetic'

    ###### create vocabulary from loading corpus 

    #mapping={}
    #shared_phonetic_mapping = Mapping.IndicPhoneticMapping()
    #shared_onehot_mapping = Mapping.IndicPhoneticMapping()

    #for lang in all_langs: 
    #    if representation[lang] in ['phonetic','onehot_and_phonetic']: 
    #        mapping[lang]=shared_phonetic_mapping 
    #    elif representation[lang]=='onehot_shared': 
    #        mapping[lang]=shared_onehot_mapping
    #    elif representation[lang]=='onehot': 
    #        mapping[lang]=Mapping.CharacterMapping()

    ## Reading Parallel Training data
    #parallel_train_data = dict()
    #for lang_pair in lang_pairs: 
    #    file_prefix = data_dir+'/parallel_train/'+lang_pair[0]+'-'+lang_pair[1]+'.'
    #    parallel_train_data[lang_pair] = ParallelDataReader.ParallelDataReader(lang_pair[0],lang_pair[1],
    #        file_prefix+lang_pair[0],file_prefix+lang_pair[1],mapping,max_sequence_length)

    ### complete vocabulary creation
    #for lang in all_langs:
    #    mapping[lang].finalize_vocab()
    #    print lang
    #    print '{} {}'.format(len(mapping[lang].vocab_i2c), mapping[lang].get_bitvector_embedding_size(representation[lang]))
    #    #print mapping[lang].vocab_i2c        

    #sys.exit(0)


    ### load the mapping
    mapping={}

    for lang in lang_pair: 
        if representation[lang]=='onehot': 
            mapping[lang]=Mapping.CharacterMapping()
        else: 
            mapping[lang]=Mapping.IndicPhoneticMapping()        
        with open(mapping_dir+'/'+'mapping_'+lang+'.json','r') as mapping_file:     
            mapping[lang].load_mapping(mapping_file)

    ## Print Representation and Mappings 
    print 'Mapping'
    print mapping

    test_data = MonoDataReader.MonoDataReader(lang_pair[0], in_fname,mapping[lang_pair[0]],max_sequence_length)

    ###################################################################
    #    Interacting with model and creating computation graph        #
    ###################################################################

    print "Start graph creation"
    # Creating Model object
    model = AttentionModel.AttentionModel(mapping,representation,max_sequence_length,enc_type,embedding_size,enc_rnn_size,dec_rnn_size) # Pass parameters

    ## Creating placeholder for sequences, masks and lengths and dropout keep probability 
    batch_sequences = tf.placeholder(shape=[None,max_sequence_length],dtype=tf.int32)
    batch_sequence_masks = tf.placeholder(shape=[None,max_sequence_length],dtype=tf.float32)
    batch_sequence_lengths = tf.placeholder(shape=[None],dtype=tf.float32)
    beam_size = tf.placeholder(dtype=tf.int32)
    topn = tf.placeholder(dtype=tf.int32)

    # Predict output for test sequences
    outputs, outputs_scores = model.transliterate_beam(lang_pair[0],batch_sequences,batch_sequence_lengths,lang_pair[1],beam_size, topn)

    #Saving model
    saver = tf.train.Saver(max_to_keep = 3)

    print "Done with creating graph. Starting session"

    #Start Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())
    saver.restore(sess,model_fname)
    
    print "Session started"

    source_lang = lang_pair[0]
    target_lang = lang_pair[1]
    sequences, _, sequence_lengths = test_data.get_data()

    test_start_time=time.time()
    predicted_sequences_ids, predicted_scores = sess.run([outputs, outputs_scores], feed_dict={batch_sequences: sequences, batch_sequence_lengths: sequence_lengths, beam_size: beam_size_val, topn: topn_val})
    test_end_time=time.time()

    natoms = sequences.shape[0]*max_sequence_length
    print 'Number of atoms: {}'.format(natoms)
    print 'Number of sequences: {}'format(sequences.shape[0])
    print 'Time taken (s): {}'.format(test_end_time-test_start_time)
    print 'Decoding speed: {} atoms/s, {} sequences/s'.format(
                        (test_end_time-test_start_time)/natoms,
                        (test_end_time-test_start_time)/sequences.shape[0]
                    )

    with codecs.open(out_fname,'w','utf-8') as outfile: 
        for sent_no, all_sent_predictions in enumerate(predicted_sequences_ids): 
            for rank, predicted_sequence_ids in enumerate(all_sent_predictions): 
                sent=[mapping[target_lang].get_char(x,target_lang) for x in predicted_sequence_ids]
                sent=u' '.join(it.takewhile(lambda x:x != u'EOW',it.dropwhile(lambda x:x==u'GO',sent))) 
                outfile.write(u'{} ||| {} ||| Distortion0= -1 LM0= -1 WordPenalty0= -1 PhrasePenalty0= -1 TranslationModel0= -1 -1 -1 -1 ||| {}\n'.format(sent_no,sent,predicted_scores[sent_no,rank]))

    print 'Process terminated at: ' + time.asctime()
