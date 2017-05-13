import argparse
import os
import sys
import codecs
import itertools as it
import pickle 
import numpy as np
import calendar,time

import AttentionModel
import LanguageModel
import Mapping
import MonoDataReader
import ParallelDataReader
import utilities

import tensorflow as tf

from indicnlp import loader

def get_average_loss(mono_data, loss_op,
        pl_batch_sequences, pl_batch_sequence_lengths, sess):
    """
    convenience function to compute the translation loss (cross entropy) 

    lang_pairs: list of language pair tuples. 
    parallel_data: Dictionary of ParallelDataReader object for various language pairs                 

    seq_loss_op: dictionary of sequence loss operation for every language pair 
    batch_sequences, batch_sequence_masks, batch_sequence_lengths,
        batch_sequences_2, batch_sequence_masks_2, batch_sequence_lengths_2,
        dropout_keep_prob: placeholder variables 

    prefix_srclang: see commandline flags 
    prefix_tgtlang: see commandline flags 
    mapping: Dictionary of mapping objects for each language
    """

    ### start computation
    validation_loss = 0.0
    
    sequences,sequence_masks,sequence_lengths = mono_data.get_data()
    
    validation_loss += sess.run(loss_op, feed_dict = {
        pl_batch_sequences:sequences,pl_batch_sequence_lengths:sequence_lengths })

    return validation_loss        

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
    parser.add_argument('--batch_size', type = int, default = 100, help = 'size of each batch used in decoding')

    parser.add_argument('--enc_type', type = str, default = 'cnn',  help = 'encoder to use. One of (1)simple_lstm_noattn (2) bilstm (3) cnn')
    parser.add_argument('--separate_output_embedding', action='store_true', default = False,  help = 'Should separate embeddings be used on the input and output side. Generally the same embeddings are to be used. This is used only for Indic-Indic transliteration, when input is phonetic and output is onehot_shared')
    parser.add_argument('--prefix_tgtlang', action='store_true', default = False,
            help = 'Prefix the input sequence with the language code for the target language')
    parser.add_argument('--prefix_srclang', action='store_true', default = False,
            help = 'Prefix the input sequence with the language code for the source language')

    parser.add_argument('--embedding_size', type = int, default = 256, help = 'size of character representation')
    parser.add_argument('--enc_rnn_size', type = int, default = 512, help = 'size of output of encoder RNN')
    parser.add_argument('--dec_rnn_size', type = int, default = 512, help = 'size of output of dec RNN')
    parser.add_argument('--representation', type = str, default = 'onehot',  help = 'input representation, which can be specified in two ways: (i) one of "phonetic", "onehot", "onehot_and_phonetic"')
    parser.add_argument('--shared_mapping_class', type = str, default = 'IndicPhoneticMapping',  help = 'class to be used for shared mapping. Possible values: IndicPhoneticMapping, CharacterMapping')

    parser.add_argument('--topn', type = int, default = 10, help = 'The top-n candidates to report')
    parser.add_argument('--beam_size', type = int, default = 5, help = 'beam size for decoding')

    parser.add_argument('--lang_pair', type = str, help = 'language pair for decoding: "lang1-lang2"')

    parser.add_argument('--model_fname', type = str, help = 'model file name')
    parser.add_argument('--mapping_dir', type = str, help = 'directory containing mapping files')
    parser.add_argument('--in_fname', type = str, help = 'input file')
    parser.add_argument('--out_fname', type = str, help = 'results file')

    parser.add_argument('--fuse_lm', type = str, help = 'Decode with shallow LM fusion. The path to the LM model file must be supplied.')
    parser.add_argument('--lm_weight', type = float, help = 'Weight of the LM score while linear interpolating with the NMT model score')
    parser.add_argument('--lm_embedding_size', type = int, default = 256, help = 'size of character representation')
    parser.add_argument('--lm_rnn_size', type = int, default = 512, help = 'size of output of encoder RNN')
    parser.add_argument('--lm_max_seq_length', type = int, default = 30, help = 'maximum sequence length')

    args = parser.parse_args()

    print '========== Parameters start ==========='
    for k,v in vars(args).iteritems():
        print '{}: {}'.format(k,v)
    print '========== Parameters end ============='

    #Parsing arguments
    ## paths and directories
    model_fname=args.model_fname
    mapping_dir = args.mapping_dir
    in_fname=args.in_fname
    out_fname=args.out_fname

    ## architecture
    enc_type = args.enc_type
    separate_output_embedding = args.separate_output_embedding
    prefix_tgtlang = args.prefix_tgtlang
    prefix_srclang = args.prefix_srclang

    embedding_size = args.embedding_size
    enc_rnn_size = args.enc_rnn_size
    dec_rnn_size = args.dec_rnn_size
    representation = None
    shared_mapping_class = args.shared_mapping_class

    ## other hyperparameters  
    max_sequence_length = args.max_seq_length
    batch_size = args.batch_size 

    ## decoding
    beam_size_val= args.beam_size
    topn_val = args.topn

    ## LM parameters for shallow fusion 
    fuse_lm = args.fuse_lm 
    lm_weight = args.lm_weight

    # Setting the language parameters
    lang_pair=tuple(args.lang_pair.split('-'))
    source_lang = lang_pair[0]
    target_lang = lang_pair[1]

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
    print 'Representation'
    print representation 

    ### load the mapping
    mapping={}
    shared_mapping_obj = Mapping.get_mapping_instance(shared_mapping_class) 

    for lang in representation.keys(): 
        if representation[lang] in ['phonetic','onehot_and_phonetic']: 
            mapping[lang]=shared_mapping_obj
        elif representation[lang]=='onehot_shared': 
            mapping[lang]=shared_mapping_obj
        elif representation[lang]=='onehot': 
            mapping[lang]=Mapping.CharacterMapping()

        with open(mapping_dir+'/'+'mapping_'+lang+'.json','r') as mapping_file:     
            mapping[lang].load_mapping(mapping_file)

    ## Print Representation and Mappings 
    print 'Mapping'
    print mapping

    print 'Vocabulary Statitics'
    for lang in representation.keys(): 
        print '{}: {}'.format(lang,mapping[lang].get_vocab_size())


    print 'Mapping Again'
    print mapping

    print 'Vocabulary Statitics Again'
    for lang in representation.keys(): 
        print '{}: {}'.format(lang,mapping[lang].get_vocab_size())
    sys.stdout.flush()

    ###################################################################
    #    Interacting with model and creating computation graph        #
    ###################################################################

    ## Creating placeholder for sequences, masks and lengths and dropout keep probability 
    print "Create placeholder variables"
    batch_sequences = tf.placeholder(shape=[None,max_sequence_length],dtype=tf.int32)
    batch_sequence_masks = tf.placeholder(shape=[None,max_sequence_length],dtype=tf.float32)
    batch_sequence_lengths = tf.placeholder(shape=[None],dtype=tf.float32)
    beam_size = tf.placeholder(dtype=tf.int32)
    topn = tf.placeholder(dtype=tf.int32)

    ## Create session 
    print "Creating Session"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    #### Creating Translation Model graph creation 
    print "Start graph creation for Translation Model"
    model = AttentionModel.AttentionModel(mapping,representation,max_sequence_length,
            embedding_size,enc_rnn_size,dec_rnn_size,
            enc_type,separate_output_embedding)

    # Predict output for test sequences
    ### a secondary purpose for crearing this graph is to allow loading of variables 
    outputs, outputs_scores = model.transliterate_beam(
                lang_pair[0],batch_sequences,batch_sequence_lengths,lang_pair[1],beam_size, topn)
    
    ### now restore variables from translation model 
    saver_trans = tf.train.Saver()
    saver_trans.restore(sess,model_fname)
    print "Loaded translation model parameters"

    # Creating Language Model graph creation 
    if fuse_lm is not None: 

        print "Start graph creation for Language Model"
        lm_model=LanguageModel.LanguageModel(target_lang,mapping[target_lang],representation[target_lang], \
                    args.lm_max_seq_length, args.lm_embedding_size, args.lm_rnn_size)

        loss_op=None
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            loss_op = lm_model.average_loss(batch_sequences, batch_sequence_lengths, 1.0)
        #tf.get_variable_scope().reuse_variables()

        ### now restore variables from language model 

        # the RNN's variables have to be explicitly obtained 
        lm_mat =tf.get_variable('lang_model/BasicLSTMCell/Linear/Matrix')
        lm_bias=tf.get_variable('lang_model/BasicLSTMCell/Linear/Bias')
        ## load only the LM variables. Trying to load all variables will fail to load trans model variables
        vars_to_restore = [
                            lm_model.Wmat,
                            lm_model.embed_b,
                            lm_model.out_W,
                            lm_model.out_b,
                            lm_mat,
                            lm_bias
                          ]
        saver_lm = tf.train.Saver(vars_to_restore)
        saver_lm.restore(sess,fuse_lm)
        print "Loaded language model parameters"

        ## sample code if the language model has successfully loaded. Its a hack, use carefully
        #test_data = MonoDataReader.MonoDataReader(lang_pair[1], in_fname, mapping[lang_pair[1]],max_sequence_length)
        #ppl=get_average_loss(test_data, loss_op, batch_sequences, batch_sequence_lengths, sess)
        #print 'Perplexity: {}'.format(ppl)

        ## Now prepare the translation model to fuse with the language model 
        model.initialize_lm(lm_model,lm_weight)
        outputs, outputs_scores = model.transliterate_beam_with_lm(
                lang_pair[0],batch_sequences,batch_sequence_lengths,lang_pair[1],beam_size, topn)
        print "Prepared transation model for shallow LM fusion" 

    print "Reading testdata"

    test_data = MonoDataReader.MonoDataReader(lang_pair[0], in_fname,mapping[lang_pair[0]],max_sequence_length)
    sequences, sequence_masks, sequence_lengths = test_data.get_data()

    test_time=0.0
    predicted_sequences_ids_list=[]
    predicted_scores_list=[]

    print 'starting execution'
    for start in xrange(0,sequences.shape[0],batch_size):
        end = min(start+batch_size,sequences.shape[0])

        batch_start_time=time.time()

        data_sequences=sequences[start:end,:]
        data_sequence_masks=sequence_masks[start:end,:]
        data_sequence_lengths=sequence_lengths[start:end]

        if prefix_tgtlang: 
            data_sequences,data_sequence_masks,data_sequence_lengths = Mapping.prefix_sequence_with_token(
                    data_sequences,data_sequence_masks,data_sequence_lengths, 
                    target_lang,mapping[target_lang])

        if prefix_srclang: 
            data_sequences,data_sequence_masks,data_sequence_lengths = Mapping.prefix_sequence_with_token(
                    data_sequences,data_sequence_masks,data_sequence_lengths, 
                feed_dict={batch_sequences: data_sequences, batch_sequence_lengths: data_sequence_lengths, 
                    beam_size: beam_size_val, topn: topn_val})

        b_sequences_ids, b_scores = sess.run([outputs, outputs_scores], 
                feed_dict={batch_sequences: data_sequences, batch_sequence_lengths: data_sequence_lengths, 
                    beam_size: beam_size_val, topn: topn_val})
        predicted_sequences_ids_list.append(b_sequences_ids)
        predicted_scores_list.append(b_scores)

        batch_end_time=time.time()
        test_time+=(batch_end_time-batch_start_time)

        print 'Decoded {} of {} sequences'.format(end,sequences.shape[0])
        sys.stdout.flush()

    predicted_sequences_ids=np.concatenate(predicted_sequences_ids_list,axis=0)
    predicted_scores=np.concatenate(predicted_scores_list,axis=0)

    natoms = sequences.shape[0]*max_sequence_length
    print 'Number of atoms: {}'.format(natoms)
    print 'Number of sequences: {}'.format(sequences.shape[0])
    print 'Time taken (hh:mm:ss): {}'.format(utilities.formatted_timeinterval(test_time))
    print 'Decoding speed: {} atoms/s, {} sequences/s'.format(
                        natoms/test_time,
                        sequences.shape[0]/test_time,
                    )

    with codecs.open(out_fname,'w','utf-8') as outfile: 
        for sent_no, all_sent_predictions in enumerate(predicted_sequences_ids): 
            for rank, sequence_at_rank in enumerate(all_sent_predictions): 
                sent=[mapping[target_lang].get_char(x,target_lang) for x in sequence_at_rank]
                sent=u' '.join(it.takewhile(lambda x:x != u'EOW',it.dropwhile(lambda x:x==u'GO',sent))) 
                outfile.write(u'{} ||| {} ||| Distortion0= -1 LM0= -1 WordPenalty0= -1 PhrasePenalty0= -1 TranslationModel0= -1 -1 -1 -1 ||| {}\n'.format(sent_no,sent,predicted_scores[sent_no,rank]))

    print 'Process terminated at: ' + time.asctime()
