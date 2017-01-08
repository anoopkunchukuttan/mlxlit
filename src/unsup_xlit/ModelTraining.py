import argparse
import os
import sys
import codecs

import Model
import AttentionModel
import Mapping
import MonoDataReader
import ParallelDataReader

import itertools as it
import tensorflow as tf
import numpy as np

import calendar,time
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

    parser.add_argument('--data_dir', type = str, help = 'data directory')
    parser.add_argument('--output_dir', type = str, help = 'output folder name')

    parser.add_argument('--lang_pairs', type = str, default = None, help = 'List of language pairs for supervised training given as: "lang1-lang2,lang3-lang4,..."')
    parser.add_argument('--langs', type = str, default = None, help = 'List of language for unsupervised training given as: "lang1,lang2,lang3,lang4,..."')

    parser.add_argument('--batch_size', type = int, default = 64, help = 'size of each batch used in training')
    parser.add_argument('--max_epochs', type = int, default = 64, help = 'maximum number of epochs')
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'learning rate of Adam Optimizer')
    parser.add_argument('--dropout_keep_prob', type = float, default = 0.5, help = 'keep probablity for the dropout layers')
    parser.add_argument('--max_seq_length', type = int, default = 30, help = 'maximum sequence length')
    parser.add_argument('--infer_every', type = int, default = 1, help = 'write predicted outputs for test data after these many epochs, 0 if not required')

    parser.add_argument('--embedding_size', type = int, default = 256, help = 'size of character representation')
    parser.add_argument('--enc_rnn_size', type = int, default = 256, help = 'size of output of encoder RNN')
    parser.add_argument('--dec_rnn_size', type = int, default = 256, help = 'size of output of dec RNN')

    parser.add_argument('--topn', type = int, default = 10, help = 'The top-n candidates to report')
    parser.add_argument('--beam_size', type = int, default = 5, help = 'beam size for decoding')

    parser.add_argument('--start_from', type = int, default = None, help = 'epoch to restore model from. This must be one of the final epochs from previous runs')

    parser.add_argument('--representation', type = str, default = 'onehot',  help = 'input representation, which can be specified in two ways: (i) one of "phonetic", "onehot", "onehot_and_phonetic"')

    parser.add_argument('--train_mode', type = str, default = 'sup', help = 'one of "unsup" for unsupervised learning, "sup" for supervised learning')
    parser.add_argument('--train_bidirectional', action = 'store_true', default = False, help = 'Train in both directions. Applicable for supervised learning only')
    parser.add_argument('--use_monolingual', action = 'store_true' , default = False, help = 'Use additional monolingual data in addition to parallel training data for monolingual reconstruction. Applicable for supervised learning only')
    parser.add_argument('--which_mono', type = int, default = -100, help = 'which monolingual to use (hack code, may not work - must be commented)')

    parser.add_argument('--train_size', type = int, default = -1, help = 'Size of the parallel training set to use for all language pairs (not implemented yet)')


    args = parser.parse_args()

    print '========== Parameters start ==========='
    for k,v in vars(args).iteritems():
        print '{}: {}'.format(k,v)
    print '========== Parameters end ============='

    #Parsing arguments
    batch_size = args.batch_size
    max_epochs = args.max_epochs
    learning_rate = args.learning_rate
    max_sequence_length = args.max_seq_length
    infer_every = args.infer_every
    dropout_keep_prob_val = args.dropout_keep_prob

    embedding_size = args.embedding_size
    enc_rnn_size = args.enc_rnn_size
    dec_rnn_size = args.dec_rnn_size
    representation = None

    beam_size_val= args.beam_size
    topn_val = args.topn

    train_mode = args.train_mode
    train_bidirectional = args.train_bidirectional 
    use_monolingual = args.use_monolingual 

    if train_mode=='unsup' and args.langs is None:
        print 'ERROR: --langs  has to be set for "unsup" mode'
        sys.exit(1)
    elif train_mode == 'sup' and args.lang_pairs is None:
        print 'ERROR: --lang_pairs has to be set for "{}" mode'.format(train_mode)
        sys.exit(1)
    elif train_mode=='unsup' and args.lang_pairs is not None:
        print 'WARNING:--lang_pairs is not valid for "unsup" mode, ignoring parameter'
    elif train_mode == 'sup' and args.langs is not None:
        print 'WARNING:--langs is not valid for "{}" mode, ignoring parameter'.format(train_mode)

    data_dir = args.data_dir
    output_dir = args.output_dir
    start_from = args.start_from

    # Setting the language parameters
    mono_langs=None
    parallel_train_langs=None
    parallel_valid_langs=None
    test_langs=None
    all_langs=None

    if train_mode=='unsup':
        parallel_train_langs=[]
        mono_langs=args.langs.split(',')
        parallel_valid_langs=list(it.combinations(mono_langs,2))
        test_langs = list(it.permutations(mono_langs,2))
        all_langs=mono_langs 

    elif train_mode=='sup':

        parallel_train_langs=[ tuple(lp.split('-')) for lp in args.lang_pairs.split(',')]
        parallel_valid_langs=parallel_train_langs

        mll=set()
        for lp in [list(x) for x in parallel_train_langs]: 
            mll.update(lp)
        all_langs=list(mll)

        if use_monolingual:             
            mono_langs=list(mll)                

            ### NOTE: temporary - use only source for monolingual optimization (works only for a single pair)
            #mono_langs=[parallel_train_langs[0][args.which_mono]]
        else:
            mono_langs=[]

        if train_bidirectional:             
            test_langs= parallel_train_langs + [ tuple(reversed(x)) for x in parallel_train_langs ]
        else: 
            test_langs=parallel_train_langs 

    print 'Parallel Train, Mono, Parallel Valid, Test Langs, All Langs'
    print parallel_train_langs
    print mono_langs
    print parallel_valid_langs
    print test_langs
    print all_langs 

    # Create output folders if required
    temp_model_output_dir = output_dir+'/temp_models/'
    outputs_dir = output_dir+'/outputs/'
    final_output_dir = output_dir+'/final_output/'
    log_dir = output_dir+'/log/'

    for folder in [temp_model_output_dir, outputs_dir, final_output_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    #######################################
    # Reading data and creating mappings  #
    #######################################

    ### parse representation argument 
    if args.representation in ['onehot','phonetic','onehot_and_phonetic']: 
        representation = {} 
        for lang in all_langs: 
            representation[lang]=args.representation 
    else: 
        representation = dict([ x.split(':') for x in args.representation.split(',') ])

    # Creating mapping object to store char-id mappings
    mapping={}
    phonetic_mapping = Mapping.IndicPhoneticMapping()

    for lang in all_langs: 
        if representation[lang] in ['phonetic','onehot_and_phonetic']: 
            mapping[lang]=phonetic_mapping 
        elif representation[lang]=='onehot': 
            mapping[lang]=Mapping.CharacterMapping()

    ## Print Representation and Mappings 
    print 'Represenation'
    print representation 

    print 'Mapping'
    print mapping

    print 'Start Reading Data'

    # Reading Monolingual Training data
    mono_train_data = dict()
    for lang in mono_langs:
        mono_train_data[lang] = MonoDataReader.MonoDataReader(lang,data_dir+'/mono_train/'+lang,mapping[lang],max_sequence_length)

    # Reading Parallel Training data
    parallel_train_data = dict()
    for lang_pair in parallel_train_langs:
        file_prefix = data_dir+'/parallel_train/'+lang_pair[0]+'-'+lang_pair[1]+'.'
        parallel_train_data[lang_pair] = ParallelDataReader.ParallelDataReader(lang_pair[0],lang_pair[1],
            file_prefix+lang_pair[0],file_prefix+lang_pair[1],mapping,max_sequence_length)

    ## complete vocabulary creation
    for lang in all_langs: 
        mapping[lang].finalize_vocab()

    print 'Vocabulary Statitics'
    for lang in all_langs: 
        print '{}: {}'.format(lang,mapping[lang].get_vocab_size())

    # Reading parallel Validation data
    parallel_valid_data = dict()
    for lang_pair in parallel_valid_langs:
        file_prefix = data_dir+'/parallel_valid/'+lang_pair[0]+'-'+lang_pair[1]+'.'
        parallel_valid_data[lang_pair] = ParallelDataReader.ParallelDataReader(lang_pair[0],lang_pair[1],
            file_prefix+lang_pair[0],file_prefix+lang_pair[1],mapping,max_sequence_length)

    # Reading Test data
    test_data = dict()
    for lang_pair in test_langs:
        file_name = data_dir+'/test/'+lang_pair[0]+'-'+lang_pair[1]
        test_data[lang_pair] = MonoDataReader.MonoDataReader(lang_pair[0],
            file_name,mapping[lang],max_sequence_length)

    print 'Stop Reading Data' 

    ###################################################################
    #    Interacting with model and creating computation graph        #
    ###################################################################

    # Creating Model object
    model = AttentionModel.AttentionModel(mapping,representation,max_sequence_length,embedding_size,enc_rnn_size,dec_rnn_size) # Pass parameters

    ## Creating placeholder for sequences, masks and lengths and dropout keep probability 
    batch_sequences = tf.placeholder(shape=[None,max_sequence_length],dtype=tf.int32)
    batch_sequence_masks = tf.placeholder(shape=[None,max_sequence_length],dtype=tf.float32)
    batch_sequence_lengths = tf.placeholder(shape=[None],dtype=tf.float32)

    batch_sequences_2 = tf.placeholder(shape=[None,max_sequence_length],dtype=tf.int32)
    batch_sequence_masks_2 = tf.placeholder(shape=[None,max_sequence_length],dtype=tf.float32)
    batch_sequence_lengths_2 = tf.placeholder(shape=[None],dtype=tf.float32)
    
    dropout_keep_prob = tf.placeholder(dtype=tf.float32)

    beam_size = tf.placeholder(dtype=tf.int32)
    topn = tf.placeholder(dtype=tf.int32)
    
    # Optimizers for training using monolingual data
    # Has only one optimizer which minimizes loss of sequence reconstruction
    unsup_optimizer = dict()
    for lang in mono_langs:
        unsup_optimizer[lang] = model.get_mono_optimizer(learning_rate,lang,batch_sequences,batch_sequence_masks,batch_sequence_lengths,dropout_keep_prob)

    # Optimizers for training using parallel data
    # For each language pair, there are 3 optimizers:
    # 1. Minimize loss for transliterating first language to second
    # 2. Minimize loss for transliterating second language to first
    # 3. Minimize difference between the hidden representations
    #  
    # It is possible to take combinations of these losses 
    #  
    #  (a) 1,2,3
    #  (b) 1+2+3 
    #  (c) 1+2,3


    sup_optimizer = dict()
    for lang1,lang2 in parallel_train_langs:

        if train_bidirectional: 

            ## (a) each loss optimized separately 

            sup_optimizer[(lang1,lang2)] = [
                model.get_parallel_optimizer(learning_rate,
                    lang1,batch_sequences,batch_sequence_masks,batch_sequence_lengths,
                    lang2,batch_sequences_2,batch_sequence_masks_2,batch_sequence_lengths_2,dropout_keep_prob),
                model.get_parallel_optimizer(learning_rate,
                    lang2,batch_sequences_2,batch_sequence_masks_2,batch_sequence_lengths_2,
                    lang1,batch_sequences,batch_sequence_masks,batch_sequence_lengths,dropout_keep_prob),
                model.get_parallel_difference_optimizer(learning_rate,
                    lang1,batch_sequences,batch_sequence_lengths,
                    lang2,batch_sequences_2,batch_sequence_lengths_2,dropout_keep_prob)]

            #### (b) sum of all losses 
            #sup_optimizer[(lang1,lang2)] = [
            #    model.get_parallel_all_optimizer(learning_rate,
            #        lang1,batch_sequences,batch_sequence_masks,batch_sequence_lengths,
            #        lang2,batch_sequences_2,batch_sequence_masks_2,batch_sequence_lengths_2,dropout_keep_prob),
            #        ]

            #### (c) optimize separately: (i) sum of translation losses (ii) representation loss
            #sup_optimizer[(lang1,lang2)] = [
            #    model.get_parallel_bi_optimizer(learning_rate,
            #        lang1,batch_sequences,batch_sequence_masks,batch_sequence_lengths,
            #        lang2,batch_sequences_2,batch_sequence_masks_2,batch_sequence_lengths_2,dropout_keep_prob),
            #    model.get_parallel_difference_optimizer(learning_rate,
            #        lang1,batch_sequences,batch_sequence_lengths,
            #        lang2,batch_sequences_2,batch_sequence_lengths_2,dropout_keep_prob),
            #       ]
        else: 

            sup_optimizer[(lang1,lang2)] = [
                model.get_parallel_optimizer(learning_rate,
                    lang1,batch_sequences,batch_sequence_masks,batch_sequence_lengths,
                    lang2,batch_sequences_2,batch_sequence_masks_2,batch_sequence_lengths_2,dropout_keep_prob),
                    ]

    # Finding validation sequence loss
    # For each pair of language, return sum of loss of transliteration one script to another and vice versa
    validation_seq_loss = dict()

    for lang_pair in parallel_valid_langs:
        lang1,lang2=lang_pair

        ## TODO: see if the 'unsup condition is required'
        if (train_mode=='sup' and train_bidirectional) or train_mode=='unsup':
            validation_seq_loss[lang_pair] = model.seq_loss_2(
                    lang1,batch_sequences,batch_sequence_masks,batch_sequence_lengths,
                    lang2,batch_sequences_2,batch_sequence_masks_2,batch_sequence_lengths_2,dropout_keep_prob) \
                + model.seq_loss_2(
                    lang2,batch_sequences_2,batch_sequence_masks_2,batch_sequence_lengths_2,
                    lang1,batch_sequences,batch_sequence_masks,batch_sequence_lengths,dropout_keep_prob)
        else:
            validation_seq_loss[lang_pair] = model.seq_loss_2(
                    lang1,batch_sequences,batch_sequence_masks,batch_sequence_lengths,
                    lang2,batch_sequences_2,batch_sequence_masks_2,batch_sequence_lengths_2,dropout_keep_prob)

    # Predict output for test sequences
    infer_output = dict()
    infer_output_scores = dict()
    for lang_pair in test_langs:
        infer_output[lang_pair], infer_output_scores[lang_pair] =  \
            model.transliterate_beam(lang_pair[0],batch_sequences,batch_sequence_lengths,lang_pair[1],beam_size,topn)

    # All training dataset
    training_langs = mono_langs+parallel_train_langs

    # Fractional epoch: stores what fraction of each dataset is used till now after last completed epoch.
    fractional_epochs = [0.0 for _ in training_langs]
    completed_epochs = 0

    #Saving model
    saver = tf.train.Saver(max_to_keep = 3)
    final_saver = tf.train.Saver()

    print "Done with creating graph. Starting session"

    #Start Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())
    if(start_from is not None):
        saver.restore(sess,'{}/final_model_epochs_{}'.format(output_dir,start_from))
        completed_epochs=start_from
    
    tf.train.SummaryWriter(log_dir,sess.graph)

    print "Session started"

    steps = 0
    validation_losses = []

    # Whether to continue or now
    cont = True

    while cont:
        # Selected the dataset whose least fraction is used for training in current epoch
        # idx = fractional_epochs.index(min(fractional_epochs))
        # opti_lang = training_langs[idx]
        for (opti_lang,idx) in zip(training_langs,range(len(training_langs))):
            if(type(opti_lang) is str):     # If it is a monolingual dataset, call optimizer
                lang = opti_lang
                sequences,sequence_masks,sequence_lengths = mono_train_data[lang].get_next_batch(batch_size)
                sess.run(unsup_optimizer[lang], feed_dict = {batch_sequences:sequences,batch_sequence_masks:sequence_masks,batch_sequence_lengths:sequence_lengths,dropout_keep_prob:dropout_keep_prob_val})
                fractional_epochs[idx] += float(len(sequences))/mono_train_data[opti_lang].num_words
            else:                           # If it is a bilingual dataset, call corresponding optimizers
                lang1 = opti_lang[0]
                lang2 = opti_lang[1]

                sequences,sequence_masks,sequence_lengths,sequences_2,sequence_masks_2,sequence_lengths_2 = parallel_train_data[opti_lang].get_next_batch(batch_size)
                
                sess.run(sup_optimizer[opti_lang], feed_dict = {
                    batch_sequences:sequences,batch_sequence_masks:sequence_masks,batch_sequence_lengths:sequence_lengths,
                    batch_sequences_2:sequences_2,batch_sequence_masks_2:sequence_masks_2,batch_sequence_lengths_2:sequence_lengths_2,
                    dropout_keep_prob:dropout_keep_prob_val
                    })

                fractional_epochs[idx] += float(len(sequences))/parallel_train_data[opti_lang].num_words

        # One more batch is processed
        steps+=1
        # If all datasets are used for training epoch is complete
        if(min(fractional_epochs) >= 1.0):
            completed_epochs += 1
            fractional_epochs = [0.0 for _ in mono_langs+parallel_train_langs]

            # Find validation loss
            validation_loss = 0.0
            for lang_pair in parallel_valid_langs:
                sequences,sequence_masks,sequence_lengths,sequences_2,sequence_masks_2,sequence_lengths_2 = parallel_valid_data[lang_pair].get_data()
                validation_loss += sess.run(validation_seq_loss[lang_pair], feed_dict = {
                    batch_sequences:sequences,batch_sequence_masks:sequence_masks,batch_sequence_lengths:sequence_lengths,
                    batch_sequences_2:sequences_2,batch_sequence_masks_2:sequence_masks_2,batch_sequence_lengths_2:sequence_lengths_2,
                    dropout_keep_prob:1.0
                    })
            validation_losses.append(validation_loss)

            print "Epochs Completed : "+str(completed_epochs).zfill(3)+"\t Validation loss: "+str(validation_loss)
            sys.stdout.flush()

            #### Comment it to avoid early stopping
            ## If validation loss is increasing since last 3 epochs, take the last 4th model and stop training process
            #if(completed_epochs>=4 and len(validation_losses)>=4 and all([i>j for (i,j) in zip(validation_losses[-3:],validation_losses[-4:-1])])):
            #    completed_epochs -= 3
            #    saver.restore(sess,temp_model_output_dir+'my_model-'+str(completed_epochs))
            #    cont = False

            # If max_epochs are done
            if(completed_epochs >= max_epochs):
                cont = False

            if(cont == False or (infer_every > 0 and completed_epochs%infer_every == 0)):
                # If this was the last epoch, output result to final output folder, otherwise to outputs folder
                if(cont == False):
                    out_folder = final_output_dir
                else:
                    out_folder = outputs_dir

                accuracies = []
                for lang_pair in test_langs:
                    source_lang = lang_pair[0]
                    target_lang = lang_pair[1]
                    sequences, _, sequence_lengths = test_data[lang_pair].get_data()
                    predicted_sequences_ids, predicted_scores = sess.run([infer_output[lang_pair],infer_output_scores[lang_pair]], feed_dict={batch_sequences: sequences, batch_sequence_lengths: sequence_lengths, beam_size: beam_size_val, topn: topn_val})
                    if completed_epochs % infer_every == 0:
                        #codecs.open(out_folder+str(completed_epochs).zfill(3)+source_lang+'-'+target_lang+'_','w','utf-8').write(u'\n'.join(predicted_sequences))
                        out_fname=out_folder+str(completed_epochs).zfill(3)+'test.nbest.'+source_lang+'-'+target_lang+'.'+target_lang
                        with codecs.open(out_fname,'w','utf-8') as outfile: 
                            for sent_no, all_sent_predictions in enumerate(predicted_sequences_ids): 
                                for rank, predicted_sequence_ids in enumerate(all_sent_predictions): 
                                    sent=[mapping[target_lang].get_char(x,target_lang) for x in predicted_sequence_ids]
                                    sent=u' '.join(it.takewhile(lambda x:x != u'EOW',it.dropwhile(lambda x:x==u'GO',sent))) 
                                    outfile.write(u'{} ||| {} ||| Distortion0= -1 LM0= -1 WordPenalty0= -1 PhrasePenalty0= -1 TranslationModel0= -1 -1 -1 -1 ||| {}\n'.format(sent_no,sent,predicted_scores[sent_no,rank]))

            # Save current model
            if(cont == True):
                if(completed_epochs==1 or (len(validation_losses)>=2 and validation_losses[-1]<validation_losses[-2])):
                    saver.save(sess, temp_model_output_dir+'my_model', global_step=completed_epochs)

    # save final model
    final_saver.save(sess,output_dir+'/final_model_epochs_'+str(completed_epochs))
