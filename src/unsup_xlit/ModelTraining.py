import argparse
import os
import sys
import codecs

import Model
import Mapping
import MonoDataReader
import ParallelDataReader

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
    parser.add_argument('--batch_size', type = int, default = 64, help = 'size of each batch used in training')
    parser.add_argument('--max_epochs', type = int, default = 32, help = 'maximum number of epochs')
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'learning rate of Adam Optimizer')
    parser.add_argument('--max_seq_length', type = int, default = 50, help = 'maximum sequence length')
    parser.add_argument('--infer_every', type = int, default = 1, help = 'write predicted outputs for test data after these many epochs, 0 if not required')

    parser.add_argument('--embedding_size', type = int, default = 256, help = 'size of character representation and RNN')
    parser.add_argument('--representation', type = str, default = 'phonetic',  help = 'input representation, one of "phonetic" or "onehot"')
    parser.add_argument('--train_mode', type = str, default = 'semisup', help = 'one of "semisup" for semi-supervised or "unsup" for unsupervised learning')
    parser.add_argument('--lang_pairs', type = str, default = None, help = 'List of language pairs for supervised training given as: "lang1-lang2,lang3-lang4,..."')
    parser.add_argument('--langs', type = str, default = None, help = 'List of language for unsupervised training given as: "lang1,lang2,lang3,lang4,..."')

    parser.add_argument('--data_dir', type = str, help = 'data directory')
    parser.add_argument('--output_dir', type = str, help = 'output folder name')
    parser.add_argument('--start_from', type = str, default = None, help = 'location of saved model, to be used for starting')

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

    embedding_size = args.embedding_size
    representation = args.representation
    train_mode = args.train_mode
    if train_mode=='unsup' and args.langs is None:  
        print 'ERROR: --langs  has to be set for "unsup" mode'
        sys.exit(1)
    elif train_mode=='semisup' and args.lang_pairs is None:  
        print 'ERROR: --lang_pairs has to be set for "semisup" mode'
        sys.exit(1)
    elif train_mode=='unsup' and args.lang_pairs is not None:  
        print 'WARNING:--lang_pairs is not valid for "unsup" mode, ignoring parameter'
    elif train_mode=='semisup' and args.langs is not None:  
        print 'WARNING:--langs is not valid for "semisup" mode, ignoring parameter'

    data_dir = args.data_dir
    output_dir = args.output_dir #+'_e'+str(embedding_size)+'_b'+str(batch_size)+'_lr'+str(learning_rate)+'_'+str(calendar.timegm(time.gmtime()))
    start_from = args.start_from
    if start_from is not None:
        assert os.path.exists(start_from), "start_from: '"+ start_from +"'' file does not exist"


    # Setting the language parameters 
    mono_langs=None
    parallel_train_langs=None
    parallel_valid_langs=None
    test_langs=None

    ##NOTE: Supports only one language pair currently
    if train_mode=='semisup':
        parallel_train_langs=[lp.split('-') for lp in args.lang_pairs.split(',')]
        if len(parallel_train_langs)>1: 
            print('Currently only one language pair is supported')
            sys.exit(0)
        mono_langs=parallel_train_langs[0]            
        parallel_valid_langs=parallel_train_langs 
        test_langs=[parallel_train_langs[0],list(reversed(parallel_train_langs[0]))]
    
    elif train_mode=='unsup':
        parallel_train_langs=[]
        mono_langs=args.langs.split(',')
        if len(mono_langs)>2: 
            print('Currently only one language pair is supported')
            sys.exit(0)
        parallel_valid_langs=[mono_langs]
        test_langs =[mono_langs,list(reversed(mono_langs))]

    print 'Parallel Train, Mono, Parallel Valid, Test Langs'
    print parallel_train_langs 
    print mono_langs
    print parallel_valid_langs 
    print test_langs 

    # Create output folders if required
    temp_model_output_dir = output_dir+'/temp_models/'
    outputs_dir = output_dir+'/outputs/'
    final_output_dir = output_dir+'/final_output/'

    for folder in [temp_model_output_dir, outputs_dir, final_output_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    #######################################
    # Reading data and creating mappings  #
    #######################################

    # Creating mapping object to store char-id mappings
    mapping = Mapping.Mapping()

    # Reading Monolingual Training data
    mono_train_data = dict()
    for lang in mono_langs:
        mono_train_data[lang] = MonoDataReader.MonoDataReader(lang,data_dir+'/mono_train/'+lang,mapping,max_sequence_length)

    # Reading Parallel Training data
    parallel_train_data = dict()
    for lang_pair in parallel_train_langs:
        file_prefix = data_dir+'/parallel_train/'+lang_pair[0]+'-'+lang_pair[1]+'.'
        parallel_train_data[lang_pair] = ParallelDataReader.ParallelDataReader(lang_pair[0],lang_pair[1],
            file_prefix+lang_pair[0],file_prefix+lang_pair[1],mapping,max_sequence_length)

    ## complete vocabulary creation         
    mapping.finalize_vocab()    

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
            file_name,mapping,max_sequence_length)

    ###################################################################
    #    Interacting with model and creating computation graph        #
    ###################################################################

    # Creating Model object
    model = Model.Model(mapping,representation,embedding_size,max_sequence_length) # Pass parameters

    # Creating placeholder for sequences, masks and lengths
    batch_sequences = tf.placeholder(shape=[None,max_sequence_length],dtype=tf.int32)
    batch_sequence_masks = tf.placeholder(shape=[None,max_sequence_length],dtype=tf.float32)
    batch_sequence_lengths = tf.placeholder(shape=[None],dtype=tf.float32)

    batch_sequences_2 = tf.placeholder(shape=[None,max_sequence_length],dtype=tf.int32)
    batch_sequence_masks_2 = tf.placeholder(shape=[None,max_sequence_length],dtype=tf.float32)
    batch_sequence_lengths_2 = tf.placeholder(shape=[None],dtype=tf.float32)

    # Optimizers for training using monolingual data
    # Has only one optimizer which minimizes loss of sequence reconstruction
    mono_optimizer = dict()
    for lang in mono_langs:
        mono_optimizer[lang] = model.get_mono_optimizer(learning_rate,lang,batch_sequences,batch_sequence_masks,batch_sequence_lengths)

    # Optimizers for training using parallel data
    # For each language pair, there are 3 optimizers:
    # 1. Minimize loss for transliterating first language to second
    # 2. Minimize loss for transliterating second language to first
    # 3. Minimize difference between the hidden representations

    parallel_optimizer = dict()
    for lang1,lang2 in parallel_train_langs:
        parallel_optimizer[(lang1,lang2)] = [
            model.get_parallel_optimizer(learning_rate,
                lang1,batch_sequences,batch_sequence_masks,batch_sequence_lengths,
                lang2,batch_sequences_2,batch_sequence_masks_2,batch_sequence_lengths_2),
            model.get_parallel_optimizer(learning_rate,
                lang2,batch_sequences_2,batch_sequence_masks_2,batch_sequence_lengths_2,
                lang1,batch_sequences,batch_sequence_masks,batch_sequence_lengths),
            model.get_parallel_difference_optimizer(learning_rate,
                lang1,batch_sequences,batch_sequence_lengths,
                lang2,batch_sequences_2,batch_sequence_lengths_2)]

    # Finding validation sequence loss
    # For each pair of language, return sum of loss of transliteration one script to another and vice versa
    validation_seq_loss = dict()
    for lang_pair in parallel_valid_langs:
        lang1,lang2=lang_pair
        validation_seq_loss[lang_pair] = model.seq_loss_2(
                lang1,batch_sequences,batch_sequence_masks,batch_sequence_lengths,
                lang2,batch_sequences_2,batch_sequence_masks_2,batch_sequence_lengths_2) \
            + model.seq_loss_2(
                lang2,batch_sequences_2,batch_sequence_masks_2,batch_sequence_lengths_2,
                lang1,batch_sequences,batch_sequence_masks,batch_sequence_lengths)

    # Predict output for test sequences
    infer_output = dict()
    for lang_pair in test_langs:
        infer_output[lang_pair] = model.transliterate(lang_pair[0],batch_sequences,batch_sequence_lengths,lang_pair[1])

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
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    if(start_from is not None):
        saver.restore(sess,start_from)

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
                sess.run(mono_optimizer[lang], feed_dict = {batch_sequences:sequences,batch_sequence_masks:sequence_masks,batch_sequence_lengths:sequence_lengths})
                fractional_epochs[idx] += float(len(sequences))/mono_train_data[opti_lang].num_words
            else:                           # If it is a bilingual dataset, call corresponding optimizers
                lang1 = opti_lang[0]
                lang2 = opti_lang[1]
                sequences,sequence_masks,sequence_lengths,sequences_2,sequence_masks_2,sequence_lengths_2 = parallel_train_data[opti_lang].get_next_batch(batch_size)
                sess.run(parallel_optimizer[opti_lang], feed_dict = {
                    batch_sequences:sequences,batch_sequence_masks:sequence_masks,batch_sequence_lengths:sequence_lengths,
                    batch_sequences_2:sequences_2,batch_sequence_masks_2:sequence_masks_2,batch_sequence_lengths_2:sequence_lengths_2
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
                    batch_sequences_2:sequences_2,batch_sequence_masks_2:sequence_masks_2,batch_sequence_lengths_2:sequence_lengths_2
                    })
            validation_losses.append(validation_loss)

            print "Epochs Completed : "+str(completed_epochs).zfill(3)+"\t Validation loss: "+str(validation_loss)

            # If validation loss is increasing since last 3 epochs, take the last 4th model and stop training process
            if(completed_epochs>=4 and all([i>j for (i,j) in zip(validation_losses[-3:],validation_losses[-4:-1])])):
                completed_epochs -= 3
                saver.restore(sess,temp_model_output_dir+'my_model-'+str(completed_epochs))
                cont = False

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
                    predicted_sequences_ids = sess.run(infer_output[lang_pair], feed_dict={batch_sequences: sequences, batch_sequence_lengths: sequence_lengths})
                    predicted_sequences = mapping.get_words_from_id_lists(predicted_sequences_ids,target_lang)
                    if completed_epochs % infer_every == 0:
                        codecs.open(out_folder+str(completed_epochs).zfill(3)+source_lang+'-'+target_lang+'_','w','utf-8').write(u'\n'.join(predicted_sequences))

            # Save current model
            if(cont == True):
                if(completed_epochs==1 or validation_losses[-1]<validation_losses[-2]):
                    saver.save(sess, temp_model_output_dir+'my_model', global_step=completed_epochs)

    # save final model
    final_saver.save(sess,output_dir+'/final_model_epochs_'+str(completed_epochs))
