import argparse
import os
import sys
import codecs

import itertools as it
import tensorflow as tf
import numpy as np

import Model
import AttentionModel
import Mapping
import MonoDataReader
import ParallelDataReader
import utilities

import calendar,time
from indicnlp import loader

def decode(out_dir,epoch_no,lang_pairs,test_data,
        decode_output_op,decode_scores_op,
        batch_sequences, batch_sequence_masks, batch_sequence_lengths,
        beam_size, topn,
        prefix_tgtlang,prefix_srclang,mapping,
        max_sequence_length, beam_size_val,topn_val): 
    """
    convenience function to decode data 
    
    out_dir: output directory to store decoded files for each language
    epoch_no: epoch number for which decoding is being done 
    lang_pairs: list of language pair tuples. 
    test_data: Dictionary of MonoDataReader object for various language pairs. 
                Holds only the source language data 

    decode_output_op: dictionary of decoder output operation for every language pair 
    decode_scores_op: dictionary of decoder output scores operation for every language pair 

    batch_sequences, batch_sequence_masks, batch_sequence_lengths,
        beam_size, topn: placeholder variables 

    prefix_srclang: see commandline flags 
    prefix_tgtlang: see commandline flags 
    mapping: Dictionary of mapping objects for each language
    max_sequence_length: 
    beam_size_val: 
    topn_val: 
    """
    
    test_loss=0.0
    for src_lang,tgt_lang in lang_pairs:
        lang_pair=(src_lang,tgt_lang)

        sequences,sequence_masks, sequence_lengths = test_data[lang_pair].get_data()
    
        if prefix_tgtlang: 
            sequences,sequence_masks,sequence_lengths = Mapping.prefix_sequence_with_token(
                                        sequences,sequence_masks,sequence_lengths,
                                        tgt_lang,mapping[tgt_lang])
    
        if prefix_srclang: 
            sequences,sequence_masks,sequence_lengths = Mapping.prefix_sequence_with_token(
                                        sequences,sequence_masks,sequence_lengths,
                                        src_lang,mapping[src_lang])
    
        predicted_sequences_ids_list=[]
        predicted_scores_list=[]
        for start in xrange(0,sequences.shape[0],batch_size):
            end = min(start+batch_size,sequences.shape[0])
    
            batch_start_time=time.time()
    
            data_sequences=sequences[start:end,:]
            data_sequence_masks=sequence_masks[start:end,:]
            data_sequence_lengths=sequence_lengths[start:end]
    
            b_sequences_ids, b_scores = sess.run([decode_output_op[lang_pair],decode_scores_op[lang_pair]], 
                feed_dict={batch_sequences: data_sequences, batch_sequence_lengths: data_sequence_lengths, 
                            beam_size: beam_size_val, topn: topn_val
                           }
                )
    
            predicted_sequences_ids_list.append(b_sequences_ids)
            predicted_scores_list.append(b_scores)
    
        predicted_sequences_ids=np.concatenate(predicted_sequences_ids_list,axis=0)
        predicted_scores=np.concatenate(predicted_scores_list,axis=0)
    
        ## write output to file 

        out_fname=out_dir+str(epoch_no).zfill(3)+'test.nbest.'+src_lang+'-'+tgt_lang+'.'+tgt_lang
        with codecs.open(out_fname,'w','utf-8') as outfile: 
            for sent_no, all_sent_predictions in enumerate(predicted_sequences_ids): 
                for rank, predicted_sequence_ids in enumerate(all_sent_predictions): 
                    sent=[mapping[tgt_lang].get_char(x,tgt_lang) for x in predicted_sequence_ids]
                    sent=u' '.join(it.takewhile(lambda x:x != u'EOW',it.dropwhile(lambda x:x==u'GO',sent))) 
                    outfile.write(u'{} ||| {} ||| Distortion0= -1 LM0= -1 WordPenalty0= -1 PhrasePenalty0= -1 TranslationModel0= -1 -1 -1 -1 ||| {}\n'.format(sent_no,sent,predicted_scores[sent_no,rank]))
    
        ### compute loss: just the negative of the likelihood of best candidates
        best_scores=predicted_scores[:,0]
        test_loss+= (-np.sum(best_scores))

    return test_loss 
    

def get_seq2seq_loss(lang_pairs, parallel_data,
                        seq_loss_op,
                        batch_sequences, batch_sequence_masks, batch_sequence_lengths,
                        batch_sequences_2, batch_sequence_masks_2, batch_sequence_lengths_2,
                        dropout_keep_prob,
                        prefix_tgtlang, prefix_srclang, 
                        mapping, max_sequence_length):
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
    for src_lang,tgt_lang in lang_pairs:
        
        lang_pair=(src_lang,tgt_lang)
    
        sequences,sequence_masks,sequence_lengths,sequences_2,sequence_masks_2,sequence_lengths_2 = \
                    parallel_data[lang_pair].get_data()
    
        if prefix_tgtlang: 
            sequences,sequence_masks,sequence_lengths = Mapping.prefix_sequence_with_token(
                                        sequences,sequence_masks,sequence_lengths,
                                        tgt_lang,mapping[tgt_lang])
        if prefix_srclang: 
            sequences,sequence_masks,sequence_lengths = Mapping.prefix_sequence_with_token(
                                        sequences,sequence_masks,sequence_lengths,
                                        src_lang,mapping[src_lang])
    
        validation_loss += sess.run(seq_loss_op[lang_pair], feed_dict = {
            batch_sequences:sequences,batch_sequence_masks:sequence_masks,batch_sequence_lengths:sequence_lengths,
            batch_sequences_2:sequences_2,batch_sequence_masks_2:sequence_masks_2,batch_sequence_lengths_2:sequence_lengths_2,
            dropout_keep_prob:1.0
            })

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
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_dir', type = str, help = 'data directory')
    parser.add_argument('--output_dir', type = str, help = 'output folder name')
    parser.add_argument('--train_size', type = int, metavar = 'N', default = -1, help = 'use the first N sequence pairs for training. N=-1 means use the entire training set.')

    parser.add_argument('--lang_pairs', type = str, default = None, help = 'List of language pairs for supervised training given as: "lang1-lang2,lang3-lang4,..."')
    parser.add_argument('--unseen_langs', type = str, default = None, help = 'List of languages not seen during training given as: "lang1,lang2,lang3,lang4,..."')

    parser.add_argument('--enc_type', type = str, default = 'cnn',  help = 'encoder to use. One of (1) simple_lstm_noattn (2) bilstm (3) cnn')
    parser.add_argument('--separate_output_embedding', action='store_true', default = False,  help = 'Should separate embeddings be used on the input and output side. Generally the same embeddings are to be used. This is used only for Indic-Indic transliteration, when input is phonetic and output is onehot_shared')
    parser.add_argument('--prefix_tgtlang', action='store_true', default = False,
            help = 'Prefix the input sequence with the language code for the target language')
    parser.add_argument('--prefix_srclang', action='store_true', default = False,
            help = 'Prefix the input sequence with the language code for the source language')

    parser.add_argument('--embedding_size', type = int, default = 256, help = 'size of character representation')
    parser.add_argument('--enc_rnn_size', type = int, default = 512, help = 'size of output of encoder RNN')
    parser.add_argument('--dec_rnn_size', type = int, default = 512, help = 'size of output of dec RNN')

    parser.add_argument('--max_seq_length', type = int, default = 30, help = 'maximum sequence length')
    parser.add_argument('--batch_size', type = int, default = 32, help = 'size of each batch used in training')
    parser.add_argument('--max_epochs', type = int, default = 30, help = 'maximum number of epochs')
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'learning rate of Adam Optimizer')
    parser.add_argument('--dropout_keep_prob', type = float, default = 0.5, help = 'keep probablity for the dropout layers')
    parser.add_argument('--infer_every', type = int, default = 1, help = 'write predicted outputs for test data after these many epochs, 0 if not required')

    parser.add_argument('--topn', type = int, default = 10, help = 'the number of best candidates to output by the decoder')
    parser.add_argument('--beam_size', type = int, default = 5, help = 'beam size for decoding')

    parser.add_argument('--start_from', type = int, default = None, help = 'epoch to restore model from. This must be one of the epochs for which model has been saved')

    parser.add_argument('--representation', type = str, default = 'onehot',  help = 'input representation, which can be specified in two ways: (i) one of "phonetic", "onehot", "onehot_and_phonetic" - this representation is used for all languages, (ii) "lang1:<rep1>,lang2:<rep2>" - this representation is used for all languages, rep can be one of "phonetic", "onehot", "onehot_and_phonetic" ') 
    parser.add_argument('--shared_mapping_class', type = str, default = 'IndicPhoneticMapping',  help = 'class to be used for shared mapping. Possible values: IndicPhoneticMapping, CharacterMapping')

    args = parser.parse_args()

    print '========== Parameters start ==========='
    for k,v in vars(args).iteritems():
        print '{}: {}'.format(k,v)
    print '========== Parameters end ============='

    ## check for required parameters
    if args.lang_pairs is None:
        print 'ERROR: --lang_pairs has to be set'
        sys.exit(1)

    if args.data_dir is None:
        print 'ERROR: --data_dir has to be set'
        sys.exit(1)

    if args.output_dir is None:
        print 'ERROR: --output_dir has to be set'
        sys.exit(1)

    #### Reading arguments

    ## directories 
    data_dir = args.data_dir
    output_dir = args.output_dir
    train_size = args.train_size 

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

    ## additional hyperparameters 
    max_sequence_length = args.max_seq_length
    batch_size = args.batch_size
    max_epochs = args.max_epochs
    learning_rate = args.learning_rate
    infer_every = args.infer_every
    dropout_keep_prob_val = args.dropout_keep_prob

    ## decoding
    beam_size_val= args.beam_size
    topn_val = args.topn

    # Setting the language parameters
    parallel_train_langs=None
    parallel_valid_langs=None
    test_langs=None
    unseen_langs=None
    all_langs=None
    

    parallel_train_langs=[ tuple(lp.split('-')) for lp in args.lang_pairs.split(',')]
    parallel_valid_langs=parallel_train_langs
    unseen_langs= [] if args.unseen_langs is None else (  [ l for l in args.unseen_langs.split(',')] )

    mll=set()
    for lp in [list(x) for x in parallel_train_langs]: 
        mll.update(lp)

    all_langs=list(mll)

    ### support for zeroshot transliteration
    all_langs.extend(unseen_langs)

    test_langs=parallel_train_langs 

    print 'Parallel Train, Parallel Valid, Test Langs, All Langs, Unseen Langs'
    print parallel_train_langs
    print parallel_valid_langs
    print test_langs
    print all_langs 
    print unseen_langs

    ## restart from this iteration number
    start_from = args.start_from

    # Create output folders if required
    temp_model_output_dir = output_dir+'/temp_models/'
    outputs_dir = output_dir+'/outputs/'
    valid_outputs_dir = output_dir+'/validation/'
    mappings_dir = output_dir+'/mappings/'
    final_output_dir = output_dir+'/final_output/'
    log_dir = output_dir+'/log/'

    for folder in [temp_model_output_dir, outputs_dir, valid_outputs_dir, mappings_dir, final_output_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    #######################################
    # Reading data and creating mappings  #
    #######################################

    ### parse representation argument 
    if args.representation in ['onehot','onehot_shared','phonetic','onehot_and_phonetic']: 
        representation = {} 
        for lang in all_langs: 
            representation[lang]=args.representation 
    else: 
        representation = dict([ x.split(':') for x in args.representation.split(',') ])

    ## Print Representation and Mappings 
    print 'Represenation'
    print representation 

    # Creating mapping object to store char-id mappings
    mapping={}
    shared_mapping_obj = Mapping.get_mapping_instance(shared_mapping_class) 

    for lang in all_langs: 
        if representation[lang] in ['phonetic','onehot_and_phonetic']: 
            mapping[lang]=shared_mapping_obj
        elif representation[lang]=='onehot_shared': 
            mapping[lang]=shared_mapping_obj
        elif representation[lang]=='onehot': 
            mapping[lang]=Mapping.CharacterMapping()

    ### support for zeroshot transliteration
    for lang in unseen_langs: 
        mapping[lang].lang_list.add(lang)

    ## Print Representation and Mappings 
    print 'Mapping'
    print mapping

    print 'Start Reading Data'

    ## add language code to vocabulary
    for lang in all_langs: 
        if prefix_srclang or prefix_tgtlang or os.getenv('ALWAYS_LANG_TOKEN') is not None: 
            mapping[lang].get_index(Mapping.create_special_token(lang),lang)

    # Reading Parallel Training data
    parallel_train_data = dict()
    for lang_pair in parallel_train_langs:
        file_prefix = data_dir+'/parallel_train/'+lang_pair[0]+'-'+lang_pair[1]+'.'
        parallel_train_data[lang_pair] = ParallelDataReader.ParallelDataReader(lang_pair[0],lang_pair[1],
            file_prefix+lang_pair[0],file_prefix+lang_pair[1],mapping,max_sequence_length,train_size)

    ## complete vocabulary creation
    for lang in all_langs: 
        mapping[lang].finalize_vocab()

    ## save the mapping
    ### NOTE: If mapping objects are shared across languages, this cannot be restored while loading the mapping
    #         But this is ok for decoding, the sharing is a concern only during determination of vocabulary 
    #         during training. Hence, this simpler `save` strategy has been implemented
    for lang in all_langs: 
        with open(mappings_dir+'/mapping_{}.json'.format(lang),'w') as mapping_json_file:
            mapping[lang].save_mapping(mapping_json_file)

    print 'Vocabulary Statitics'
    for lang in all_langs: 
        print '{}: {}'.format(lang,mapping[lang].get_vocab_size())
    sys.stdout.flush()

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
            file_name,mapping[lang_pair[0]],max_sequence_length)

    # Reading validation data for decoding 
    valid_decode_data = dict()
    for lang_pair in parallel_valid_langs:
        file_name = data_dir+'/parallel_valid/'+lang_pair[0]+'-'+lang_pair[1]+'.'+lang_pair[0]
        valid_decode_data[lang_pair] = MonoDataReader.MonoDataReader(lang_pair[0],
            file_name,mapping[lang_pair[0]],max_sequence_length)

    print 'Finished Reading Data' 

    ###################################################################
    #    Interacting with model and creating computation graph        #
    ###################################################################

    # Creating Model object
    model = AttentionModel.AttentionModel(mapping,representation,max_sequence_length,
            embedding_size,enc_rnn_size,dec_rnn_size, 
            enc_type,separate_output_embedding) # Pass parameters

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
    
    # Optimizers for training using parallel data

    sup_optimizer = dict()
    for lang_pair in parallel_train_langs:
        lang1,lang2=lang_pair
        print 'Created optimizer for language pair: {}-{}'.format(lang1,lang2)
        sup_optimizer[(lang1,lang2)] = model.get_parallel_optimizer(
                learning_rate,
                lang1,batch_sequences,batch_sequence_masks,batch_sequence_lengths,
                lang2,batch_sequences_2,batch_sequence_masks_2,batch_sequence_lengths_2,dropout_keep_prob)

    # Finding validation sequence loss
    # For each pair of language, return sum of loss of transliteration one script to another and vice versa
    validation_seq_loss = dict()

    for lang_pair in parallel_valid_langs:
        lang1,lang2=lang_pair
        print 'Created validation loss calculator for language pair: {}-{}'.format(lang1,lang2)
        validation_seq_loss[lang_pair] = model.seq_loss_2(
                lang1,batch_sequences,batch_sequence_masks,batch_sequence_lengths,
                lang2,batch_sequences_2,batch_sequence_masks_2,batch_sequence_lengths_2,dropout_keep_prob)

    # Predict output for test sequences
    infer_output = dict()
    infer_output_scores = dict()
    for lang_pair in test_langs:
        lang1,lang2=lang_pair
        print 'Created decoder for language pair: {}-{}'.format(lang1,lang2)
        infer_output[lang_pair], infer_output_scores[lang_pair] =  \
            model.transliterate_beam(lang_pair[0],batch_sequences,batch_sequence_lengths,lang_pair[1],beam_size,topn)

    # All training dataset
    training_langs = parallel_train_langs

    # Fractional epoch: stores what fraction of each dataset is used till now after last completed epoch.
    fractional_epochs = [0.0 for _ in training_langs]
    completed_epochs = 0

    #Saving model
    saver = tf.train.Saver(max_to_keep = 0)
    final_saver = tf.train.Saver()

    print "Done with creating graph. Starting session"
    sys.stdout.flush()

    #Start Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    if(start_from is not None):
        saver.restore(sess,'{}/temp_models/my_model-{}'.format(output_dir,start_from))
        completed_epochs=start_from
    
    tf.summary.FileWriter(log_dir,sess.graph)

    print "Session started"

    steps = 0
    validation_losses = []

    epoch_train_time=0.0
    epoch_train_loss=0.0

    start_time=time.time()

    # Whether to continue or now
    cont = True

    print 'Starting training ...'
    while cont:
        # Selected the dataset whose least fraction is used for training in current epoch
        for (opti_lang,idx) in zip(training_langs,range(len(training_langs))):

            ### TRAIN
            update_start_time=time.time()

            # If it is a bilingual dataset, call corresponding optimizers
            lang1 = opti_lang[0]
            lang2 = opti_lang[1]

            sequences,sequence_masks,sequence_lengths,sequences_2,sequence_masks_2,sequence_lengths_2 = \
                    parallel_train_data[opti_lang].get_next_batch(batch_size)
            
            if prefix_tgtlang: 
                sequences,sequence_masks,sequence_lengths = Mapping.prefix_sequence_with_token(sequences,sequence_masks,sequence_lengths, 
                                                            lang2,mapping[lang2])
            if prefix_srclang: 
                sequences,sequence_masks,sequence_lengths = Mapping.prefix_sequence_with_token(sequences,sequence_masks,sequence_lengths, 
                                                            lang1,mapping[lang1])

            _, step_loss = sess.run(sup_optimizer[opti_lang], feed_dict = {
                batch_sequences:sequences,batch_sequence_masks:sequence_masks,batch_sequence_lengths:sequence_lengths,
                batch_sequences_2:sequences_2,batch_sequence_masks_2:sequence_masks_2,batch_sequence_lengths_2:sequence_lengths_2,
                dropout_keep_prob:dropout_keep_prob_val
                })

            fractional_epochs[idx] += float(len(sequences))/parallel_train_data[opti_lang].num_words

            epoch_train_loss+=step_loss

            update_end_time=time.time()
            epoch_train_time+=(update_end_time-update_start_time)

        # One more batch is processed
        steps+=1
        # If all datasets are used for training epoch is complete
        if(min(fractional_epochs) >= 1.0):

            # update epoch number 
            completed_epochs += 1

            print "Epochs Completed : "+str(completed_epochs).zfill(3)+"\t Training loss: "+str(epoch_train_loss)

            ### VALIDATION LOSS 
            # Find validation loss
            valid_start_time=time.time()

            validation_loss=get_seq2seq_loss(parallel_valid_langs, parallel_valid_data,
                                validation_seq_loss,
                                batch_sequences, batch_sequence_masks, batch_sequence_lengths,
                                batch_sequences_2, batch_sequence_masks_2, batch_sequence_lengths_2,
                                dropout_keep_prob,
                                prefix_tgtlang, prefix_srclang, 
                                mapping, max_sequence_length)
            validation_losses.append(validation_loss)

            valid_end_time=time.time()
            epoch_valid_time=(valid_end_time-valid_start_time)

            print "Epochs Completed : "+str(completed_epochs).zfill(3)+"\t Validation loss: "+str(validation_loss)

            #### Comment it to avoid early stopping
            ## If validation loss is increasing since last 3 epochs, take the last 4th model and stop training process
            #if(completed_epochs>=4 and len(validation_losses)>=4 and all([i>j for (i,j) in zip(validation_losses[-3:],validation_losses[-4:-1])])):
            #    completed_epochs -= 3
            #    saver.restore(sess,temp_model_output_dir+'my_model-'+str(completed_epochs))
            #    cont = False

            # If max_epochs are done
            if(completed_epochs >= max_epochs):
                cont = False

            ### Decode the test set

            test_start_time=0.0
            test_end_time=0.0
            validdecode_start_time=0.0
            validdecode_end_time=0.0

            if(cont == False or (infer_every > 0 and completed_epochs%infer_every == 0)):

                # If this was the last epoch, output result to final output folder, otherwise to outputs folder

                if completed_epochs % infer_every == 0:

                    test_start_time=time.time()
                    test_loss=decode(outputs_dir, completed_epochs, test_langs, test_data,
                            infer_output, infer_output_scores,
                            batch_sequences, batch_sequence_masks, batch_sequence_lengths,
                            beam_size, topn,
                            prefix_tgtlang, prefix_srclang, mapping,
                            max_sequence_length, beam_size_val, topn_val) 
                    test_end_time=time.time()

                    validdecode_start_time=time.time()
                    valid_decode_loss=decode(valid_outputs_dir, completed_epochs, parallel_valid_langs, valid_decode_data,
                            infer_output, infer_output_scores,
                            batch_sequences, batch_sequence_masks, batch_sequence_lengths,
                            beam_size, topn,
                            prefix_tgtlang, prefix_srclang, mapping,
                            max_sequence_length, beam_size_val, topn_val) 
                    validdecode_end_time=time.time()

                    print "Epochs Completed : "+str(completed_epochs).zfill(3)+"\t Test loss: "+str(test_loss)

            epoch_test_time=(test_end_time-test_start_time)
            epoch_validdecode_time=(validdecode_end_time-validdecode_start_time)

            # Save current model
            #if(cont == True):
                #if(completed_epochs==1 or (len(validation_losses)>=2 and validation_losses[-1]<validation_losses[-2])):
            saver.save(sess, temp_model_output_dir+'my_model', global_step=completed_epochs)

            print "Epochs Completed : "+str(completed_epochs).zfill(3)+ \
                    "\t Time (hh:mm:ss)::: train> {} valid> {} test> {} validdecode> {}".format(
                            utilities.formatted_timeinterval(epoch_train_time), 
                            utilities.formatted_timeinterval(epoch_valid_time), 
                            utilities.formatted_timeinterval(epoch_test_time),
                            utilities.formatted_timeinterval(epoch_validdecode_time),
                            )

            ## update epoch variables 
            fractional_epochs = [0.0 for _ in parallel_train_langs]
            epoch_train_time=0.0
            epoch_train_loss=0.0

            print "Epochs Completed : "+str(completed_epochs).zfill(3)+ \
                    "\t Number of training steps: {}".format(steps)

            print "Epochs Completed : "+str(completed_epochs).zfill(3)+ \
                    "\t Time of completion: {}".format(time.asctime())

            sys.stdout.flush()

    # save final model
    final_saver.save(sess,output_dir+'/final_model_epochs_'+str(completed_epochs))

    print 'End training' 

    print 'Final number of training steps: {}'.format(steps)

    #### Print total training time
    end_time=time.time()
    print 'Total Time for Training (hh:mm:ss) : {}'.format(utilities.formatted_timeinterval(end_time-start_time))

    print 'Process terminated at: ' + time.asctime()

