import argparse
import os
import sys
import codecs
import calendar,time
import itertools as it

import tensorflow as tf
import numpy as np

from indicnlp import loader

import Mapping
import encoders
import MonoDataReader
import utilities

class LanguageModel(object):
    """
    Neural Language Model
    """

    def __init__(self,lang, mapping, representation,
                    max_sequence_length, embedding_size,rnn_size):

        ## save parameters 
        self.lang = lang
        self.mapping = mapping
        self.representation=representation
        
        self.max_sequence_length = max_sequence_length
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size

        self.vocab_size = self.mapping.get_vocab_size()

        max_val = 0.1

        ####### Input embeddings 
        self.bitvector_embeddings = tf.constant(
                                        self.mapping.get_bitvector_embeddings(self.lang,self.representation),
                                        dtype=tf.float32)
        self.bitvector_embedding_size = self.mapping.get_bitvector_embedding_size(self.representation)

        ###### Embeddings 

        self.Wmat = tf.Variable(tf.random_uniform([self.bitvector_embedding_size, self.embedding_size], 
                                                     -1*max_val, max_val), 
                                name = 'lm_Wmat')
        self.embed_W = tf.matmul(self.bitvector_embeddings, self.Wmat, name='lm_embed_W')
        self.embed_b = tf.Variable(tf.constant(0., shape=[self.embedding_size]), name = 'lm_embed_b')

        ##### Encoder Cell
        ## use variable scope='lang_model'
        self.encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size,state_is_tuple=False)

        ### Output layer
        # Output decoder to vocab_size vector
        self.out_W = tf.Variable(tf.random_uniform([self.rnn_size,self.vocab_size], -1*max_val, max_val), 
                                    name='lm_out_W')
        self.out_b = tf.Variable(tf.constant(0., shape = [self.vocab_size]), name='lm_out_b')

    def compute_logits(self, sequences, seq_lengths, dropout_keep_prob): 
        """
        Compute logits for the sequences 

        sequences: Dimension - [ batchsize, max_sequence_length ]
        seq_lengths:  dimension - batchsize 
        dropout_keep_prob: scalar. Dropout applied to the output of RNN layer

        Returns logits for each element in all the sequences
            dimesion: [ max_sequence_length x batchsize, vocab_size ]
        """

        batch_size = tf.shape(sequences)[0]

        ### embedding lookup 
        sequence_embeddings = tf.add(tf.nn.embedding_lookup(self.embed_W,sequences),self.embed_b)

        ### reshape 
        x = tf.transpose(sequence_embeddings,[1,0,2])
        x = tf.reshape(x,[-1,self.embedding_size])
        x = tf.split(axis=0,num_or_size_splits=self.max_sequence_length,value=x)

        ## run through RNN layer 
        cell=tf.nn.rnn_cell.DropoutWrapper(self.encoder_cell,output_keep_prob=dropout_keep_prob)
        enc_outputs, states = tf.nn.rnn(self.encoder_cell, x, dtype = tf.float32, sequence_length = seq_lengths, 
                scope = 'lang_model')
        output = tf.reshape(tf.concat(axis=0, values=enc_outputs), [-1, self.rnn_size])
        logits = tf.matmul(output, self.out_W) + self.out_b

        return logits 

    def average_loss(self, sequences, seq_lengths, dropout_keep_prob): 
        """
        computes average perplexity per input element over entire dataset

        sequences: Dimension - [ batchsize, max_sequence_length ]
        seq_lengths:  dimension - batchsize 
        dropout_keep_prob: scalar. Dropout applied to the output of RNN layer
        """

        batch_size = tf.shape(sequences)[0]
        logits = self.compute_logits(sequences, seq_lengths, dropout_keep_prob)

        ## compute the loss
        placeholder=tf.ones([batch_size,1],dtype=tf.int32)*self.mapping.get_index(Mapping.Mapping.PAD)
        label_sequences=tf.concat(axis=1,values=[sequences[:,1:],placeholder])

        loss = tf.nn.seq2seq.sequence_loss(
            [logits],
            [tf.reshape(label_sequences, [-1])],
            [tf.ones([batch_size * self.max_sequence_length])]
            )

        return loss

    def per_seq_loss(self,sequences,seq_lengths): 
        """
        computes average perplexity per input element over entire dataset

        sequences: Dimension - [ batchsize, max_sequence_length ]
        seq_lengths:  dimension - batch_size 

        returns average perplexity  sequence (averaged over each element)
            dimension: batch_size

        """

        batch_size = tf.shape(sequences)[0]
        l=self.max_sequence_length

        logits = self.compute_logits(sequences, seq_lengths, tf.constant(1.0))

        ## compute the loss
        placeholder=tf.ones([batch_size,1],dtype=tf.int32)*self.mapping.get_index(Mapping.Mapping.PAD)
        ## shift gold standard by one timestep 
        label_sequences=tf.concat(axis=1, values=[sequences[:,1:],placeholder])

        label_sequences_flat = tf.reshape(
                                            tf.transpose(label_sequences,[1,0]),
                                            [-1]
                                         )

        t2l=lambda t: tf.split(axis=0,num_or_size_splits=l,value=t)

        loss = tf.nn.seq2seq.sequence_loss_by_example(
                                t2l(logits),
                                t2l(label_sequences_flat),
                                t2l(tf.ones([l* batch_size]))
                            )

        return loss

    def get_optimizer(self,sequences,sequence_lengths,
                    learning_rate,dropout_keep_prob):
        """
        computes average perplexity per input element over entire dataset

        sequences: Dimension - [batchsize, max_sequence_length]
        seq_lengths:  dimension - batchsize 
        learning_rate: scalar
        dropout_keep_prob: scalar. Dropout applied to the output of RNN layer
        """
        loss=self.average_loss(sequences, sequence_lengths, dropout_keep_prob)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return [ optimizer, loss ]

    def logit_next_char(self,current_input,prev_state): 
        """
        Given the current input and the prev_state, compute the logit output 
        and next state

        current_input: [ batch_size , 1 ]
        prev_state: [ batch_size , state_size ]

        returns (logit,next_state) dimension: ( [batch_size, vocab_size], [batch_size,state_size]  )
            
        """

        batch_size = tf.shape(current_input)[0]
        current_embeddings = tf.squeeze(
                     tf.add(tf.nn.embedding_lookup(self.embed_W,current_input),self.embed_b),
                     [1]
                    )

        output, state = self.encoder_cell(current_embeddings,prev_state)
        logit_words = tf.add(tf.matmul(output,self.out_W),self.out_b)

        return (logit_words, state)

    def initial_state(self,batch_size):

        return self.encoder_cell.zero_state(batch_size,tf.float32)

    def state_size(self):

        return self.encoder_cell.state_size

###################### CODE FOR THE COMMANDLINE OPERATIONS ################

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

def run_train(args): 
    """
     Training script
    """

    ## check for required parameters
    if args.lang is None:
        print 'ERROR: --lang has to be set'
        sys.exit(1)

    if args.data_dir is None and args.mode == 'train':
        print 'ERROR: --data_dir has to be set'
        sys.exit(1)

    if args.data_dir is None and args.mode == 'test':
        print 'ERROR: --output_dir has to be set'
        sys.exit(1)

    # Create output folders if required
    models_dir = args.output_dir+'/models/'
    mappings_dir = args.output_dir+'/mappings/'
    log_dir = args.output_dir+'/log/'

    for folder in [models_dir, mappings_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    #######################################
    # Reading data and creating mappings  #
    #######################################

    # Creating mapping object to store char-id mappings
    mapping={}
    mapping = Mapping.get_mapping_instance(args.mapping_class) 

    if args.use_mapping is not None: 
        print 'Using existing vocabulary'
        with open(args.use_mapping,'r') as mapping_json_file:     
            mapping.load_mapping(mapping_json_file)

    print 'Start Reading Data'
    train_data = MonoDataReader.MonoDataReader(args.lang, args.data_dir+'/train.'+args.lang , mapping,args.max_seq_length)

    ## complete vocabulary creation
    if args.use_mapping is None: 
        mapping.finalize_vocab()

    with open(mappings_dir+'/mapping_{}.json'.format(args.lang),'w') as mapping_json_file:
        mapping.save_mapping(mapping_json_file)

    ## Print Representation and Mappings 
    print 'Mapping'
    print mapping

    print 'Vocabulary Statitics'
    print '{}: {}'.format(args.lang,mapping.get_vocab_size())
    sys.stdout.flush()

    # Reading Validation data
    valid_data = MonoDataReader.MonoDataReader(args.lang,args.data_dir+'/tun.'+args.lang ,mapping,args.max_seq_length)
    ## Reading test data 
    test_data  = MonoDataReader.MonoDataReader(args.lang,args.data_dir+'/test.'+args.lang ,mapping,args.max_seq_length)

    print 'Finished Reading Data' 

    ###################################################################
    #    Interacting with model and creating computation graph        #
    ###################################################################

    # Creating Model object
    model = LanguageModel(args.lang,mapping,args.representation,
                    args.max_seq_length, args.embedding_size,args.rnn_size)

    ## Creating placeholder for sequences, masks and lengths and dropout keep probability 
    pl_batch_sequences = tf.placeholder(shape=[None,args.max_seq_length],dtype=tf.int32)
    pl_batch_sequence_lengths = tf.placeholder(shape=[None],dtype=tf.float32)

    optimizer_op = model.get_optimizer(
                        pl_batch_sequences, pl_batch_sequence_lengths,
                        args.learning_rate, args.dropout_keep_prob)

    tf.get_variable_scope().reuse_variables()

    loss_op = model.average_loss(pl_batch_sequences, pl_batch_sequence_lengths, 1.0)

    print "Done with creating graph. Starting session"
    sys.stdout.flush()

    #Saving model
    saver = tf.train.Saver(max_to_keep = 0)
    final_saver = tf.train.Saver()

    #Start Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    if(args.start_from is not None):
        saver.restore(sess,'{}/models/model-{}'.format(args.output_dir,args.start_from))
        completed_epochs=args.start_from
    
    tf.summary.FileWriter(log_dir,sess.graph)

    print "Session started"

    # Fractional epoch: stores what fraction of each dataset is used till now after last completed epoch.
    fractional_epochs = 0.0
    completed_epochs = 0

    steps = 0
    prev_steps = 0
    validation_losses = []

    epoch_train_time=0.0
    epoch_train_loss=0.0

    start_time=time.time()

    # Whether to continue or now
    cont = True

    print 'Starting training ...'
    while cont:
        # Selected the dataset whose least fraction is used for training in current epoch

        ### TRAIN
        update_start_time=time.time()

        sequences,sequence_masks,sequence_lengths, = \
                train_data.get_next_batch(args.batch_size)
        
        _, step_loss = sess.run(optimizer_op, feed_dict = {
           pl_batch_sequences:sequences,pl_batch_sequence_lengths:sequence_lengths})

        fractional_epochs += float(len(sequences))/train_data.num_words

        epoch_train_loss+=step_loss

        update_end_time=time.time()
        epoch_train_time+=(update_end_time-update_start_time)

        # One more batch is processed
        steps+=1
        # If all datasets are used for training epoch is complete
        if(fractional_epochs >= 1.0):

            # update epoch number 
            completed_epochs += 1


            ### VALIDATION LOSS 
            # Find validation loss
            valid_start_time=time.time()
            validation_loss=get_average_loss(valid_data, loss_op,
                                pl_batch_sequences, pl_batch_sequence_lengths, sess)
            validation_losses.append(validation_loss)
            valid_end_time=time.time()
            epoch_valid_time=(valid_end_time-valid_start_time)

            ## TEST LOSS
            test_start_time=time.time()
            test_loss=get_average_loss(test_data, loss_op,
                                pl_batch_sequences, pl_batch_sequence_lengths, sess)
            test_end_time=time.time()
            epoch_test_time=(test_end_time-test_start_time)

            print "Epochs Completed : "+str(completed_epochs).zfill(3)+"\t Training loss: "+str(epoch_train_loss/(steps-prev_steps))
            print "Epochs Completed : "+str(completed_epochs).zfill(3)+"\t Validation loss: "+str(validation_loss)
            print "Epochs Completed : "+str(completed_epochs).zfill(3)+"\t Test loss: "+str(test_loss)

    #        #### Comment it to avoid early stopping
    #        ## If validation loss is increasing since last 3 epochs, take the last 4th model and stop training process
    #        #if(completed_epochs>=4 and len(validation_losses)>=4 and all([i>j for (i,j) in zip(validation_losses[-3:],validation_losses[-4:-1])])):
    #        #    completed_epochs -= 3
    #        #    saver.restore(sess,models_dir+'my_model-'+str(completed_epochs))
    #        #    cont = False

            # If max_epochs are done
            if(completed_epochs >= args.max_epochs):
                cont = False

            saver.save(sess, models_dir+'model', global_step=completed_epochs)

            print "Epochs Completed : "+str(completed_epochs).zfill(3)+ \
                    "\t Time (hh:mm:ss)::: train> {} valid> {} test> {}".format(
                            utilities.formatted_timeinterval(epoch_train_time), 
                            utilities.formatted_timeinterval(epoch_valid_time), 
                            utilities.formatted_timeinterval(epoch_test_time),
                            )

            ## update epoch variables 
            prev_steps = steps 
            fractional_epochs = 0.0
            epoch_train_time = 0.0
            epoch_train_loss = 0.0

            print "Epochs Completed : "+str(completed_epochs).zfill(3)+ \
                    "\t Number of training steps: {}".format(steps)

            print "Epochs Completed : "+str(completed_epochs).zfill(3)+ \
                    "\t Time of completion: {}".format(time.asctime())

            sys.stdout.flush()

    # save final model
    final_saver.save(sess,args.output_dir+'/final_model_epochs_'+str(completed_epochs))

    print 'End training' 

    print 'Final number of training steps: {}'.format(steps)

    #### Print total training time
    end_time=time.time()
    print 'Total Time for Training (hh:mm:ss) : {}'.format(utilities.formatted_timeinterval(end_time-start_time))

def run_test(args): 
    """
    Test a trained language mode
    """


    ## check for required parameters
    if args.lang is None:
        print 'ERROR: --lang has to be set'
        sys.exit(1)

    if args.model_fname is None:
        print 'ERROR: --model_fname has to be set'
        sys.exit(1)

    if args.mapping_fname is None:
        print 'ERROR: --mapping_fname has to be set'
        sys.exit(1)

    if args.in_fname is None:
        print 'ERROR: --in_fname has to be set'
        sys.exit(1)

    #######################################
    # Reading data and creating mappings  #
    #######################################

    # Creating mapping object to store char-id mappings
    mapping = Mapping.get_mapping_instance(args.mapping_class) 
    with open(args.mapping_fname,'r') as mapping_json_file:     
        mapping.load_mapping(mapping_json_file)
    print 'Mapping'
    print mapping

    print 'Vocabulary Statitics'
    print '{}: {}'.format(args.lang,mapping.get_vocab_size())

    ## Reading test data 
    test_data  = MonoDataReader.MonoDataReader(args.lang,args.in_fname,mapping,args.max_seq_length)

    print 'Finished Reading Data' 

    ###################################################################
    #    Interacting with model and creating computation graph        #
    ###################################################################

    # Creating Model object
    model = LanguageModel(args.lang,mapping,args.representation,
                    args.max_seq_length, args.embedding_size,args.rnn_size)

    ## Creating placeholder for sequences, masks and lengths and dropout keep probability 
    pl_batch_sequences = tf.placeholder(shape=[None,args.max_seq_length],dtype=tf.int32)
    pl_batch_sequence_lengths = tf.placeholder(shape=[None],dtype=tf.float32)
    loss_op = model.average_loss(pl_batch_sequences, pl_batch_sequence_lengths, 1.0)

    #Saving model
    saver = tf.train.Saver(max_to_keep = 0)

    #Start Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    #sess.run(tf.initialize_all_variables())
    saver.restore(sess,args.model_fname)
    
    print "Session started"

    ## TEST LOSS
    test_start_time=time.time()
    test_loss=get_average_loss(test_data, loss_op,
                        pl_batch_sequences, pl_batch_sequence_lengths, sess)
    test_end_time=time.time()
    epoch_test_time=(test_end_time-test_start_time)

    print "Test Perplexity: {}".format(test_loss)
    print "Test Time (hh:mm:ss)::: {}".format(utilities.formatted_timeinterval(epoch_test_time))

def run_print_vars(args): 
    """
    Debugging: print the names of the variables in the model
    """


    ## check for required parameters
    if args.lang is None:
        print 'ERROR: --lang has to be set'
        sys.exit(1)

    if args.model_fname is None:
        print 'ERROR: --model_fname has to be set'
        sys.exit(1)

    if args.mapping_fname is None:
        print 'ERROR: --mapping_fname has to be set'
        sys.exit(1)

    #######################################
    # Reading data and creating mappings  #
    #######################################

    # Creating mapping object to store char-id mappings
    mapping = Mapping.get_mapping_instance(args.mapping_class) 
    with open(args.mapping_fname,'r') as mapping_json_file:     
        mapping.load_mapping(mapping_json_file)
    print 'Mapping'
    print mapping

    print 'Vocabulary Statitics'
    print '{}: {}'.format(args.lang,mapping.get_vocab_size())

    print 'Finished Reading Data' 

    ###################################################################
    #    Interacting with model and creating computation graph        #
    ###################################################################

    # Creating Model object
    model = LanguageModel(args.lang,mapping,args.representation,
                    args.max_seq_length, args.embedding_size,args.rnn_size)

    for v in tf.global_variables(): 
        print v.name

    ## Creating placeholder for sequences, masks and lengths and dropout keep probability 
    pl_batch_sequences = tf.placeholder(shape=[None,args.max_seq_length],dtype=tf.int32)
    pl_batch_sequence_lengths = tf.placeholder(shape=[None],dtype=tf.float32)
    loss_op = model.average_loss(pl_batch_sequences, pl_batch_sequence_lengths, 1.0)

    #Saving model
    saver = tf.train.Saver(max_to_keep = 0)

    #Start Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    #sess.run(tf.initialize_all_variables())
    saver.restore(sess,args.model_fname)

    print 'after loading' 

    for v in tf.global_variables(): 
        print v.name

if __name__ == '__main__' :

    print 'Process started at: ' + time.asctime()

    #### Load Indic NLP Library ###
    ## Note: Environment variable: INDIC_RESOURCES_PATH must be set
    loader.load()

    #####################################
    #    Command line argument parser   #
    #####################################

    # Creating parser
    parser = argparse.ArgumentParser(description="""Train and test character level neural language models""" 
            """using onehot and phonetic representations""")
    subparsers = parser.add_subparsers(description='Various subcommands for language modelling')

    ## common options
    parent_parser = argparse.ArgumentParser(description="Common Options",add_help=False) 
    parent_parser.add_argument('--lang', type = str, default = None, help = 'Language')
    parent_parser.add_argument('--embedding_size', type = int, default = 256, help = 'size of character representation')
    parent_parser.add_argument('--rnn_size', type = int, default = 512, help = 'size of output of encoder RNN')
    parent_parser.add_argument('--max_seq_length', type = int, default = 30, help = 'maximum sequence length')
    parent_parser.add_argument('--batch_size', type = int, default = 32, help = 'size of each batch used in training')
    parent_parser.add_argument('--representation', type = str, default = 'onehot',  help = """input representation, which can be specified in two ways: 
    (i) one of "phonetic", "onehot", "onehot_and_phonetic" """)
    parent_parser.add_argument('--mapping_class', type = str, default = 'IndicPhoneticMapping',  help = """class to be used for mapping. 
            Possible values: IndicPhoneticMapping, CharacterMapping""")

    ### Training options 
    parser_train = subparsers.add_parser('train', help='Train a neural language model', parents=[parent_parser])
    parser_train.add_argument('--data_dir', type = str, help = """input directory contaning three files. (train,valid,test).txt. 
                    Format of each file: one sequence per line, space separate elements on each line""")
    parser_train.add_argument('--output_dir', type = str, help = 'Output directory')
    parser_train.add_argument('--use_mapping', type = str, default = None, help = 'If you want to use an existing vocabulary, provide the path to the mapping file')

    parser_train.add_argument('--max_epochs', type = int, default = 30, help = 'maximum number of epochs')
    parser_train.add_argument('--learning_rate', type = float, default = 0.001, help = 'learning rate of Adam Optimizer')
    parser_train.add_argument('--dropout_keep_prob', type = float, default = 0.5, help = 'keep probablity for the dropout layers')
    parser_train.add_argument('--infer_every', type = int, default = 1, help = 'write predicted outputs for test data after these many epochs, 0 if not required')
    parser_train.add_argument('--start_from', type = int, default = None, help = 'epoch to restore model from. This must be one of the final epochs from previous runs')
    parser_train.set_defaults(func=run_train)

    ## test options 
    parser_test = subparsers.add_parser('test', help='test a neural language model',parents=[parent_parser])
    parser_test.add_argument('--model_fname', type = str, help = 'model file name')
    parser_test.add_argument('--mapping_fname', type = str, help = 'mapping file')
    parser_test.add_argument('--in_fname', type = str, help = 'input file')
    parser_test.set_defaults(func=run_test)

    ## debugging options 
    parser_test = subparsers.add_parser('print_vars', help='print list of variable names',parents=[parent_parser])
    parser_test.add_argument('--model_fname', type = str, help = 'model file name')
    parser_test.add_argument('--mapping_fname', type = str, help = 'mapping file')
    parser_test.set_defaults(func=run_print_vars)

    args = parser.parse_args()

    print '========== Parameters start ==========='
    for k,v in vars(args).iteritems():
        print '{}: {}'.format(k,v)
    print '========== Parameters end ============='

    args.func(args)


    print 'Process terminated at: ' + time.asctime()

