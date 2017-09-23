import argparse
import os
import sys
import codecs
import itertools as it
import pickle 
import numpy as np
import mpld3
import calendar,time
from collections import defaultdict

import Mapping
import MonoDataReader
import utilities

import AttentionModel

import tensorflow as tf
import numpy as np

from sklearn.manifold import TSNE;
from sklearn.decomposition import PCA;
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from indicnlp import loader
from indicnlp.script import indic_scripts as isc
from indicnlp.transliterate import unicode_transliterate  as indtrans

from cfilt.transliteration.analysis import slavic_characters 

"""
Flags to be used in the program 
"""

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('in_fname', '', 
        'input file containing sequences to analyse. one sequence per line.')
tf.app.flags.DEFINE_string('out_img_fname', '', 
        'output image file showing contextual representations of characters to be analyzed')
tf.app.flags.DEFINE_string('out_html_fname', '', 
        'output HTML file showing contextual representations of characters to be analyzed, along with interactive context display')
#tf.app.flags.DEFINE_string('char_fname', '', 
#        'file containing characters for which embeddings are to be extracted. one character per line.')
tf.app.flags.DEFINE_string('model_fname', '', 
        'model file to load parameters from')
tf.app.flags.DEFINE_string('mapping_dir', '', 
        'mapping directory which contains mappings')
tf.app.flags.DEFINE_string('lang', '', 
        'language under study')

tf.app.flags.DEFINE_string('enc_type',  'cnn', 
        'encoder to use. One of (1)simple_lstm_noattn (2) bilstm (3) cnn')
tf.app.flags.DEFINE_string('representation',  'onehot', 
        'input representation, which can be specified in two ways: (i) one of "phonetic", "onehot", "onehot_and_phonetic"')
tf.app.flags.DEFINE_string('shared_mapping_class',  'IndicPhoneticMapping', 
        'class to be used for shared mapping. Possible values: IndicPhoneticMapping, CharacterMapping')

tf.app.flags.DEFINE_integer('embedding_size',  256, 'size of character representation')
tf.app.flags.DEFINE_integer('enc_rnn_size',  512, 'size of output of encoder RNN')
tf.app.flags.DEFINE_integer('dec_rnn_size',  512, 'size of output of decoder RNN')
tf.app.flags.DEFINE_integer('max_seq_length',  30, 'maximum sequence length')
tf.app.flags.DEFINE_integer('batch_size',  32, 'size of each batch used in training')
tf.app.flags.DEFINE_integer('window_size', 1, 'size of window for building representation')

tf.app.flags.DEFINE_boolean('separate_output_embedding',  False, 
        """Should separate embeddings be used on the input and output side."""
        """Generally the same embeddings are to be used. This is used only for Indic-Indic transliteration, """
        """when input is phonetic and output is onehot_shared""")

def input_chars_to_analyze():
    """
    Input in code: what characters to input 
    """
    chars_to_analyze = []
    if FLAGS.lang == 'en': 
        chars_to_analyze=['A','E','I','O','U']
    elif isc.is_supported_language(FLAGS.lang): 
        offsets_to_analyze= range(0x3e, 0x4d)  ## only vowel diacritics included
        chars_to_analyze = [ isc.offset_to_char(x,FLAGS.lang) for x in offsets_to_analyze ]
    elif slavic_characters.is_supported_language_latin(FLAGS.lang): 
        chars_to_analyze = slavic_characters.latin_vowels
        #chars_to_analyze=['A','E','I','O','U']
        #chars_to_analyze=['K','C','F','V','P','B']
    
    return chars_to_analyze

####  Global functions ####


def init_representation(): 

    representation={}

    ### parse representation argument 
    if FLAGS.representation in ['onehot','onehot_shared','phonetic','onehot_and_phonetic']: 
        representation = {} 
        representation[FLAGS.lang]=FLAGS.representation 
    else: 
        representation = dict([ x.split(':') for x in FLAGS.representation.split(',') ])

    ## Print Representation and Mappings 
    print 'Representation'
    print representation 

    return representation

def init_mapping(representation): 
    ### load the mapping

    mapping={}
    shared_mapping_obj = Mapping.get_mapping_instance(FLAGS.shared_mapping_class) 

    for lang in representation.keys(): 
        if representation[lang] in ['phonetic','onehot_and_phonetic']: 
            mapping[lang]=shared_mapping_obj
        elif representation[lang]=='onehot_shared': 
            mapping[lang]=shared_mapping_obj
        elif representation[lang]=='onehot': 
            mapping[lang]=Mapping.CharacterMapping()

        with open(FLAGS.mapping_dir+'/'+'mapping_'+lang+'.json','r') as mapping_file:     
            mapping[lang].load_mapping(mapping_file)

    ## Print Representation and Mappings 
    print 'Mapping'
    print mapping

    print 'Vocabulary Statitics'
    for lang in representation.keys(): 
        print '{}: {}'.format(lang,mapping[lang].get_vocab_size())

    return mapping        

def compute_hidden_representation(model, sequences, sequence_lengths, lang):
    """

    Compute hidden representation using the encoder for the 'sequences'

    Parameters: 

    sequences: Tensor of integers of shape containing the input symbol ids (batch_size x max_seq_length)
    sequence_lengths: Tensor of shape (batch_size) containing length of each  sequence input 
    lang: language of the input
    dropout_keep_prob: dropout keep probability

    Return: 
     a tuple (states, enc_outputs)
      states: final state of the encoder, shape: (batch_size x encoder_state_size)
      enc_outputs: list of Tensors with shape (batch_size x enc_output_size). 
        Length of list=max_seq_length. One element in the list for each timestamp

    """
    dropout_keep_prob=tf.constant(1.0)
    sequence_embeddings = tf.add(tf.nn.embedding_lookup(model.embed_W[lang],sequences),model.embed_b[lang])
    _ , enc_outputs = model.input_encoder.encode(sequence_embeddings,sequence_lengths,dropout_keep_prob)

    return tf.transpose(tf.stack(enc_outputs),perm=[1,0,2])


def get_label(x,lang): 
    if isc.is_supported_language(lang): 
        if isc.in_coordinated_range(x,lang): 
            return indtrans.ItransTransliterator.to_itrans(x,lang) + '({:2x})'.format(isc.get_offset(x,lang))
        else: 
            return str(hex(ord(x)))
    else: 
        return x


def main(argv=None): 
    """
     Main function for the program 
    """

    def prepare_data(): 

        print 'Reading test data' 

        test_data = MonoDataReader.MonoDataReader(FLAGS.lang, FLAGS.in_fname,mapping[FLAGS.lang],FLAGS.max_seq_length)
        sequences, sequence_masks, sequence_lengths = test_data.get_data()

        #return (sequences, None, sequence_lengths, sequence_masks)

        new_seq_data=[] 
        new_seq_lengths=[]
        new_seq_pos=[]

        for i in range(0,sequences.shape[0]): 
            for j in range(0,sequence_lengths[i]): 
                if sequences[i,j] in char_ids_to_analyze:
                    start= max(1,j-FLAGS.window_size)                      ## GO not considered 
                    end  = min(sequence_lengths[i]-1,j+FLAGS.window_size+1) ## EOW not considered 
                    l=end-start+2                                           ## 2 to account for EOW and GO
                    seq_slice=np.concatenate(   [ 
                                                    [mapping[FLAGS.lang].get_index(Mapping.Mapping.GO)],
                                                    sequences[i,start:end], 
                                                    [mapping[FLAGS.lang].get_index(Mapping.Mapping.EOW)],
                                                    [mapping[FLAGS.lang].get_index(Mapping.Mapping.PAD)]*(FLAGS.max_seq_length-l),
                                                ]  )
                    new_seq_data.append(seq_slice)
                    new_seq_lengths.append(l)
                    new_seq_pos.append(j- start + 1)

        ### add points for the vocabulary without context 
        ## single character 
        #for cid in char_ids_to_analyze: 
        #    seq_slice=np.concatenate(   [ 
        #                                    [mapping[FLAGS.lang].get_index(Mapping.Mapping.GO)],
        #                                    [cid], 
        #                                    [mapping[FLAGS.lang].get_index(Mapping.Mapping.EOW)],
        #                                    [mapping[FLAGS.lang].get_index(Mapping.Mapping.PAD)]*(FLAGS.max_seq_length-3),
        #                                ]  )
        #    new_seq_data.append(seq_slice)
        #    new_seq_lengths.append(3)
        #    new_seq_pos.append(1)

        #for cid in char_ids_to_analyze: 
        ## character thrice
        #    seq_slice=np.concatenate(   [ 
        #                                    [mapping[FLAGS.lang].get_index(Mapping.Mapping.GO)],
        #                                    [cid]*3, 
        #                                    [mapping[FLAGS.lang].get_index(Mapping.Mapping.EOW)],
        #                                    [mapping[FLAGS.lang].get_index(Mapping.Mapping.PAD)]*(FLAGS.max_seq_length-5),
        #                                ]  )
        #    new_seq_data.append(seq_slice)
        #    new_seq_lengths.append(5)
        #    new_seq_pos.append(2)

        # Creating masks. Mask has size = size of list of sequence. 
        # Corresponding to each PAD character there is a zero, for all other there is a 1
        new_seq_masks = np.zeros([ len(new_seq_data), FLAGS.max_seq_length], dtype = np.float32)  
        for i in range(len(new_seq_data)):
            new_seq_masks[i][:new_seq_lengths[i]]=1

        return (np.asarray(new_seq_data,dtype=np.int32), 
                np.asarray(new_seq_pos,dtype=np.int32), 
                np.asarray(new_seq_lengths,dtype=np.int32), 
                new_seq_masks )

    def create_graph(): 

        print "Start graph creation"
        # Creating Model object
        model = AttentionModel.AttentionModel(mapping,representation,FLAGS.max_seq_length,
                FLAGS.embedding_size,FLAGS.enc_rnn_size,FLAGS.dec_rnn_size,
                FLAGS.enc_type,FLAGS.separate_output_embedding)

        ## Creating placeholder for sequences, masks and lengths and dropout keep probability 
        batch_sequences = tf.placeholder(shape=[None,FLAGS.max_seq_length],dtype=tf.int32)
        batch_sequence_lengths = tf.placeholder(shape=[None],dtype=tf.float32)

        # Predict output for test sequences
        o_enc_outputs = compute_hidden_representation(model,batch_sequences,batch_sequence_lengths,FLAGS.lang)

        return batch_sequences, batch_sequence_lengths, o_enc_outputs
        print "Done with creating graph. Starting session"

    def run_graph(): 

        print "Starting session"

        saver = tf.train.Saver(max_to_keep = 3)

        #Start Session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,FLAGS.model_fname)
        
        print "Session started"


        test_time=0.0
        b_enc_outputs_list=[]

        print 'Starting execution'
        for start in xrange(0,sequences.shape[0],FLAGS.batch_size):
            end = min(start+FLAGS.batch_size,sequences.shape[0])

            batch_start_time=time.time()

            data_sequences=sequences[start:end,:]
            data_sequence_lengths=sequence_lengths[start:end]

            b_enc_outputs = sess.run(o_enc_outputs, 
                    feed_dict={batch_sequences: data_sequences, batch_sequence_lengths: data_sequence_lengths}) 
            b_enc_outputs_list.append(b_enc_outputs)

            batch_end_time=time.time()
            test_time+=(batch_end_time-batch_start_time)

            print 'Encoded {} of {} sequences'.format(end,sequences.shape[0])
            sys.stdout.flush()

        enc_outputs=np.concatenate(b_enc_outputs_list,axis=0)
        print 'Ending execution'

        return enc_outputs

    ################## WORK STARTS HERE ############

    ##### Obtaining Encoder Embeddings 
    representation=init_representation()

    mapping=init_mapping(representation)

    chars_to_analyze = input_chars_to_analyze()

    char_ids_to_analyze= [ mapping[FLAGS.lang].get_index(x,FLAGS.lang) for x in chars_to_analyze ]

    sequences, sequence_pos, sequence_lengths, sequence_masks, = prepare_data()

    batch_sequences, batch_sequence_lengths, o_enc_outputs = create_graph()

    enc_outputs=run_graph()

    #### Prepare data for visualization

    char_ctx_embed_list=[]
    char_list=[]

    #### for window based approach 
    for i in range(0,sequences.shape[0]): 
        pos=sequence_pos[i]
        char_ctx_embed_list.append(enc_outputs[i,pos,:])
        char_list.append(sequences[i,pos])

    char_ctx_embed_flat=np.array(char_ctx_embed_list)

    ### call tsne 
    low_embedder=TSNE()
    low_embeddings=low_embedder.fit_transform(char_ctx_embed_flat)
    
    ### plot
    N = low_embeddings.shape[0]
    x = low_embeddings[:,0]
    y = low_embeddings[:,1]

    cols_list = np.arange(0.0,1.0,1/float(len(chars_to_analyze)))
    char_col_map={}
    for i,c in enumerate(char_ids_to_analyze): 
        char_col_map[c]=cols_list[i]

    colors_data = [ char_col_map[c] for c in char_list ]        

    print N
    cm = plt.get_cmap('jet') 
    vs=len(char_ids_to_analyze)

    fig,ax=plt.subplots()
    scatter=ax.scatter(x, y, c=colors_data, cmap=cm, alpha=0.5)
    #plt.scatter(x[-vs:], y[-vs:], c=colors_data[-vs:], cmap=cm, alpha=1.0, marker='x')

    gen_label=lambda char: (indtrans.ItransTransliterator.to_itrans(char,FLAGS.lang) + '({:2x})'.format(isc.get_offset(char,FLAGS.lang)))
    patches=[ mpatches.Patch(label=get_label(char,FLAGS.lang), color=cm(color)) for char, color in zip(chars_to_analyze, cols_list) ]
    ax.legend(handles=patches,ncol=3,fontsize='xx-small')

    labels=[]
    for i in range(0,sequences.shape[0]): 
        labels.append(u''.join([ mapping[FLAGS.lang].get_char(sequences[i,j],FLAGS.lang)  for j in range(1,sequence_lengths[i]-1) ]))

    tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
    mpld3.plugins.connect(fig, tooltip)
    
    if FLAGS.out_html_fname != '': 
        mpld3.save_html(fig,FLAGS.out_html_fname)
    ##mpld3.show(ip='10.129.2.170',port=10002, open_browser=False)

    if FLAGS.out_img_fname != '': 
        plt.savefig(FLAGS.out_img_fname)
    ##plt.show()

if __name__ == '__main__' :

    print 'Process started at: ' + time.asctime()

    #### Load Indic NLP Library ###
    ## Note: Environment variable: INDIC_RESOURCES_PATH must be set
    loader.load()

    tf.app.run()

    print 'Process ended at: ' + time.asctime()

