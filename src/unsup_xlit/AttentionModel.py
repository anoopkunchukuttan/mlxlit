import Mapping
import encoders

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

class AttentionModel():

    def __init__(self,mapping,representation,max_sequence_length,
                    embedding_size,enc_rnn_size,dec_rnn_size,
                    enc_type='cnn',separate_output_embedding=False):

        self.max_sequence_length = max_sequence_length
        self.enc_type = enc_type 

        self.embedding_size = embedding_size
        self.enc_rnn_size = enc_rnn_size
        self.dec_rnn_size = dec_rnn_size

        self.mapping = mapping
        self.representation=representation

        self.lang_list = self.mapping.keys()

        ### use shared decoders or not
        print 'Using a shared decoder for all target languages'
        self.is_shared_decoder=True
        #print 'Using a separate decoder for each target language'
        #self.is_shared_decoder=False

        ### share output layers or not
        print 'Using a different output layer for all target languages'
        self.use_shared_output=False
        #print 'Using a shared output for all target languages'
        #self.use_shared_output=True

        ### do u want separate input and output embedding vectors (for the same language)
        ### Value should be False. 
        ### Set to True only to experiment for indic-indic pair where input embedding  is phonetic (experimental support)
        self.separate_output_embedding=separate_output_embedding
        if self.separate_output_embedding: 
            print 'Using separate input and output embeddings'
        else: 
            print 'Using the same embedding for input and output'

        self.vocab_size={}
        for lang in self.lang_list: 
            self.vocab_size[lang] = self.mapping[lang].get_vocab_size()

        max_val = 0.1

        ####### Input embeddings 
        self.embed_W = dict()
        self.embed_b = dict()

        # Finds bit-vector representation for each character of each language
        self.bitvector_embeddings={}
        self.bitvector_embedding_size={}
        for lang in self.lang_list:
            self.bitvector_embeddings[lang] = tf.constant(self.mapping[lang].get_bitvector_embeddings(lang,self.representation[lang]),dtype=tf.float32)
            self.bitvector_embedding_size[lang]=self.mapping[lang].get_bitvector_embedding_size(self.representation[lang])

        # Converting the character representation to embedding_size vector
        ## for sharing embeddings 
        self.embed_W0=None
        self.embed_b0=None

        for lang in self.lang_list:
            if self.representation[lang] in  ['phonetic','onehot_and_phonetic','onehot_shared']:
                if self.embed_W0 is None: 
                    self.embed_W0 = tf.Variable(tf.random_uniform([self.bitvector_embedding_size[lang],self.embedding_size], -1*max_val, max_val), name = 'embed_W0')
                    self.embed_b0 = tf.Variable(tf.constant(0., shape=[self.embedding_size]), name = 'embed_b0')

                self.embed_W[lang] = tf.matmul(self.bitvector_embeddings[lang], self.embed_W0, name='embed_W_{}'.format(lang))
                self.embed_b[lang] = self.embed_b0
            elif self.representation[lang] ==  'onehot':
                ### TODO: maybe multiplication by identity bitvector embedding is not required
                x = tf.Variable(tf.random_uniform([self.bitvector_embedding_size[lang],self.embedding_size], -1*max_val, max_val))
                self.embed_W[lang] = tf.matmul(self.bitvector_embeddings[lang], x, name='embed_W_{}'.format(lang))
                self.embed_b[lang] = tf.Variable(tf.constant(0., shape=[self.embedding_size]), name = 'embed_b_{}'.format(lang))

        ####### Output embeddings 
        # Finds bit-vector representation for each character of each language (for the output side)
        # This code block is needed only if you want a onehot shared representation on the output side irrespective
        # of the input representation 

        self.embed_outW = dict()
        self.embed_outb = dict()

        if not self.separate_output_embedding: 
            self.embed_outW=self.embed_W
            self.embed_outb=self.embed_b
        else:            

            out_representation='onehot_shared'  ### output side uses onehot_shared representation

            self.out_bitvector_embeddings={}
            self.out_bitvector_embedding_size={}
            for lang in self.lang_list:
                self.out_bitvector_embeddings[lang] = tf.constant(self.mapping[lang].get_bitvector_embeddings(lang,out_representation),dtype=tf.float32)
                self.out_bitvector_embedding_size[lang]=self.mapping[lang].get_bitvector_embedding_size(out_representation)

            # Converting the character representation to embedding_size vector
            ## for sharing embeddings 
            self.embed_outW0=None
            self.embed_outb0=None

            for lang in self.lang_list:
                if self.embed_outW0 is None: 
                    self.embed_outW0 = tf.Variable(tf.random_uniform([self.out_bitvector_embedding_size[lang],self.embedding_size], -1*max_val, max_val), name = 'embed_outW0')
                    self.embed_outb0 = tf.Variable(tf.constant(0., shape=[self.embedding_size]), name = 'embed_outb0')

                self.embed_outW[lang] = tf.matmul(self.out_bitvector_embeddings[lang], self.embed_outW0, name='embed_outW_{}'.format(lang))
                self.embed_outb[lang] = self.embed_outb0

        ##### Create Encoder
        self.input_encoder=None
        
        if self.enc_type == 'simple_lstm_noattn':
            ## Simple RNN Encoder 
            self.input_encoder=encoders.SimpleRnnEncoder(embedding_size,max_sequence_length,enc_rnn_size)
        elif self.enc_type == 'bilstm':
            ### Bidirectional RNN encoder 
            self.input_encoder=encoders.BidirectionalRnnEncoder(embedding_size,max_sequence_length,enc_rnn_size)
        elif self.enc_type == 'cnn':
            ## CNN Encoder
            filter_sizes=[1,2,3,4]
            self.input_encoder=encoders.CNNEncoder(embedding_size,max_sequence_length,filter_sizes,enc_rnn_size*2/len(filter_sizes))

        ## FIXME: what is the best way to initialize the input - I suppose with embedding for GO symbol
        ## the variable need not even be saved
        #self.decoder_input = dict()
        #for lang in self.lang_list:
        #    self.decoder_input[lang] = tf.random_uniform([1, embedding_size], dtype = tf.float32)

        ### Create Decoder

        self.decoder_cell = dict()
        with tf.variable_scope('decoder'):
            shared_decoder=rnn_cell.BasicLSTMCell(dec_rnn_size)
            for lang in self.lang_list:  ## TODO: can we restrict which languages decoder is created for?
                if self.is_shared_decoder:
                    #### shared decoder
                    self.decoder_cell[lang] =  shared_decoder 
                else:
                    #### decoder per language 
                    self.decoder_cell[lang] = rnn_cell.BasicLSTMCell(dec_rnn_size)
            self.dec_state_size=self.decoder_cell.values()[0].state_size
       
        ### Encoder state to decoder adapter 
        self.state_adapt_W = tf.Variable(tf.random_uniform([self.input_encoder.get_state_size(),self.dec_state_size],
            -1*max_val, max_val), name = 'state_adapt_W')
        self.state_adapt_b = tf.Variable(tf.constant(0., shape=[self.dec_state_size]), name = 'state_adapt_b')

        ### Output layer

        # Output decoder to vocab_size vector
        self.out_W = dict()
        self.out_b = dict()

        ### shared output params
            #  a hackish way of getting one of the shared languages Be careful - this may break - only experimental use!
        shared_outvocab_size = self.vocab_size[filter(lambda l:l not in ['en','ar','zh'] , self.lang_list)[0]]
        out_W_shared = tf.Variable(tf.random_uniform([self.dec_rnn_size,shared_outvocab_size], -1*max_val, max_val), 
                                    name='out_W_shared')
        out_b_shared = tf.Variable(tf.constant(0., shape = [shared_outvocab_size]), name='out_b_shared')

        for lang in self.lang_list:
            if self.use_shared_output: 
                self.out_W[lang]=out_W_shared
                self.out_b[lang]=out_b_shared
            else: 
                self.out_W[lang] = tf.Variable(tf.random_uniform([self.dec_rnn_size,self.vocab_size[lang]], -1*max_val, max_val),
                                                name='out_W_{}'.format(lang))
                self.out_b[lang] = tf.Variable(tf.constant(0., shape = [self.vocab_size[lang]]),
                                            name='out_b_{}'.format(lang))
        
        ## Hack for zeroshot transliteratoin for Model 2 (en-indic)               
        #self.out_W['hi']=(self.out_W['ta']+self.out_W['kn']+self.out_W['bn'])/3.0
        #self.out_b['hi']=(self.out_b['ta']+self.out_b['kn']+self.out_b['bn'])/3.0
        #self.out_W['hi']=self.out_W['bn']
        #self.out_b['hi']=self.out_b['bn']

        ## Attention mechanism paramters
        ## size of the context vector (will be twice encoder cell output size for bidirectional RNN)
        self.ctxvec_size=self.input_encoder.get_output_size()

        ## Attention Neural Network
        self.attn_W = tf.Variable(tf.random_uniform([self.dec_state_size+self.embedding_size+self.ctxvec_size,1],
            -1*max_val, max_val), name = 'attn_W')
        self.attn_b = tf.Variable(tf.constant(0., shape=[1]), name = 'attn_b')

    def compute_hidden_representation(self, sequences, sequence_lengths, lang, dropout_keep_prob):
        """

        Compute hidden representation using the encoder for the 'sequences'

        Parameters: 

        sequences: Tensor of integers of shape containing the input symbol ids (batch_size x max_sequence_length)
        sequence_lengths: Tensor of shape (batch_size) containing length of each  sequence input 
        lang: language of the input
        dropout_keep_prob: dropout keep probability

        Return: 
         a tuple (states, enc_outputs)
          states: final state of the encoder, shape: (batch_size x encoder_state_size)
          enc_outputs: list of Tensors with shape (batch_size x enc_output_size). 
            Length of list=max_sequence_length. One element in the list for each timestamp

        """
        sequence_embeddings = tf.add(tf.nn.embedding_lookup(self.embed_W[lang],sequences),self.embed_b[lang])
        states, enc_outputs = self.input_encoder.encode(sequence_embeddings,sequence_lengths,dropout_keep_prob)
        return states, enc_outputs

    def compute_attention_context(self,prev_state,prev_out_embed,enc_outputs):
        """
            Compute the annotation/context vector using the attention mechanism,
            which can be used by the decoder for predicting the next symbol

            Paramters: 

            prev_state: Previous decoder state. 
                        Shape: batch_size x decoder_state_size 
            prev_out_embed: Embedding for the previous decoder output. 
                            Shape: batch_size x output_embedding_size
            enc_outputs: list of Tensors with shape (batch_size x enc_output_size). 
                Length of list=max_sequence_length. One element in the list for each timestamp


            Returns: annotation/context vector of shape (batch_size x enc_output_size)
        """

        batch_size=tf.shape(prev_state)[0]

        ## reshaping and transposing enc_outputs
        a3=tf.pack(enc_outputs)
        a4=tf.transpose(a3,[1,0,2])
        a5=tf.reshape(a4,[-1,self.ctxvec_size],name='attn__a5__enc_outputs_shaped')

        num_ctx_vec=self.max_sequence_length
        #num_ctx_vec=tf.shape(a3)[0]

        ## getting prev_state and prev_out_embed in the correct shape
        ## and duplicate them for cacatenating with enc_outputs
        att_ref=tf.concat(1,[prev_state,prev_out_embed])
        a1=tf.tile(att_ref,[1,num_ctx_vec])
        a2=tf.reshape(a1,[-1,self.dec_state_size+self.embedding_size],name='attn__a2__state__embed_shaped')

        ## preparing the input to the attention network
        a6=tf.concat(1,[a2,a5],name='attn__a6__network_input')

        ###### Passing through attention network
        ## The network takes input for one encoder input and the current decoder state and output embedding and gives a scalar output
        ## for all encoder vectors, at current decoder position, for the entire batch)

        a7=tf.matmul(a6,self.attn_W)+self.attn_b
        a8=tf.nn.tanh(a7,name='attn__a7__network_output')
        a9=tf.reshape(a8,[-1,num_ctx_vec],name='attn__a9')

        ## apply softmax to compute the weights for the encoder outputs
        a10=tf.nn.softmax(a9,name='attn__a10__softmax')

        # computing context vector

        #### (a)
        #a11=tf.unpack(a10,name='attn__a11__ctxweighting_input1')
        #a12=tf.unpack(a4,num=batch_size,name='attn__a12__ctxweighting_input2')  ## organized as list, one for each input in batch (used later)
        #a13=[]
        #for i  in range(batch_size):
        #    a13.append(tf.matmul(tf.expand_dims(a11[i],0),a12[i],name='attn__a13__{}__ctx_weighting'.format(i)))
        #a14=tf.squeeze(tf.pack(a13),name='attn__a14__output')

        ###### (b)
        #a13=[]
        #def loop_func(batch_no):
        #    a11=tf.slice(a10,[batch_no,0],[1,self.max_sequence_length])
        #    a12=tf.slice(a5,[batch_no*self.max_sequence_length,0],[self.max_sequence_length,self.ctxvec_size])
        #    a13.append(tf.matmul(a11,a12,name='attn__a13__{}__ctx_weighting'.format(batch_no)))
        #    return tf.add(batch_no,1)

        #cond=lambda bno:tf.less(bno,batch_size)
        #bno=tf.constant(0)

        #tf.while_loop(cond,loop_func,[bno])
        #a14=tf.squeeze(tf.pack(a13),[1],name='attn__a14__output')

        ###### (c)
        #def loop_func(batch_no,_):
        #    a11=tf.slice(a10,[batch_no,0],[1,self.max_sequence_length])
        #    a12=tf.slice(a5,[batch_no*self.max_sequence_length,0],[self.max_sequence_length,self.ctxvec_size])
        #    c=tf.matmul(a11,a12,name='attn__a13__{}__ctx_weighting'.format(batch_no))
        #    return (tf.add(batch_no,1),c)

        #cond=lambda bno, _:tf.less(bno,batch_size) ## FIXME: hardcoding
        #bno=tf.constant(0)
        #ctxv=tf.random_uniform((1,self.ctxvec_size))

        #results=tf.while_loop(cond,loop_func,[bno,ctxv])
        #[ x[1] for x in results ]
        #a13=[ x[1] for x in results ]
        #print 'hello'
        #print a13
        #a14=tf.squeeze(tf.pack(a13),[1],name='attn__a14__output')

        #### (e)  This method finally worked and it is so simple and elegant!
        def loop_func(batch_no):
            a11=tf.slice(a10,[batch_no,0],[1,num_ctx_vec])
            a12=tf.slice(a5,[batch_no*num_ctx_vec,0],[num_ctx_vec,self.ctxvec_size])
            return tf.matmul(a11,a12)

        a13=tf.map_fn(loop_func,tf.range(0,batch_size),dtype=tf.float32,parallel_iterations=100)
        a14=tf.squeeze(a13,[1])

        return a14

    # Find cross entropy loss in predicting target_sequences from computed hidden representation (intial state)
    def seq_loss(self, target_sequence, target_masks, lang, initial_state, enc_output,dropout_keep_prob):

        batch_size = tf.shape(target_sequence)[0]

        state = tf.matmul(initial_state,self.state_adapt_W) + self.state_adapt_b 

        loss = 0.0
        cell = rnn_cell.DropoutWrapper(self.decoder_cell[lang],output_keep_prob=dropout_keep_prob)

        # One step generate one character for each sequence
        for i in range(self.max_sequence_length):
            # for first iteration, decoder_input embedding is used, otherwise, output from previous iteration is used
            # embedding lookup replace the character index with its embedding_size vector representation, which is given to the rnn_cell
            if(i==0):
                #current_emb = tf.reshape(tf.tile(self.decoder_input[lang],[batch_size,1]),[-1,self.embedding_size])
                x = tf.expand_dims(
                        tf.nn.embedding_lookup(self.embed_outW[lang],self.mapping[lang].get_index(Mapping.Mapping.GO))+self.embed_outb[lang],
                        0) 
                current_emb = tf.reshape(tf.tile(x,[batch_size,1]),[-1,self.embedding_size])
            else:
                current_emb = tf.nn.embedding_lookup(self.embed_outW[lang],target_sequence[:,i-1])+self.embed_outb[lang] 
            if i > 0 : tf.get_variable_scope().reuse_variables()

            ### compute the context vector 
            current_input=None
            if self.enc_type=='simple_lstm_noattn':
                current_input=current_emb
            else: 
                ## using the attention mechanism
                context=self.compute_attention_context(state,current_emb,enc_output)
                current_input=tf.concat(1,[current_emb,context])

            # Run one step of the decoder cell. Updates 'state' and store output in 'output'
            output = None
            with tf.variable_scope('decoder'):
                if i > 0 : tf.get_variable_scope().reuse_variables()
                output, state = cell(current_input,state)

            # Generating one-hot labels for target_sequences
            labels = tf.expand_dims(target_sequence[:,i],1)
            indices = tf.expand_dims(tf.range(0,batch_size),1)
            concated = tf.concat(1,[indices,labels])
            onehot_labels = tf.sparse_to_dense(concated,tf.pack([batch_size,self.vocab_size[lang]]),1.0,0.0)

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
    def seq_loss_2(self,
                    lang1,sequences,sequence_masks,sequence_lengths,
                    lang2,target_sequences,target_sequence_masks,target_sequence_lengths,
                    dropout_keep_prob):
        hidden_representation, enc_output = self.compute_hidden_representation(sequences,sequence_lengths,lang1,dropout_keep_prob)
        loss = self.seq_loss(target_sequences,target_sequence_masks,lang2,hidden_representation,enc_output,dropout_keep_prob)
        return loss

    # Get a monolingual optimizer for 'lang' language
    # sequences, sequence masks: tensors of shape: [batch_size, max_sequence lengths]
    # sequence_lengths: tensor of shape: [batch_sizes]
    def get_parallel_optimizer(self,learning_rate,
                    lang1,sequences,sequence_masks,sequence_lengths,
                    lang2,target_sequences,target_sequence_masks,target_sequence_lengths,
                    dropout_keep_prob):
        hidden_representation, enc_output = self.compute_hidden_representation(sequences,sequence_lengths,lang1,dropout_keep_prob)
        loss = self.seq_loss(target_sequences,target_sequence_masks,lang2,hidden_representation,enc_output,dropout_keep_prob)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return [ optimizer, loss ]

    def transliterate_beam(self, source_lang, sequences, sequence_lengths, target_lang, beam_size, topn):
        """
            Decode using the trained seq2seq model with beam search and return the topn results and scores

            Note: The implementation generates a sequence of length max_sequence_length always. 
            It might be better to stop the sequence generation after EOW is generated.

            Parameters: 

            source_lang: input language
            sequences: Tensor of integers of shape containing the input symbol ids 
                        Shape: (batch_size x max_sequence_length)
            sequence_lengths: Tensor of shape (batch_size) containing length of each  sequence input 
            target_lang: target language
            beam_size: size of beam used for beam search while decoding
            topn: get the 'topn' best outputs 

            Outputs: 

            final_outputs: Tensor containing symbol ids of shape (batch_size x topn x max_sequence_length)
            final_scores : Tensor containing log-likelihood scores for each candidate of shape (batch_size x topn).  
                           The scores are length normalized
            
        """

        #### compute hidden representation first     
        initial_state, enc_output = self.compute_hidden_representation(sequences,sequence_lengths, source_lang,tf.constant(1.0))
        initial_state = tf.matmul(initial_state,self.state_adapt_W) + self.state_adapt_b 

        ### start decoding 

        batch_size = tf.shape(sequences)[0]
        cur_beam_size=1

        prev_states = tf.reshape(tf.tile(initial_state, [1,cur_beam_size]), [-1,self.dec_state_size]) 
        prev_scores = tf.tile(tf.constant(0.0,shape=[1,1]),[batch_size*cur_beam_size,1])
        prev_symbols = None

        cell = rnn_cell.DropoutWrapper(self.decoder_cell[target_lang],output_keep_prob=tf.constant(1.0))

        prev_best_outputs=None

        final_outputs=None
        final_scores=None

        for i in range(self.max_sequence_length):
            if(i==0):
                #current_emb = tf.reshape(tf.tile(self.decoder_input[target_lang],[batch_size,1]),[-1,self.embedding_size])
                x = tf.expand_dims(
                        tf.nn.embedding_lookup(self.embed_outW[target_lang],self.mapping[target_lang].get_index(Mapping.Mapping.GO))+self.embed_outb[target_lang],
                        0)
                current_emb = tf.reshape(tf.tile(x,[batch_size,1]),[-1,self.embedding_size])
            else:
                current_emb = tf.nn.embedding_lookup(self.embed_outW[target_lang],tf.reshape(prev_symbols,[-1]))+self.embed_outb[target_lang]

            if i > 0 : tf.get_variable_scope().reuse_variables()

            ### compute the context vector 
            current_input=None
            if self.enc_type=='simple_lstm_noattn':
                current_input=current_emb
            else: 
                ## using the attention mechanism
                context=self.compute_attention_context(prev_states,current_emb,enc_output)
                current_input=tf.concat(1,[current_emb,context])

            # Run one step of the decoder cell. Updates 'state' and store output in 'output'
            output = None
            state = None
            with tf.variable_scope('decoder'):
                if i > 0 : tf.get_variable_scope().reuse_variables()
                output, state = cell(current_input,prev_states)

            logit_words = tf.add(tf.matmul(output,self.out_W[target_lang]),self.out_b[target_lang])

            ### check dimentionality and orientation 
            prev_scores = prev_scores + tf.nn.log_softmax(logit_words)

            prev_scores_by_instance = tf.reshape(prev_scores,[-1,cur_beam_size*self.vocab_size[target_lang]])

            best_scores, best_indices = tf.nn.top_k(prev_scores_by_instance, beam_size)

            best_symbols = best_indices % self.vocab_size[target_lang] 
            best_prev_beams = best_indices // self.vocab_size[target_lang] 

            best_flat_indices = tf.tile(tf.reshape(tf.range(0,batch_size),[-1,1]),[1,cur_beam_size])*cur_beam_size + best_prev_beams

            ### update variables for the next iteration 
            prev_scores = tf.reshape(best_scores, [-1,1])
            prev_symbols = tf.reshape(best_symbols, [-1,1])
           
            prev_states=  tf.gather( state, tf.reshape(best_flat_indices,[-1]) )

            #### compute best-k candidates now 
            prev_best_outputs_for_final=prev_best_outputs
            if(i==0): 
                prev_best_outputs=prev_symbols
            else: 
                top_beam_prev_outputs = tf.gather( prev_best_outputs, tf.reshape(best_flat_indices,[-1]) )
                prev_best_outputs = tf.concat(1,[top_beam_prev_outputs,prev_symbols])

            ##### update beam size after first iteration 
            if i==0:
                cur_beam_size=beam_size
                enc_output=tf.unpack(
                        tf.reshape(   tf.tile(tf.pack(  enc_output   ),[1,1,beam_size]),
                                    [self.max_sequence_length,-1,self.input_encoder.get_output_size()]
                                  )
                        )

            #### get top-n outputs for the last iteration                 
            if i==self.max_sequence_length-1: 
                final_scores, final_indices = tf.nn.top_k(prev_scores_by_instance, topn)

                final_symbols    = final_indices %  self.vocab_size[target_lang] 
                final_prev_beams = final_indices // self.vocab_size[target_lang] 

                final_flat_indices = tf.tile(tf.reshape(tf.range(0,batch_size),[-1,1]),[1,topn])*beam_size + final_prev_beams

                ### update variables for the next iteration 
                final_flat_symbols = tf.reshape(final_symbols, [-1,1])
           
                #### compute best-n candidates now 
                top_n_prev_outputs = tf.gather( prev_best_outputs_for_final, tf.reshape(final_flat_indices,[-1]) )
                final_outputs = tf.concat(1,[top_n_prev_outputs,final_flat_symbols])

        return (tf.reshape(final_outputs,[-1,topn,self.max_sequence_length]),final_scores/self.max_sequence_length)

    ## Given source sequences, and target language, predict character ids sequences in target_lang
    ## Explanation same as that of seq_loss
    ## Output is tensorflow op
    #def transliterate(self, source_lang, sequences, sequence_lengths, target_lang):

    #    batch_size = tf.shape(sequences)[0]
    #    initial_state, enc_output = self.compute_hidden_representation(sequences,sequence_lengths, source_lang,tf.constant(1.0))
    #    state = tf.matmul(initial_state,self.state_adapt_W) + self.state_adapt_b 
    #    outputs=[]

    #    cell = rnn_cell.DropoutWrapper(self.decoder_cell[target_lang],output_keep_prob=tf.constant(1.0))

    #    for i in range(self.max_sequence_length):
    #        if(i==0):
    #            #current_emb = tf.reshape(tf.tile(self.decoder_input[target_lang],[batch_size,1]),[-1,self.embedding_size])
    #            x = tf.expand_dims(
    #                    tf.nn.embedding_lookup(self.embed_outW[target_lang],self.mapping[target_lang].get_index(Mapping.Mapping.GO))+self.embed_outb[target_lang],
    #                    0) 
    #            current_emb = tf.reshape(tf.tile(x,[batch_size,1]),[-1,self.embedding_size])
    #        else:
    #            current_emb = tf.nn.embedding_lookup(self.embed_outW[target_lang],outputs[-1])+self.embed_outb[target_lang]

    #        if i > 0 : tf.get_variable_scope().reuse_variables()

    #        ### compute the context vector 
    #        current_input=None
    #        if self.enc_type=='simple_lstm_noattn':
    #            current_input=current_emb
    #        else: 
    #            ## using the attention mechanism
    #            context=self.compute_attention_context(state,current_emb,enc_output)
    #            current_input=tf.concat(1,[current_emb,context])

    #        # Run one step of the decoder cell. Updates 'state' and store output in 'output'
    #        output = None
    #        with tf.variable_scope('decoder'):
    #            if i > 0 : tf.get_variable_scope().reuse_variables()
    #            output, state = cell(current_input,state)

    #        logit_words = tf.add(tf.matmul(output,self.out_W[target_lang]),self.out_b[target_lang])

    #        output_id = tf.argmax(logit_words,1)
    #        outputs += [output_id]

    #    return tf.transpose(tf.pack(outputs), perm=[1,0])

    ## Arguments: target_sequences, and predicted_sequences. Both should have same size, and must be numpy arrays

    ## Get a monolingual optimizer for 'lang' language
    ## sequences, sequence masks: tensors of shape: [batch_size, max_sequence lengths]
    ## sequence_lengths: tensor of shape: [batch_sizes]
    #def get_mono_optimizer(self,learning_rate,lang,sequences,sequence_masks,sequence_lengths,dropout_keep_prob):
    #    hidden_representation, enc_output =  self.compute_hidden_representation(sequences,sequence_lengths,lang,dropout_keep_prob)
    #    loss = self.seq_loss(sequences,sequence_masks,lang,hidden_representation,enc_output,dropout_keep_prob)
    #    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    #    return optimizer

    ## Optimizer to minimize mean squared differences of hidden representation of same word from different languages
    ## (mean taken over all batch_size x embedding_size elements)
    #def get_parallel_difference_optimizer(self,learning_rate,lang1,sequences1,sequence_lengths1,lang2,sequences2,sequence_lengths2,dropout_keep_prob):
    #    hidden_representation1, enc_output1 = self.compute_hidden_representation(sequences1,sequence_lengths1,lang1,dropout_keep_prob)
    #    hidden_representation2, enc_output2 = self.compute_hidden_representation(sequences2,sequence_lengths2,lang2,dropout_keep_prob)

    #    squared_difference = tf.squared_difference(hidden_representation1,hidden_representation2)
    #    mean_squared_difference = tf.reduce_mean(squared_difference)

    #    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(mean_squared_difference)
    #    return optimizer

    ## Optimizer to
    ##   minimize L12 + L21
    ## where,
    ##   L12 =  (lang1->lang2 negative log likelihood)
    ##   L21 =  (lang2->lang1 negative log likelihood)

    #def get_parallel_bi_optimizer(self,learning_rate,lang1,sequences1,sequence_masks1,sequence_lengths1,lang2,sequences2,sequence_masks2,sequence_lengths2,dropout_keep_prob):
    #    hidden_representation1, enc_output1 = self.compute_hidden_representation(sequences1,sequence_lengths1,lang1,dropout_keep_prob)
    #    hidden_representation2, enc_output2 = self.compute_hidden_representation(sequences2,sequence_lengths2,lang2,dropout_keep_prob)

    #    # L12
    #    loss12 = self.seq_loss(sequences2,sequence_masks2,lang2,hidden_representation1, enc_output1,dropout_keep_prob)

    #    # L21
    #    loss21 = self.seq_loss(sequences1,sequence_masks1,lang1,hidden_representation2, enc_output2,dropout_keep_prob)

    #    total_loss=loss12+loss21

    #    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
    #    return optimizer

    ## Optimizer to
    ##   minimize L12 + L21 + D
    ## where,
    ##   L12 =  (lang1->lang2 negative log likelihood)
    ##   L21 =  (lang2->lang1 negative log likelihood)
    ##     D =  mean squared differences of hidden representation of same word from different languages

    #def get_parallel_all_optimizer(self,learning_rate,lang1,sequences1,sequence_masks1,sequence_lengths1,lang2,sequences2,sequence_masks2,sequence_lengths2,dropout_keep_prob):
    #    hidden_representation1, enc_output1 = self.compute_hidden_representation(sequences1,sequence_lengths1,lang1,dropout_keep_prob)
    #    hidden_representation2, enc_output2 = self.compute_hidden_representation(sequences2,sequence_lengths2,lang2,dropout_keep_prob)

    #    # L12
    #    loss12 = self.seq_loss(sequences2,sequence_masks2,lang2,hidden_representation1, enc_output1,dropout_keep_prob)

    #    # L21
    #    loss21 = self.seq_loss(sequences1,sequence_masks1,lang1,hidden_representation2, enc_output2,dropout_keep_prob)

    #    # D
    #    squared_difference = tf.squared_difference(hidden_representation1,hidden_representation2)
    #    mean_squared_difference = tf.reduce_mean(squared_difference)

    #    total_loss=loss12+loss21+mean_squared_difference

    #    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
    #    return optimizer
