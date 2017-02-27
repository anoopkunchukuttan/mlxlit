import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

class Encoder(object):

    def encode(self, sequences, sequence_lengths):
        '''
            sequences: batch_size * max_length
            sequence_lengths: batch_size
        '''
        pass 

    def get_output_size(self): 
        pass 

    def get_state_size(self): 
        pass 

class SimpleRnnEncoder(Encoder):

    def __init__(self,embedding_size,max_sequence_length,rnn_size):
        self.embedding_size=embedding_size
        self.rnn_size=rnn_size 
        self.max_sequence_length=max_sequence_length 
        self.encoder_cell = rnn_cell.BasicLSTMCell(rnn_size)

    def encode(self, sequence_embeddings, sequence_lengths,dropout_keep_prob):
        x = tf.transpose(sequence_embeddings,[1,0,2])
        x = tf.reshape(x,[-1,self.embedding_size])
        x = tf.split(0,self.max_sequence_length,x,name='encoder_input')
        cell=rnn_cell.DropoutWrapper(self.encoder_cell,output_keep_prob=dropout_keep_prob)
        enc_outputs, states = rnn.rnn(cell, x, dtype = tf.float32, sequence_length = sequence_lengths)
        return states, enc_outputs

    def get_output_size(self): 
        return self.encoder_cell.output_size

    def get_state_size(self): 
        return self.encoder_cell.state_size 

class BidirectionalRnnEncoder(Encoder):

    def __init__(self,embedding_size,max_sequence_length,rnn_size):
        self.embedding_size=embedding_size
        self.rnn_size=rnn_size 
        self.max_sequence_length=max_sequence_length 
        self.fw_encoder_cell = rnn_cell.BasicLSTMCell(rnn_size)
        self.bw_encoder_cell = rnn_cell.BasicLSTMCell(rnn_size)

    def encode(self, sequence_embeddings, sequence_lengths,dropout_keep_prob):
        x = tf.transpose(sequence_embeddings,[1,0,2])
        x = tf.reshape(x,[-1,self.embedding_size])
        x = tf.split(0,self.max_sequence_length,x,name='encoder_input')
        fw_cell=rnn_cell.DropoutWrapper(self.fw_encoder_cell,output_keep_prob=dropout_keep_prob)
        bw_cell=rnn_cell.DropoutWrapper(self.bw_encoder_cell,output_keep_prob=dropout_keep_prob)
        enc_outputs, states, _ = rnn.bidirectional_rnn(fw_cell, bw_cell, x, dtype = tf.float32, sequence_length = sequence_lengths)
        return states, enc_outputs

    def get_output_size(self): 
        #return self.fw_encoder_cell.output_size + self.bw_encoder_cell.output_size
        return 2*self.rnn_size 

    def get_state_size(self): 
        return self.fw_encoder_cell.state_size 

class CNNEncoder(Encoder):

    def __init__(self,embedding_size,max_sequence_length,filter_sizes,num_filters):
        self.filter_sizes=filter_sizes
        self.embedding_size=embedding_size
        self.num_filters=num_filters
        self.max_sequence_length=max_sequence_length 
        self.maxpool_width=4

        self.W=[]
        self.b=[]
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                self.W.append(tf.Variable(tf.random_uniform(filter_shape, -0.1, 0.1), name="W"))
                self.b.append(tf.Variable(tf.constant(0.0, shape=[self.num_filters]), name="b"))

    def encode(self, sequence_embeddings, sequence_lengths,dropout_keep_prob):
        input_data = tf.expand_dims(sequence_embeddings, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                conv = tf.nn.conv2d(
                    input_data,
                    self.W[i],
                    strides=[1, 1, self.embedding_size, 1],
                    padding="SAME",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, self.b[i]), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.maxpool_width, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    name="pool")
                pooled_outputs.append(pooled)

        ## encoder output matrix of dimension: [batch,max_sequence_length,len(filter_sizes)*num_filters]
        total_num_filters=self.num_filters*len(self.filter_sizes)
        enc_output_matrix=tf.reshape(tf.squeeze(tf.concat(3,pooled_outputs)),
                [-1,self.max_sequence_length,total_num_filters])

        ## output encoding  and dropout 
        enc_outputs=tf.unpack(tf.transpose(tf.nn.dropout(enc_output_matrix,dropout_keep_prob),[1,0,2]))

        ## final state generation: taking as average of all time step vectors
        def state_gen_func(batch_no): 
            s=tf.slice(enc_output_matrix,[batch_no,0,0],[1,self.max_sequence_length,self.get_output_size()])
            return tf.add_n(tf.unpack(tf.squeeze(s)))/self.max_sequence_length

        batch_size=tf.shape(enc_output_matrix)[0]
        states=tf.map_fn(state_gen_func,tf.range(0,batch_size),dtype=tf.float32)

        return states, enc_outputs 

    def get_output_size(self): 
        return self.num_filters*len(self.filter_sizes)

    def get_state_size(self): 
        return self.num_filters*len(self.filter_sizes)
