import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class FactorizedGRUCell(tf.keras.layers.Layer):
    
    def __init__(self, nf, ns, units, hypo_query_size,
                 temperature=0.001,
                ):
        '''
        nf -- number of Object files
        ns -- number of Dynamics
        units -- dimension of Object files' embedding
        '''
        super(FactorizedGRUCell, self).__init__()
        self.units = units
        self.nf = nf
        self.ns = ns
        
        self.hypo_query_size = hypo_query_size
        self.temperature = temperature
        
    @property
    def state_size(self):
        return tf.TensorShape([self.nf, self.units])
    
    def build(self, input_shape):
        self.i2h_param = self.add_weight(name = 'factorized_gru_i2h',
                                        shape = (self.ns, int(input_shape[-1]), self.units*3),
                                        initializer = 'uniform',
                                        trainable = True)
        self.h2h_param = self.add_weight(name = 'factorized_gru_h2h',
                                        shape = (self.ns, self.units, self.units*3),
                                        initializer = 'uniform',
                                        trainable = True)
        self.prev2query_param = self.add_weight(name='factorized_gru_prev2query',
                                                shape = (self.units, self.hypo_query_size),
                                                initializer = 'uniform',
                                                trainable = True)
        self.current2key_param = self.add_weight(name='factorized_gru_current2key',
                                                shape = (self.units, self.hypo_query_size),
                                                initializer = 'uniform',
                                                trainable = True)
        
    def call(self, inputs, states):
        '''
        inputs of shape (batch_size, nf, input_feature_size)
        states of shape (batch_size, nf, units)
        '''
        
        x = inputs # (batch_size, nf, input_feature_size)
        h, = states # (batch_size, nf, units)
        i2h = self.i2h_param # (ns, input_feature_size,   units*3)
        h2h = self.h2h_param # (ns, units,                units*3)

        preact_i = tf.einsum('bks,jsh->bkjh', x, i2h) # (batch_size, nf, ns, units*3)
        preact_h = tf.einsum('bks,jsh->bkjh', h, h2h) # (batch_size, nf, ns, units*3)

        i_reset, i_input, i_new = tf.split(preact_i, 3, -1) # (batch_size, nf, ns, units)
        h_reset, h_input, h_new = tf.split(preact_i, 3, -1) # (batch_size, nf, ns, units)
        
        reset_gate = tf.sigmoid(i_reset + h_reset) # (batch_size, nf, ns, units)
        input_gate = tf.sigmoid(i_input + h_input) # (batch_size, nf, ns, units)
        new_cell = tf.tanh(i_new + tf.multiply(reset_gate, h_new)) # (batch_size, nf, ns, units)
        
        h_tkj = tf.multiply(tf.expand_dims(h, 2), input_gate) + tf.multiply(new_cell, 1-input_gate) # (batch_size, nf, ns, units)
        
        # Softmax on Gumbel trick
        # Attention on ns Dynamics
        prev2query_param = self.prev2query_param  # (units, hypo_query_size)
        current2key_param= self.current2key_param # (units, hypo_query_size)
        hypo_query = tf.einsum('bks,sh->bkh', h,     prev2query_param)  # (batch_size, nf, hypo_query_size)
        hypo_key = tf.einsum('bkjs,sh->bkjh', h_tkj, current2key_param) # (batch_size, nf, ns, hypo_query_size)
        
        qk = tf.einsum('bkh,bkjh->bkj', hypo_query, hypo_key) # (batch_size, nf, ns)
        r = self.sample_gumbel(tf.shape(qk))
        attention_score = tf.nn.softmax((qk+r) / self.temperature, axis=-1) # (batch_size, nf, ns-softmax)

        h_tkj = h_tkj * tf.expand_dims(attention_score, axis=-1) # (batch_size, nf, ns, units)
        h_tk  = tf.reduce_mean(h_tkj, axis=2)
        return h_tk, (h_tk)

    @staticmethod
    def sample_gumbel(shape, eps=1e-20):
        '''
        sample gumbel random variables with given shape
        '''
        U = tf.random.uniform(shape, minval=0, maxval=1)
        return -tf.math.log(-tf.math.log(U + eps) + eps)


class SCOFFCell(tf.keras.layers.Layer):
    def __init__(self, nf, topk, ns, hypo_query_size,
                 OF_comp_key_size, OF_comp_value_size, inp_heads,  inp_keep_prob,
                 OF_comm_key_size, OF_comm_value_size, comm_heads, comm_keep_prob,
                 temperature=0.001):
        super(SCOFFCell, self).__init__()
        self.nf = nf
        self.topk = topk
        self.ns = ns
        self.units = OF_comm_value_size * comm_heads
        self.hypo_query_size = hypo_query_size
        self.temperature = temperature
        
        self.OF_comp_query_size = OF_comp_key_size
        self.OF_comp_key_size   = OF_comp_key_size
        self.OF_comp_value_size = OF_comp_value_size
        self.input_keep_prob    = inp_keep_prob
        self.num_input_heads    = inp_heads
        
        self.OF_comm_query_size = OF_comm_key_size
        self.OF_comm_key_size   = OF_comm_key_size
        self.OF_comm_value_size = OF_comm_value_size
        self.comm_keep_prob     = comm_keep_prob
        self.num_comm_heads     = comm_heads
        
    @property
    def state_size(self):
        return tf.TensorShape([self.nf, self.units])
    
    def build(self, input_shape):
        
        self.OF_comp_query = tf.keras.layers.Dense(units=self.num_input_heads*self.OF_comp_query_size, activation=None, use_bias=False)
        self.OF_comp_key   = tf.keras.layers.Dense(units=self.num_input_heads*self.OF_comp_key_size,   activation=None, use_bias=False)
        self.OF_comp_value = tf.keras.layers.Dense(units=self.num_input_heads*self.OF_comp_value_size, activation=None, use_bias=False)
        self.input_attention_dropout = tf.keras.layers.Dropout(rate = 1-self.input_keep_prob)
        
        self.rnn_cell = FactorizedGRUCell(nf=self.nf, 
                                          ns=self.ns, 
                                          units = self.units, 
                                          hypo_query_size = self.hypo_query_size, 
                                          temperature = self.temperature)
        
        self.OF_comm_query = tf.keras.layers.Dense(units=self.num_comm_heads*self.OF_comm_query_size, activation=None, use_bias=False)
        self.OF_comm_key   = tf.keras.layers.Dense(units=self.num_comm_heads*self.OF_comm_key_size,   activation=None, use_bias=False)
        self.OF_comm_value = tf.keras.layers.Dense(units=self.num_comm_heads*self.OF_comm_value_size, activation=None, use_bias=False)
        self.comm_attention_dropout = tf.keras.layers.Dropout(rate = 1-self.comm_keep_prob)
        
    def call(self, inputs, states, training=False):
        '''
        inputs of shape (batch_size, input_feature_size)
        states of shape (batch_size, nf, unitf)
        '''
        hs, =states # hs of shape (batch_size, nf, units)
        h_old = hs
        xx = tf.expand_dims(inputs, 1)
        ak, mask = self.OF_compete_step(xx, hs, training=training) # ak/mask of shape (batch_size, nf, OF_compete_value_size/1)
        
        _, h_tk = self.rnn_cell(ak, (hs,))
        h_tk = tf.stop_gradient(h_old*(1-mask)) + h_tk*mask
        
        h_update = self.OF_communicate_step(h_tk, mask, training=training) # (batch_size, nf, OF_comm_value_size = units)
        h_update = h_update*mask + h_old*(1-mask) #?Do masked OFs participate in communication?
        
        return tf.reshape(h_update, [tf.shape(inputs)[0], self.units*self.nf]), (h_update)
        
    def OF_compete_step(self, inputs, hs, training=False):
        '''
        OFs softmax to read from inputs
        xx of shape (batch_size, 1,  input_feature_size)
        hs of shape (batch_size, nf, unitf)
        '''
        qk = self.OF_comp_query(hs)     # qk of shape (batch_size, nf, OF_compete_query_size*inp_heads)
        kt = self.OF_comp_key(inputs)   # kt of shape (batch_size, 1,  OF_compete_key_size*inp_heads)
        vt = self.OF_comp_value(inputs) # vt of shape (batch_size, 1,  OF_compete_value_size*inp_heads)
        
        kt = tf.stack(tf.split(kt, num_or_size_splits=self.num_input_heads, axis=-1), axis=1) # (batch_size, inp_heads,  1, OF_comm_key_size)
        vt = tf.stack(tf.split(vt, num_or_size_splits=self.num_input_heads, axis=-1), axis=1) # (batch_size, inp_heads,  1, OF_comm_value_size)
        qk = tf.stack(tf.split(qk, num_or_size_splits=self.num_input_heads, axis=-1), axis=1) # (batch_size, inp_heads, nf, OF_comm_query_size)
        vt = tf.reduce_mean(vt, axis=1) # (batch_size, 1, OF_comm_value_size)
        
        att = tf.matmul(qk, kt, transpose_b=True)/np.sqrt(self.OF_comp_key_size) # att (batch_size, inp_heads, nf, 1)
        att = tf.reduce_mean(att, axis=1) # (batch_size, nf, 1)
        att_prob = tf.nn.softmax(att, axis = -2)  # att_prob (batch_size, nf-softmax, 1)
        
        # hard select topK OFs to update
        if self.topk < self.nf:
            signal_attention = tf.reshape(att_prob, [-1, self.nf])
            topk = tf.math.top_k(signal_attention, self.topk)
            indices = topk.indices
            mesh = tf.meshgrid( tf.range(indices.shape[1]), tf.range(tf.shape(indices)[0]) )[1]
            full_indices = tf.reshape(tf.stack([mesh, indices], axis=-1), [-1,2])
            sparse_tensor = tf.sparse.SparseTensor(indices=tf.cast(full_indices, tf.int64),
                                                  values=tf.ones(tf.shape(full_indices)[0]),
                                                  dense_shape=[tf.shape(signal_attention)[0],self.nf])
            sparse_tensor = tf.sparse.reorder(sparse_tensor)
            mask_ = tf.sparse.to_dense(sparse_tensor)
            mask  = tf.reshape(mask_, [-1, self.nf, 1]) #(batch_size, nf, 1)
        else:
            mask  = tf.ones_like(att_prob) #(batch_size, nf, 1)
        # End hard select
        
        att_prob = self.input_attention_dropout(att_prob, training=training)
        ak = tf.matmul(att_prob, vt) # ak of shape (batch_size, nf, OF_compete_value_size)
        
        ak = ak*mask
        return ak, mask
        
    def OF_communicate_step(self, h_tk, mask, training=False):
        '''
        OFs communicate softmax
        h_tk of shape (batch_size, nf, units)
        mask of shape (batch_size, nf, 1)
        '''
        qhat = self.OF_comm_query(h_tk) # (batch_size, nf, OF_comm_query_size*comm_heads)
        khat = self.OF_comm_key(h_tk)   # (batch_size, nf, OF_comm_key_size*comm_heads)
        vhat = self.OF_comm_value(h_tk) # (batch_size, nf, OF_comm_value_size*comm_heads = units)
        
        khat = tf.stack(tf.split(khat, num_or_size_splits=self.num_comm_heads, axis=-1), axis=1) # (batch_size, comm_heads, nf, OF_comm_key_size)
        vhat = tf.stack(tf.split(vhat, num_or_size_splits=self.num_comm_heads, axis=-1), axis=1) # (batch_size, comm_heads, nf, OF_comm_value_size)
        qhat = tf.stack(tf.split(qhat, num_or_size_splits=self.num_comm_heads, axis=-1), axis=1) # (batch_size, comm_heads, nf, OF_comm_query_size)
        
        att = tf.matmul(qhat, khat, transpose_b=True)/np.sqrt(self.OF_comm_key_size) # (batch_size, comm_heads, nf, nf)
        att_prob = tf.nn.softmax(att, axis = -1) # (batch_size, comm_heads, nf, nf-softmax)
        comm_mask = tf.expand_dims(mask, axis=1) # (batch_size, 1, nf, 1)
        att_prob = att_prob*comm_mask
        att_prob = self.comm_attention_dropout(att_prob, training=training)
        delta_h = tf.matmul(att_prob, vhat) # (batch_size, comm_heads, nf, OF_comm_value_size)
        delta_h = tf.reshape(tf.transpose(delta_h, [0,2,1,3]), [-1, self.nf, self.units]) # (batch_size, nf, units)
        out = h_tk+delta_h
        return out
             

