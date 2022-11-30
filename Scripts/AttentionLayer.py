import tensorflow as tf
from tensorflow.keras.layers import *

class CustomAttention(tf.keras.Model):
    '''
    LSTM attention layer
    '''
    def __init__(self, units):
        super().__init__()
        self.W1 = Dense(units, use_bias=False)
        self.W2 = Dense(units, use_bias=False)
        self.V = Dense(1)

    def call(self, features, hidden):
        # features = (batch, timesteps, rnn units)
        # hidden = (batch, rnn units)
        # hidden_with_time_axis = (batch, 1, rnn units)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score = (batch, timesteps, attention units)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # attention_weights = (batch, timesteps, 1)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context vector = (batch. rnn units)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class CustomAttentionTF(tf.keras.Model):
    '''
    LSTM attention model using TF2.0 built-in attention layer
    '''
    def __init__(self, units):
        super().__init__()
        self.W1 = Dense(units, use_bias=False)
        self.W2 = Dense(units, use_bias=False)
        self.attention = AdditiveAttention()

    def call(self, value, query):
        # query = (batch, 1, rnn units)
        # value = (batch, timesteps, rnn units)
        # w1_query = (batch, 1, attention units)
        w1_query = self.W1(tf.expand_dims(query, axis=1))

        # w2_key = (batch, timesteps, attention units)
        w2_key = self.W2(value)

        context_vector, attention_weights = self.attention(
                inputs=[w1_query, value, w2_key],
                return_attention_scores = True,
        )

        return context_vector, attention_weights      

class LSTMAttentionLayer(tf.keras.layers.Layer):
    '''
    LSTM attention layer using TF2.0 built-in attention layer
    '''
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.W1 = Dense(self.units, use_bias=False)
        self.W2 = Dense(self.units, use_bias=False)
        self.attention = AdditiveAttention()
    
    def compute_output_shape(self, input_shape):
        return input_shape
 
    def call(self, value, query):
        # query = (batch, 1, rnn units)
        # value = (batch, timesteps, rnn units)
        # w1_query = (batch, 1, attention units)
        w1_query = self.W1(tf.expand_dims(query, axis=1))

        # w2_key = (batch, timesteps, attention units)
        w2_key = self.W2(value)

        context_vector, attention_weights = self.attention(
                inputs=[w1_query, value, w2_key],
                return_attention_scores = True,
        )

        return context_vector, attention_weights      
    
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

class ChannelAttention(tf.keras.layers.Layer):
    '''
    Channel attention layer from CBAM
    '''
    def __init__(self, ratio):
        super(ChannelAttention, self).__init__()
        self.ratio = ratio

    def build(self, input_shape):
        channel = input_shape[-1]

        self.shared_layer_one = Dense(
                channel // self.ratio, 
                activation='relu', 
                kernel_initializer='he_normal', 
                use_bias=True, 
                bias_initializer='zeros',
                )
        
        self.shared_layer_two = Dense(
                channel, 
                kernel_initializer='he_normal', 
                use_bias=True, 
                bias_initializer='zeros',
                )

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        channel = inputs.get_shape().as_list()[-1]
        
        avg_pool = GlobalAveragePooling2D()(inputs)
        avg_pool = Reshape((1, 1, channel))(avg_pool)
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)

        max_pool = GlobalMaxPooling2D()(inputs)
        max_pool = Reshape((1, 1, channel))(max_pool)
        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)

        attention = Add()([avg_pool,max_pool])
        attention = Activation('sigmoid')(attention)

        return Multiply()([inputs, attention])
    
    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config

class SpatialAttention(tf.keras.layers.Layer):
    '''
    Spatial attention layer in CBAM
    '''
    def __init__(self, kernel_size):
      super().__init__()
      self.kernel_size = kernel_size
      
    def build(self, input_shape):
        self.conv2d = Conv2D(
                filters=1, 
                kernel_size=self.kernel_size, 
                strides=1, 
                padding='same', 
                activation='sigmoid', 
                kernel_initializer='he_normal', 
                use_bias=False,
                )

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, inputs):
        
        avg_pool = Lambda(lambda x: tf.keras.backend.mean(x, axis=-1, keepdims=True))(inputs)
        max_pool = Lambda(lambda x: tf.keras.backend.max(x, axis=-1, keepdims=True))(inputs)
        attention = Concatenate(axis=-1)([avg_pool, max_pool])
        attention = self.conv2d(attention)

        return Multiply()([inputs, attention]) 
    
    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config

class SimAM(tf.keras.layers.Layer):
    '''
    Implementation SimAM
    '''
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd
    
    def build(self, input_shape):
        self.height = input_shape[1]
        self.width = input_shape[2]
        self.channels = input_shape[3]
        super().build(input_shape)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, inputs):
        n = self.height * self.width - 1
        d = tf.math.square(inputs - tf.math.reduce_mean(inputs, axis=(1, 2), keepdims=True))
        v = tf.reduce_sum(d, axis=(1,2), keepdims=True) / n
        E_inv = d / (4 * (v + self.lambd)) + 0.5
       
        return Multiply()([inputs, Activation('sigmoid')(E_inv)])
     
    def get_config(self):
        config = super().get_config()
        config.update({"lambd": self.lambd})
        return config
        
