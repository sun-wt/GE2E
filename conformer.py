import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Conv1D, Dropout, BatchNormalization
from tensorflow.keras.models import Model

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout_rate, max_position=500):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.positional_encoding = tf.keras.layers.Embedding(input_dim=max_position, output_dim=embed_dim)
        self.layernorm = LayerNormalization()
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs):
        positions = tf.range(start=0, limit=tf.shape(inputs)[1], delta=1)
        positional_embedding = self.positional_encoding(positions)
        attn_output = self.attention(inputs + positional_embedding, inputs + positional_embedding)
        x = self.layernorm(inputs + attn_output)
        return self.dropout(x)

class ConvolutionModule(tf.keras.layers.Layer):
    def __init__(self, embed_dim, kernel_size_list, dropout_rate):
        super().__init__()
        self.depthwise_convs = [
            Conv1D(filters=embed_dim, kernel_size=ks, padding="same", groups=embed_dim)
            for ks in kernel_size_list
        ]
        self.batchnorm = BatchNormalization()
        self.pointwise_conv2 = Conv1D(filters=embed_dim, kernel_size=1, activation="linear")
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs):
        x = inputs
        for depthwise_conv in self.depthwise_convs:
            x = depthwise_conv(x)
        x = self.batchnorm(x)
        x = tf.nn.swish(x)
        x = self.pointwise_conv2(x)
        return self.dropout(inputs + x)

class FeedForwardModule(tf.keras.layers.Layer):
    def __init__(self, embed_dim, expansion_factor, dropout_rate):
        super().__init__()
        self.linear1 = Dense(embed_dim * expansion_factor, activation="swish", kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dropout1 = Dropout(dropout_rate)
        self.linear2 = Dense(embed_dim)
        self.dropout2 = Dropout(dropout_rate)
        self.layernorm = LayerNormalization()

    def call(self, inputs):
        x = self.linear1(inputs)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return self.layernorm(inputs + x)

class ConformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ffn_expansion_factor, conv_kernel_size_list, dropout_rate):
        super().__init__()
        self.ffn1 = FeedForwardModule(embed_dim, ffn_expansion_factor, dropout_rate)
        self.mhsa = MultiHeadSelfAttention(embed_dim, num_heads, dropout_rate)
        self.conv = ConvolutionModule(embed_dim, conv_kernel_size_list, dropout_rate)
        self.ffn2 = FeedForwardModule(embed_dim, ffn_expansion_factor, dropout_rate)
        self.layernorm = LayerNormalization()

    def call(self, inputs):
        x = self.ffn1(inputs)
        x = self.mhsa(x)
        x = self.conv(x)
        x = self.ffn2(x)
        return self.layernorm(x)

def build_conformer_encoder(input_dim, num_layers, embed_dim, num_heads, ffn_expansion_factor, conv_kernel_size_list, dropout_rate):
    inputs = Input(shape=(None, input_dim))
    x = inputs
    for _ in range(num_layers):
        x = ConformerBlock(embed_dim, num_heads, ffn_expansion_factor, conv_kernel_size_list, dropout_rate)(x)
    return Model(inputs, x, name="ConformerEncoder")
