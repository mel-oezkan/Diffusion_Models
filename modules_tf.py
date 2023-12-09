
import tensorflow as tf
from tensorflow.keras import layers 
from tensorflow.keras.activations import gelu
from tensorflow.nn import silu



class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, channels: int, size):
        super(SelfAttention, self).__init__()

        self.channels = channels
        self.size = size
        self.mha = layers.MultiHeadAttention(num_heads=4, key_dim=channels, batch_first=True)
        self.ln = layers.LayerNormalization() # expected shape: (batch_size, channels)
        self.ff_self = tf.keras.Sequential([
            layers.LayerNorm(),
            layers.Dense(channels),
            layers.Lambda(gelu),
            layers.Dense(channels),
        ])

    def call(self, x):
        x = x.reshape(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value

        return attention_value.swapaxes(2, 1).reshape(-1, self.channels, self.size, self.size)


class DoubleConv(tf.keras.layers.Layer):
    def __init__(
            self, 
            out_channels: int, 
            mid_channels=None, 
            residual=False
        ):

        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = tf.keras.Sequential([
            layers.Conv2D(mid_channels, kernel_size=3, padding='same', use_bias=False),
            layers.GroupNormalization(groups=1, scale=False, epsilon=1e-5),
            layers.Lambda(gelu),

            layers.Conv2D(out_channels, kernel_size=3, padding='same', use_bias=False),
            layers.GroupNormalization(groups=1, scale=False, epsilon=1e-5), # expected shape: (out_channels)
        ])

    def call(self, x):
        # change output if residual
        if self.residual:
            return gelu(x + self.double_conv(x))

        return self.double_conv(x)



class Down(tf.keras.layers.Layer):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            emb_dim=256
        ):
        
        super().__init__()
        self.maxpool_conv = tf.keras.Sequential([
            layers.MaxPooling2D((2,2)),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        ])

        self.emb_layer = tf.keras.Sequential([
            layers.Lambda(silu),
            layers.Dense(out_channels),
        ])

    def call(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb



class UNet(tf.keras.Model):
    def __init__(self, c_in=3, c_out=3, time_dim=256):
        super().__init__()
#         self.device = device
#         self.time_dim = time_dim
#         self.inc = DoubleConv(c_in, 64)
#         self.down1 = Down(64, 128)
#         self.sa1 = SelfAttention(128, 32)
#         self.down2 = Down(128, 256)
#         self.sa2 = SelfAttention(256, 16)
#         self.down3 = Down(256, 256)
#         self.sa3 = SelfAttention(256, 8)

#         self.bot1 = DoubleConv(256, 512)
#         self.bot2 = DoubleConv(512, 512)
#         self.bot3 = DoubleConv(512, 256)

#         self.up1 = Up(512, 128)
#         self.sa4 = SelfAttention(128, 16)
#         self.up2 = Up(256, 64)
#         self.sa5 = SelfAttention(64, 32)
#         self.up3 = Up(128, 64)
#         self.sa6 = SelfAttention(64, 64)
#         self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (tf.range(0, channels, delta=2, dtype=float) / channels)

        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

#     def forward(self, x, t):
#         t = t.unsqueeze(-1).type(torch.float)
#         t = self.pos_encoding(t, self.time_dim)

#         x1 = self.inc(x)
#         x2 = self.down1(x1, t)
#         x2 = self.sa1(x2)
#         x3 = self.down2(x2, t)
#         x3 = self.sa2(x3)
#         x4 = self.down3(x3, t)
#         x4 = self.sa3(x4)

#         x4 = self.bot1(x4)
#         x4 = self.bot2(x4)
#         x4 = self.bot3(x4)

#         x = self.up1(x4, x3, t)
#         x = self.sa4(x)
#         x = self.up2(x, x2, t)
#         x = self.sa5(x)
#         x = self.up3(x, x1, t)
#         x = self.sa6(x)
#         output = self.outc(x)
#         return output